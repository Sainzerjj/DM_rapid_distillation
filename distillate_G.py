import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5, 6'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

import json
import torch
from torch import nn
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from v_diffusion import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.distributed.elastic.multiprocessing import errors
from functools import partial
import copy
from templates import *
from templates_latent import *

@errors.record
def main(args):

    distributed = args.distributed

    def logger(msg, **kwargs):
        if not distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    root = os.path.expanduser(args.root)
    dataset = args.dataset

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    multitags = DATA_INFO[dataset].get("multitags", False)
    if args.use_cfg:
        num_classes = DATA_INFO[dataset].get("num_classes", 0)
        w_guide = args.w_guide
        p_uncond = args.p_uncond
    else:
        num_classes = 0
        w_guide = 0.
        p_uncond = 0.

    # set seed for all rngs
    seed = args.seed
    seed_all(seed)

    configs_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(configs_path, "r") as f:
        configs: dict = json.load(f)

    # train parameters
    gettr = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    batch_size = gettr("batch_size")
    beta1, beta2 = gettr("beta1"), gettr("beta2")
    weight_decay = gettr("weight_decay")
    lr = gettr("lr")
    epochs = gettr("epochs")
    grad_norm = gettr("grad_norm")
    warmup = gettr("warmup")
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # diffusion parameters
    getdif = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    logsnr_schedule = getdif("logsnr_schedule")
    logsnr_min, logsnr_max = getdif("logsnr_min"), getdif("logsnr_max")
    train_timesteps = getdif("train_timesteps")
    teacher_sample_timesteps = getdif("teacher_sample_timesteps")
    reweight_type = getdif("reweight_type")
    logsnr_fn = get_logsnr_schedule(logsnr_schedule, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    model_out_type = getdif("model_out_type")
    model_var_type = getdif("model_var_type")
    loss_type = getdif("loss_type")
    speed_up = getdif("speed_up")

    teacher_diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        sample_timesteps=teacher_sample_timesteps,
        model_out_type=model_out_type,
        model_var_type=model_var_type,
        reweight_type=reweight_type,
        loss_type=loss_type,
        intp_frac=args.intp_frac,
        w_guide=w_guide,
        p_uncond=p_uncond
    )
    student_diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        sample_timesteps=teacher_sample_timesteps // speed_up,
        model_out_type=model_out_type,
        model_var_type=model_var_type,
        reweight_type=reweight_type,
        loss_type=loss_type,
        intp_frac=args.intp_frac,
        w_guide=w_guide,
        p_uncond=p_uncond
    )
    # denoise parameters
    # currently, model_var_type = "learned" is not supported
    # out_channels = 2 * in_channels if model_var_type == "learned" else in_channels
    out_channels = 2 * in_channels if model_out_type == "both" else in_channels
    _model = UNet(
        out_channels=out_channels,
        num_classes=num_classes,
        multitags=multitags,
        **configs["denoise"])
    
    conf = cifar10_autoenc()
    conf = celeba64d2c_autoenc()
    student_model = conf.make_model_conf().make_model()
    # student_model = UNet(
    #     out_channels=out_channels,
    #     num_classes=num_classes,
    #     multitags=multitags,
    #     **configs["denoise"])
    _model = conf.make_model_conf().make_model()
    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()
        dist.init_process_group("nccl")
        # dist.init_process_group(backend='nccl', init_method='env://', rank=rank)
        rank = dist.get_rank()  # global process id across all node(s)
        local_rank = int(os.environ["LOCAL_RANK"])  # local device id on a single node
        torch.cuda.set_device(local_rank)
        _model.cuda()
        student_model.cuda()
        teacher_model = DDP(_model, device_ids=[local_rank, ])
        # z_model = DDP(z_model, device_ids=[local_rank, ])
        student_model = DDP(student_model, device_ids=[local_rank, ])
        train_device = torch.device(f"cuda:{local_rank}")
        # print(train_device)
    else:
        rank = local_rank = 0  # main process by default
        teacher_model = _model.to(train_device)
        # z_model.to(train_device)
        student_model.to(train_device)

    # z_model = student_model.encoder
    z_model = student_model.module.encoder
    # student_model = copy.deepcopy(teacher_model)
    student_optimizer = AdamW(student_model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    z_optimizer = AdamW(z_model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    # Note1: lr_lambda is used to calculate the **multiplicative factor**
    # Note2: index starts at 0
    scheduler = lr_scheduler.LambdaLR(
        student_optimizer, lr_lambda=lambda t: min((t + 1) / warmup, 1.0)) if warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=batch_size // args.num_accum, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    configs["train"]["epochs"] = epochs
    configs["use_ema"] = args.use_ema
    configs["conditional"] = {
        "use_cfg": args.use_cfg,
        "w_guide": w_guide,
        "p_uncond": p_uncond
    }
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

    chkpt_dir = args.chkpt_dir
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    # keep a record of hyperparameter settings used for current experiment
    with open(os.path.join(chkpt_dir, f"exp_{timestamp}.json"), "w") as f:
        json.dump(configs, f)

    if args.teacher_chkpt_name !="":
        teacher_chkpt_path = os.path.join(chkpt_dir, args.teacher_chkpt_name or f"vdpm_{dataset}.pt")
    student_chkpt_path = os.path.join(chkpt_dir, args.student_chkpt_name or f"vdpm_{dataset}.pt")
    chkpt_intv = args.chkpt_intv
    logger(f"Checkpoint will be saved to {os.path.abspath(teacher_chkpt_path)}", end=" ")
    logger(f"every {chkpt_intv} epoch(s)")

    image_dir = os.path.join(args.image_dir, f"{dataset}")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_intv = args.image_intv
    num_save_images = args.num_save_images
    logger(f"Generated images (x{num_save_images}) will be saved to {os.path.abspath(image_dir)}", end=" ")
    logger(f"every {image_intv} epoch(s)")

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        z_model=z_model,
        student_optimizer=student_optimizer,
        z_optimizer=z_optimizer,
        teacher_diffusion=teacher_diffusion,
        student_diffusion=student_diffusion,
        speed_up=speed_up,
        timesteps=train_timesteps,
        epochs=epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        use_cfg=args.use_cfg,
        use_ema=args.use_ema,
        grad_norm=grad_norm,
        num_accum=args.num_accum,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=image_intv,
        num_save_images=num_save_images,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    # in case of elastic launch, resume should always be turned on
    resume = args.resume or distributed
    map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
    # trainer.load_checkpoint("student", student_chkpt_path, map_location=map_location)
    try:
        trainer.load_checkpoint("teacher", student_chkpt_path, map_location=map_location)
        if resume:
            # trainer.load_checkpoint("student", student_chkpt_path, map_location=map_location)
            trainer.load_checkpoint("student", student_chkpt_path, map_location=map_location)
        else:
            trainer.load_checkpoint("student", teacher_chkpt_path, map_location=map_location)
    except FileNotFoundError:
        logger("Checkpoint file does not exist!")
        logger("Starting from scratch...")
        

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train_G(
        evaluator,
        student_chkpt_path=student_chkpt_path,
        image_dir=image_dir,
        use_ddim=args.use_ddim,
        sample_bsz=args.sample_bsz
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="celeba")
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=2400, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--weight-decay", default=0., type=float,
                        help="decoupled weight_decay factor in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=1, type=int, help=(
        "number of batches before weight update, a.k.a. gradient accumulation"))
    parser.add_argument("--teacher-sample-timesteps", default=20, type=int, help="number of teacher diffusion steps for sampling")
    parser.add_argument("--train-timesteps", default=10, type=int, help=(
        "number of student diffusion steps for training (0 indicates continuous training)"))
    parser.add_argument("--logsnr-schedule", choices=["linear", "sigmoid", "cosine", "legacy"], default="cosine")
    parser.add_argument("--logsnr-max", default=20., type=float)
    parser.add_argument("--logsnr-min", default=-20., type=float)
    parser.add_argument("--model-out-type", choices=["x_0", "eps", "both", "v"], default="v", type=str)
    parser.add_argument("--model-var-type", choices=["fixed_small", "fixed_large", "fixed_medium"], default="fixed_large", type=str)
    parser.add_argument("--reweight-type", choices=["constant", "snr", "truncated_snr", "alpha2"], default="truncated_snr", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--intp-frac", default=0., type=float)
    parser.add_argument("--use-cfg", default=True, help="whether to use classifier-free guidance")
    parser.add_argument("--w-guide", default=0.3, type=float, help="classifier-free guidance strength")
    parser.add_argument("--p-uncond", default=0.1, type=float, help="probability of unconditional training")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=1, type=int)
    parser.add_argument("--num-save-images", default=32, type=int, help="number of images to generate & save")
    parser.add_argument("--sample-bsz", default=-1, type=int, help="batch size for sampling")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./checkpoints", type=str)
    parser.add_argument("--teacher-chkpt-name", default="teacher/celeba_v_diffusion_enc_240.pt", type=str)
    parser.add_argument("--student-chkpt-name", default="teacher/celeba_v_diffusion_enc_240.pt", type=str)
    parser.add_argument("--chkpt-intv", default=10, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", default=True, help="to resume training from a checkpoint")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", default=True, help="whether to use exponential moving average")
    parser.add_argument("--use-ddim", default=True, help="whether to use DDIM sampler")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")
    # Distillation
    parser.add_argument("--speed_up", type=int, default=2)

    main(parser.parse_args())

    # python train.py --use-ema --use-ddim --use-cfg --eval --resume
    # python -m torch.distributed.run --standalone --nproc_per_node 2 --rdzv_backend c10d distillate_G.py --distributed




