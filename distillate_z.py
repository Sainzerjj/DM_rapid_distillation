#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import argparse
import importlib
from distill_enc.v_diffusion import make_beta_schedule
from distill_enc.train_utils import *
from torch.utils.tensorboard import SummaryWriter
from distill_enc.models.utils import create_model, get_score_fn
from distill_enc.configs.vp import cifar10_ddpmpp_continuous
from distill_enc.models import ddpm, ncsnv2, ncsnpp

from templates import *
from templates_latent import *
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, default="cifar_u")
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, default="cifar_exp")
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, default="distill_x")
    parser.add_argument("--base_checkpoint", help="Path to base checkpoint.", type=str, default="./checkpoints/cifar_exp/distill_v/checkpoint_1000.pt")  #
    parser.add_argument("--gamma", help="Gamma factor for SNR weights.", type=float, default=0.9)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="./checkpoints/cifar_exp/distill_v/checkpoint_1000.pt")   # ./checkpoints/cifar_exp/distill_x/checkpoint_50.pt
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=6400000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=64)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=2e-4)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=1500)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--speed_up", type=int, default=100)
    return parser

def distill_model(args, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    need_student_ema = True
    if args.scheduler.endswith("SWA"):
        need_student_ema = False

    config = cifar10_ddpmpp_continuous.get_config()
    # print(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda")
    num_iters = args.num_iters
    train_dataset = test_dataset = InfinityDataset(make_dataset(), num_iters)

    len(train_dataset), len(test_dataset)

    img, anno = train_dataset[0]

    teacher_ema = create_model(config).to(device)

    image_size = [64,3,32,32]
    # image_size = teacher_ema.image_size

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    ckpt = torch.load(args.base_checkpoint)
    teacher_ema.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    del ckpt

    # teacher_ema = create_model(config).to(device)
    # image_size = [32,3,32,32]
    # teacher_ema, ema = get_Diffusion_Model(config, args.base_checkpoint)
    # ema.copy_to(teacher_ema.parameters())
    # n_timesteps = 1000
    # time_scale = 1
    print(f"Num timesteps: {n_timesteps}, time scale: {time_scale}.")

    def make_scheduler():
        M = importlib.import_module("distill_enc.train_utils")
        D = getattr(M, args.scheduler)
        return D()

    

    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("distill_enc.v_diffusion")
        D = getattr(M, args.diffusion)
        r = D(model, betas, time_scale=time_scale)
        r.gamma = args.gamma
        return r

    teacher_ema_diffusion = make_diffusion(teacher_ema, n_timesteps, time_scale, device)
    # img = make_visualization(teacher_ema_diffusion, device, image_size, need_tqdm=True, eta=0, clip_value=4)
    # student = create_model(config).to(device)
    conf = cifar10_autoenc()
    student = conf.make_model_conf().make_model().to(device)
    if need_student_ema:
        student_ema = copy.deepcopy(student).to(device)
        # student_ema = create_model(config).to(device)
    else:
        student_ema = None

    scheduler = make_scheduler()
    optimizerG = optim.Adam(student.encoder.parameters(), lr=2e-4, betas = (0.5, 0.9))
        
    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        student.load_state_dict(ckpt["G"])
        student_ema.load_state_dict(ckpt["G"])
        student.encoder.load_state_dict(ckpt["encoder"])
        optimizerG.load_state_dict(ckpt["optimizerG"])
        del ckpt
    distillation_model = DiffusionDistillation(scheduler)
    # distill_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    # if args.checkpoint_to_continue == "":
    #     init_ema_model(teacher_ema, student, device)
    #     init_ema_model(teacher_ema, student_ema, device)
    #     print("Teacher parameters copied.")
    # else:
    #     print("Continue training...")
    speed_up = args.speed_up
    while teacher_ema_diffusion.num_timesteps >= 100:
        print(teacher_ema_diffusion.num_timesteps)
        distill_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        student_diffusion = make_diffusion(student, teacher_ema_diffusion.num_timesteps // speed_up, teacher_ema_diffusion.time_scale * speed_up, device)
        student_diffusion.encoder = student.encoder
        if need_student_ema:
            student_ema_diffusion = make_diffusion(student_ema, teacher_ema_diffusion.num_timesteps // speed_up, teacher_ema_diffusion.time_scale * speed_up, device)

        on_iter = make_iter_callback(student_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_interval, args.ckpt_interval, False)
        distillation_model.train_student(distill_train_loader, teacher_ema_diffusion, student_diffusion, student_ema, optimizerG, args.lr, speed_up, device, make_extra_args=make_condition, on_iter=on_iter)
        
        # ckpt = torch.load("./checkpoints/cifar_exp/distill_v/checkpoint_{}.pt".format(n_timesteps//2))
        # student.load_state_dict(ckpt["G"])
        # student_ema.load_state_dict(ckpt["G"])
        # teacher_ema.load_state_dict(ckpt["G"])
        # n_timesteps = ckpt["n_timesteps"]
        # time_scale = ckpt["time_scale"]
        # del ckpt

        # teacher_ema_diffusion = make_diffusion(teacher_ema, n_timesteps, time_scale, device)
        
    print("Finished.")
#teacher_ema_diffusion

def make_dataset():
    return CifarWrapper(dataset_dir="./data")

class CifarWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset_dir):
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()
        ])
        self.dataset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    distill_model(args, make_dataset)