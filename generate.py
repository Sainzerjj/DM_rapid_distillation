if __name__ == "__main__":
    import os
    import json
    import math
    import uuid
    import torch
    from tqdm import trange
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from v_diffusion import *
    from argparse import ArgumentParser
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    from v_diffusion.metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
    from torchvision.utils import save_image

    parser = ArgumentParser()
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./checkpoints", type=str)
    parser.add_argument("--chkpt-path", default="./checkpoints/teacher/v_diffusion_1200.pt", type=str)
    parser.add_argument("--save-dir", default="./images/eval", type=str)
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--use-ema", default=True)
    parser.add_argument("--use-ddim", default=True)
    parser.add_argument("--sample-timesteps", default=16, type=int)
    parser.add_argument("--uncond", default=False)   # action="store_true"
    parser.add_argument("--w-guide", default=0.3, type=float)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--task", default="single_sample", choices=["fid", "single_sample"], type=str)
    # action="store_true"
    args = parser.parse_args()

    dataset = args.dataset
    root = os.path.expanduser("~/datasets")

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    device = torch.device(args.device)

    chkpt_dir = args.chkpt_dir
    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
    folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
    use_ema = args.use_ema
    if use_ema:
        state_dict = torch.load(chkpt_path, map_location=device)["ema"]["shadow"]
    else:
        state_dict = torch.load(chkpt_path, map_location=device)["model"]

    for k in list(state_dict.keys()):
        if k.startswith("module."):  # state_dict of DDP
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

    use_cfg = "class_embed" in {k.split(".")[0] for k in state_dict.keys()}
    multitags = DATA_INFO[dataset].get("multitags", False)
    if use_cfg:
        num_classes = DATA_INFO[dataset]["num_classes"]
        w_guide = 0. if args.uncond else args.w_guide
    else:
        num_classes = 0
        w_guide = 0

    config_dir = args.config_dir
    with open(os.path.join(config_dir, dataset + ".json")) as f:
        configs = json.load(f)

    diffusion_kwargs = configs["diffusion"]
    logsnr_schedule = diffusion_kwargs.pop("logsnr_schedule")
    logsnr_max = diffusion_kwargs.pop("logsnr_max")
    logsnr_min = diffusion_kwargs.pop("logsnr_min")
    logsnr_fn = get_logsnr_schedule(logsnr_schedule, logsnr_min, logsnr_max)

    diffusion = GaussianDiffusion(
        logsnr_fn=logsnr_fn,
        sample_timesteps=args.sample_timesteps,
        w_guide=w_guide,
        use_ddim=args.use_ddim,
        **diffusion_kwargs)

    model_out_type = diffusion_kwargs.get("model_out_type", "both")
    out_channels = (2 if model_out_type == "both" else 1) * in_channels
    model = UNet(
        out_channels=out_channels,
        num_classes=num_classes,
        multitags=multitags,
        **configs["denoise"],
    )
    model.to(device)

    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    folder_name = folder_name + args.suffix
    save_dir = os.path.join(args.save_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = args.batch_size
    total_size = args.total_size
    num_eval_batches = math.ceil(total_size / batch_size)
    shape = (batch_size, 3, image_res, image_res)

    def save_image_arr(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")
    
    def sample_image(x, path, nrow=10, normalize=True, value_range=(-1., 1.)):
        img = make_grid(x, nrow=nrow, normalize=normalize, value_range=value_range)
        img = img.permute(1, 2, 0)
        _ = plt.imsave(path, img.numpy())

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    uncond = args.uncond
    if multitags:
        labels = DATA_INFO[dataset]["data"](root=args.root, split="all").targets

        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, num_classes), dtype=torch.float32, device=to_device)
                else:
                    yield labels[torch.randint(len(labels), size=(batch_size, ))].float().to(to_device)
    else:
        def get_label_loader(to_device):
            while True:
                if uncond:
                    yield torch.zeros((batch_size, ), dtype=torch.int64, device=to_device)
                else:
                    yield torch.randint(num_classes, size=(batch_size, ), device=to_device) + 1

    label_loader = get_label_loader(to_device=device)

    def random_labels():
        labels = torch.arange(num_classes, dtype=torch.float32) + 1
        repeats = torch.as_tensor([
            (batch_size // num_classes
                + int(i < batch_size % num_classes))
            for i in range(num_classes)])
        labels = labels.repeat_interleave(repeats)
        return labels
    
    if args.task == "single_sample":
        seed_all(args.seed)
        x = diffusion.p_sample(
                    model, shape=shape, device=device,
                    noise=torch.randn(shape, device=device),
                    label= next(label_loader)[:batch_size], # random_labels(),        
                    use_ddim=args.use_ddim
                )
        sample_image(x, "result.jpg")
    else:
        evaluator = Evaluator(dataset=dataset, device=args.device)
        evaluator.istats.reset()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            for i in trange(num_eval_batches):
                if i == num_eval_batches - 1:
                    batch_size = total_size - i * batch_size
                    shape = (batch_size, 3, image_res, image_res)
                x = diffusion.p_sample(
                    model, shape=shape, device=device,
                    noise=torch.randn(shape, device=device),
                    label=next(label_loader)[:batch_size],
                    use_ddim=args.use_ddim
                )
                evaluator.istats(x.to(device))
                to_range_0_1 = lambda x: (x + 1.) / 2.
                for i in range(batch_size):
                    save_image(to_range_0_1(x[i]), f"{save_dir}/{uuid.uuid4()}.jpg")
                # x = x.cpu()
                # x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                # pool.map(save_image_arr, list(x))
                gen_mean, gen_var = evaluator.istats.get_statistics()
                print("fid: {}".format(calc_fd(gen_mean, gen_var, evaluator.target_mean, evaluator.target_var)))
