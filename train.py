#!/usr/bin/env python
# coding: utf-8
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
import argparse
import importlib
from distill_enc.v_diffusion import make_beta_schedule
from distill_enc.moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from distill_enc.train_utils import *
from distill_enc.configs.vp import cifar10_ddpmpp_continuous
from templates import *
from templates_latent import *
# python ./train.py --module cifar_u --name cifar_exp --dname base_0 
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, default="cifar_u")
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, default="cifar_exp")
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, default="base_v")
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="./checkpoints/cifar_exp/base_v/checkpoint_z_1000.pt")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1000)   # ./checkpoints/cifar_exp/base_v/checkpoint_z_1000.pt
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=20000000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=64)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=2e-4)  #5e-5
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=3000)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def train_model(args, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    # print(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda")
    train_dataset = test_dataset = InfinityDataset(make_dataset(), args.num_iters)

    print(len(train_dataset)), len(test_dataset)

    img, anno = train_dataset[0]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    def make_sheduler():
        M = importlib.import_module("distill_enc.train_utils")
        D = getattr(M, args.scheduler)
        return D()

    scheduler = make_sheduler()
    
    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("distill_enc.v_diffusion")
        D = getattr(M, args.diffusion)
        return D(model, betas, time_scale=time_scale)

    conf = cifar10_autoenc()
    teacher = conf.make_model_conf().make_model().to(device)
    teacher_ema = conf.make_model_conf().make_model().to(device)
    optimizerG = optim.Adam(teacher.encoder.parameters(), lr=2e-4, betas = (0.5, 0.9))
    # teacher = make_model().to(device)
    # teacher_ema = make_model().to(device)
    # config = cifar10_ddpmpp_continuous.get_config()
    # teacher = create_model(config).to(device)
    # teacher_ema = create_model(config).to(device)
    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        teacher.load_state_dict(ckpt["G"])
        teacher_ema.load_state_dict(ckpt["G"])
        time_scale = ckpt["time_scale"]
        # ema = ckpt["state"]
        N = ckpt["N"]
        teacher.encoder.load_state_dict(ckpt["encoder"])
        optimizerG.load_state_dict(ckpt["optimizerG"])
        del ckpt
        print("Continue training...")
    else:
        print("Training new model...")
        time_scale = 1
    N = 0
    init_ema_model(teacher, teacher_ema)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    teacher_diffusion = make_diffusion(teacher, args.num_timesteps, time_scale, device)
    teacher_diffusion.encoder = teacher.encoder
    teacher_ema_diffusion = make_diffusion(teacher, args.num_timesteps, time_scale, device)

    image_size = [32,3,32,32] # teacher.image_size
    # image_size = teacher.image_size

    on_iter = make_iter_callback(teacher_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_interval, args.ckpt_interval, False)
    diffusion_train = DiffusionTrain(scheduler)
    diffusion_train.train(train_loader, teacher_diffusion, teacher_ema, optimizerG, args.lr, device, N, make_extra_args=make_condition, on_iter=on_iter)
    print("Finished.")


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
    train_model(args, make_dataset)