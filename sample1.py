import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import importlib
from distill_enc.v_diffusion import make_beta_schedule
from distill_enc.moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
from distill_enc.train_utils import make_visualization, get_Diffusion_Model, get_data_inverse_scaler
import cv2
from model.unet import BeatGANsEncoderConfig
# from configs.vp import cifar10_ddpmpp_continuous
# from model import ddpm, ncsnv2, ncsnpp
# from model.utils import get_score_fn
# from ddim import generalized_steps, get_beta_schedule, inverse
# from model.utils import create_model
# import numpy as np
from distill_enc.train_utils import *
from templates import *
from templates_latent import *
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, default="cifar_u")
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="./checkpoints/cifar_exp/base_v/checkpoint_z_1000.pt")
    parser.add_argument("--out_file", help="Path to image.", type=str, default="10.png")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=200)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=True)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=4)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
    
    return parser

def sample_images(args):
    device = torch.device("cuda")

    # teacher_ema = make_model().to(device)

    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("distill_enc.v_diffusion")
        D = getattr(M, args.diffusion)
        sampler = "ddpm"
        if args.clipped_sampling:
            sampler = "clipped"
        return D(model, betas, time_scale=time_scale, sampler=sampler)
    
    # config = cifar10_ddpmpp_continuous.get_config()
    # teacher = create_model(config).to(device)
    image_size = [32,3,32,32]
    
    # model_dir = "checkpoints/SDEModels/vp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
    # teacher, ema = get_Diffusion_Model(config, model_dir)
    # ema.copy_to(teacher.parameters())
    # n_timesteps = 1000
    # time_scale = 1
    encoder = None
    conf = cifar10_autoenc()
    teacher = conf.make_model_conf().make_model().to(device)
    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    del ckpt
    print(n_timesteps)
    n_timesteps = 10
    time_scale = 100
    # b0=1e-4
    # bT=2e-2
    # beta = np.linspace(b0, bT, 1000)
    # alpha = 1 - beta
    # alphabar = np.cumprod(alpha)
    # x_1 = inverse(net=teacher, shape=(3,32,32), device=device,beta=beta,alpha=alpha,alphabar=alphabar)
    # x_1 = torch.randn(32, 3, 32, 32, device=device)
    # with torch.no_grad():
    #     # teacher = create_model(config).to(device)
    #     teacher = make_model().to(device)
    #     ckpt = torch.load(model_dir)
    #     teacher.load_state_dict(ckpt["G"])
    #     n_timesteps = ckpt["n_timesteps"]//args.time_scale
    #     time_scale = ckpt["time_scale"]*args.time_scale  
    #     N = ckpt['N']
    #     print(N)
    #     del ckpt
    #     print("Model loaded.")
        # score_fn = get_score_fn(sde, teacher, continuous=True)
    # betas = get_beta_schedule(
    #         beta_schedule="linear",
    #         beta_start= 0.0001 ,                     #    config.model.beta_min//args.num_scales,
    #         beta_end= 0.02 ,                        #   config.model.beta_max//args.num_scales,
    #         num_diffusion_timesteps= 1000
    #     )
    # betas = torch.from_numpy(betas).float().to(device)
    # for i in range(999):
    #     print(i)
    #     x_1 = generalized_steps(x_1, i, teacher, betas)
    # save(x_1,'hhh')
    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
    image_size = [32,3,32,32] #deepcopy(teacher.image_size)
    image_size[0] = args.batch_size
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.RandomHorizontalFlip()
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)
    for id, (data, label) in enumerate(dataloader):
        data = data.to(device)
        img = make_visualization_encoder(teacher_diffusion, encoder, data, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value, num=id)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)
    print("Finished.")

parser = make_argument_parser()

args = parser.parse_args()


sample_images(args)









