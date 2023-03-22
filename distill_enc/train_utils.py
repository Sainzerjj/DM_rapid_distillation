import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from distill_enc.moving_average import moving_average
from distill_enc.strategies import *
from distill_enc.models.ema import ExponentialMovingAverage
from distill_enc.models.utils import create_model
from torchvision.utils import make_grid, save_image
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from scipy.stats import norm
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save(img,name):
    inverse_scaler = get_data_inverse_scaler(config=None)
    img = inverse_scaler(img)
    image_grid = make_grid(img, nrow=8, padding=2,normalize=True)
    save_image(image_grid, f"{name}.png")
import datetime
def distributions(x, step):
    # config=cifar10_ddpm_continuous.get_config()
    # x=prior_sampling([128,3,32,32])
    # print(x)
    # w = np.random.normal(-2, 2, 10000)
    x=x.cpu().detach()
    norm_data=data_norm(x)
    sns.distplot(norm_data, bins=100, fit=norm)
    plt.title("honest parameters")
    cur_time = datetime.datetime.now()
    time = cur_time.strftime("%F-%H:%M")
    plt.savefig('db_G/db_{}.png'.format(str(step) + '_' + time), format='png', dpi=300)
    plt.show()
    plt.clf()

def data_norm(x):
    x_min = x.min()
    if x_min<0 :
        x+=torch.abs(x_min)
        x_min=x.min()
    x_max=x.max()
    dst=x_max-x_min
    norm=(x-x_min).true_divide(dst)
    return norm

@torch.no_grad()
def p_sample_loop(diffusion, noise, extra_args, device, eta=0, samples_to_capture=-1, need_tqdm=True, clip_value=3, num=1):
    mode = diffusion.net_.training
    diffusion.net_.eval()
    img = noise
    imgs = []
    iter_ = reversed(range(0, diffusion.num_timesteps))
    c_step = diffusion.num_timesteps/samples_to_capture
    next_capture = c_step
    if need_tqdm:
        iter_ = tqdm(iter_)
    
    for i in iter_:
        img = diffusion.p_sample(
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            extra_args,
            eta=eta,
            clip_value=clip_value
        )
        # distributions(img, i)
        if diffusion.num_timesteps - i > next_capture:
            imgs.append(img)
            next_capture += c_step
    # to_range_0_1 = lambda x: (x + 1.) / 2.
    # batch_size = noise.shape[0]
    # for i in range(batch_size):
    #     save_image(to_range_0_1(img[i]), './x/result_{}.jpg'.format(i + num * batch_size))
    save(img, "result_x")

    imgs.append(img)
    diffusion.net_.train(mode)   
    return imgs

import torchvision
from torchvision import transforms
@torch.no_grad()
def p_sample_loop_encoder(diffusion, encoder, noise, data, extra_args, device, eta=0, samples_to_capture=-1, need_tqdm=True, clip_value=3, num=1):
    mode = diffusion.net_.training
    diffusion.net_.eval()
    img = noise
    imgs = []
    iter_ = reversed(range(0, diffusion.num_timesteps))
    c_step = diffusion.num_timesteps/samples_to_capture
    next_capture = c_step
    batch_size = noise.shape[0]
    if need_tqdm:
        iter_ = tqdm(iter_)
    for i in iter_:
        img = diffusion.p_sample(
            img,
            data,
            # time_line,
            # i,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            extra_args,
            eta=eta,
            clip_value=clip_value
        )
        # distributions(img, i)
        if diffusion.num_timesteps - i > next_capture:
            imgs.append(img)
            next_capture += c_step

    to_range_0_1 = lambda x: (x + 1.) / 2.
    # save_image(to_range_0_1(img), 'yyy.jpg')
    for i in range(batch_size):
        save_image(to_range_0_1(img[i]), './x/result_{}.jpg'.format(i + num * batch_size))
    save(img, "1")

    # imgs.append(img)
    diffusion.net_.train(mode)   
    return imgs


def make_none_args(img, label, device):
    return {}


def default_iter_callback(N, loss, last=False):
    None


def make_visualization_(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2, num=1):
    extra_args = {}
    noise = torch.randn(image_size, device=device)
    # fourior fast solver
    # alpha=5
    # freq_mask_path = "mask/cifar10_freq.npy"
    # space_mask_path = None
    # freq_mask, space_mask = get_mask(freq_mask_path, space_mask_path, image_size[-1], alpha)
    # noise = idft2d(dft2d(noise * space_mask) * freq_mask).real
    # noise = noise.type(torch.cuda.FloatTensor)
    imgs = p_sample_loop(diffusion, noise, extra_args, "cuda", samples_to_capture=5, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value, num=num)
    images_ = []
    for images in imgs:
        images = images.split(1, dim=0)
        images = torch.cat(images, -1)
        images_.append(images)
    images_ = torch.cat(images_, 2)
    return images_

def make_visualization_encoder_(diffusion, encoder, data, device, image_size, need_tqdm=False, eta=0, clip_value=1.2, num=1):
    extra_args = {}
    noise = torch.randn(image_size, device=device)
    # fourior fast solver
    # alpha=5
    # freq_mask_path = "mask/cifar10_freq.npy"
    # space_mask_path = None
    # freq_mask, space_mask = get_mask(freq_mask_path, space_mask_path, image_size[-1], alpha)
    # noise = idft2d(dft2d(noise * space_mask) * freq_mask).real
    # noise = noise.type(torch.cuda.FloatTensor)
    imgs = p_sample_loop_encoder(diffusion, encoder, noise, data, extra_args, "cuda", samples_to_capture=5, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value, num=num)
    images_ = []
    # for images in imgs:
    #     images = images.split(1, dim=0)
    #     images = torch.cat(images, -1)
    #     images_.append(images)
    # images_ = torch.cat(images_, 2)
    return images_


def make_visualization(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2, num=1):
    images_ = make_visualization_(diffusion, device, image_size, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value, num=num)
    images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    images_ = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    return images_

def make_visualization_encoder(diffusion, encoder, data, device, image_size, need_tqdm=False, eta=0, clip_value=1.2, num=1):
    images_ = make_visualization_encoder_(diffusion, encoder, data, device, image_size, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value, num=num)
    # images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    # images_ = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    return images_

def make_iter_callback(diffusion, device, checkpoint_path, image_size, tensorboard, log_interval, ckpt_interval, need_tqdm=False):
    state = {
        "initialized": False,
        "last_log": None,
        "last_ckpt": None
    }
    
    def iter_callback_D(netD, optimizerG, optimizerD, schedulerG, schedulerD, N, loss, ema, last=False):
        from datetime import datetime
        t = datetime.now()
        if True:
            tensorboard.add_scalar("loss", loss, N)
        if not state["initialized"]:
            state["initialized"] = True
            state["last_log"] = t
            state["last_ckpt"] = t
            return
        if ((t - state["last_ckpt"]).total_seconds() / 60 > ckpt_interval) or last:
            torch.save({"G": diffusion.net_.state_dict(), "n_timesteps": diffusion.num_timesteps, "time_scale": diffusion.time_scale, "N": N, "ema": ema,
                       "optimizerG": optimizerG.state_dict()}, 
                        os.path.join(checkpoint_path, "checkpoint_{}.pt".format(diffusion.num_timesteps)))
            print("Saved_G.")
            torch.save({"D": netD.state_dict(), "optimizerD": optimizerD.state_dict(), "schedulerD": schedulerD.state_dict()}, os.path.join(checkpoint_path, "netD_{}.pt".format(diffusion.num_timesteps)))
            print("Saved_D.")
            state["last_ckpt"] = t
        if ((t - state["last_log"]).total_seconds() / 60 > log_interval) or last:
            ema.store(diffusion.net_.parameters())
            ema.copy_to(diffusion.net_.parameters())
            images_ = make_visualization(diffusion, device, image_size, need_tqdm)
            images_ = cv2.cvtColor(images_, cv2.COLOR_BGR2RGB)
            tensorboard.add_image("visualization", images_, global_step=N, dataformats='HWC')
            tensorboard.flush()
            ema.restore(diffusion.net_.parameters())
            state["last_log"] = t
    
    def iter_callback(N, loss, ema, encoder, optimizerG, scheduler, last=False):
        from datetime import datetime
        t = datetime.now()
        if True:
            tensorboard.add_scalar("loss", loss, N)
        if not state["initialized"]:
            state["initialized"] = True
            state["last_log"] = t
            state["last_ckpt"] = t
            return
        if ((t - state["last_ckpt"]).total_seconds() / 60 > ckpt_interval) or last:
            torch.save({"G": diffusion.net_.state_dict(), "n_timesteps": diffusion.num_timesteps, "time_scale": diffusion.time_scale, "N": N, "ema": ema, 
                            "encoder": encoder.state_dict(), "optimizerG": optimizerG.state_dict(), "scheduler": scheduler.student_optimizer.state_dict()}, 
                        os.path.join(checkpoint_path, "checkpoint_z_{}.pt".format(diffusion.num_timesteps)))
            print("Saved_G.")
            state["last_ckpt"] = t
        if ((t - state["last_log"]).total_seconds() / 60 > log_interval) or last:
            ema.store(diffusion.net_.parameters())
            ema.copy_to(diffusion.net_.parameters())
            images_ = make_visualization(diffusion, device, image_size, need_tqdm)
            images_ = cv2.cvtColor(images_, cv2.COLOR_BGR2RGB)
            tensorboard.add_image("visualization", images_, global_step=N, dataformats='HWC')
            tensorboard.flush()
            ema.restore(diffusion.net_.parameters())
            state["last_log"] = t
            
    return iter_callback
    


class InfinityDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, L):
        self.dataset = dataset
        self.L = L

    def __getitem__(self, item):
        idx = random.randint(0, len(self.dataset) - 1)
        r = self.dataset[idx]
        return r

    def __len__(self):
        return self.L


def make_condition(img, label, device):
    return {}


def get_Diffusion_Model(config, model_dir):
    """get_Diffusion_Model initialize diffusion model

    Args:
        args (struct): input arguments
        config (_type_): the original configs of the score-based model
        model_dir (_type_): the path of the trained checkpoint of the score-based model

    Returns:
        score_model: the trained score model
        ema: exponential moving average of a set of parameters
    """

    score_model = create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    loaded_state = torch.load(model_dir)
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    score_model = state['model']
    ema = state['ema']
    return score_model, ema 

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config is None:
      is_center = True
  else:
      is_center = config.data.centered
  if is_center:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

class DiffusionTrain:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train(self, train_loader, diffusion, model_ema, optimizerG, model_lr, device, N, make_extra_args=make_none_args, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(train_loader)
        scheduler.init(diffusion, model_lr, total_steps)
        checkpoint_to_continue = "./checkpoints/cifar_exp/base_v/checkpoint_z_1000.pt"
        if checkpoint_to_continue != "":
            ckpt = torch.load(checkpoint_to_continue)
            scheduler.student_optimizer.load_state_dict(ckpt["scheduler"])
            del ckpt
        diffusion.net_.train()
        diffusion.encoder.to(device)
        diffusion.encoder.train()
        print(f"Training...")
        pbar = tqdm(train_loader)
        # pbar = enumerate(train_loader)
        L_tot = 0
        # N = 0
        ema = ExponentialMovingAverage(diffusion.net_.parameters(), decay=0.9999)
        for (img, label) in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = torch.randint(0, diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = diffusion.p_loss_z(img, time, extra_args)
            L_tot += loss.item()
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.net_.parameters(), 1)
            scheduler.step()
            optimizerG.step()
            moving_average(diffusion.net_, model_ema)
            ema.update(diffusion.net_.parameters())
            on_iter(N, loss.item(), ema, diffusion.encoder, optimizerG, scheduler)
            # if N % 6250 == 0:
            #     on_iter(N, loss.item(), ema, diffusion.encoder, optimizerG, scheduler, last=True)
            if scheduler.stop(N, total_steps):
                break
        on_iter(N, loss.item(), ema, diffusion.encoder, optimizerG, scheduler, last=True)


class DiffusionDistillation:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train_student_debug(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, student_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        total_steps = len(distill_train_loader)
        scheduler = self.scheduler
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0

        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = 2 * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            moving_average(student_diffusion.net_, student_ema)
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item())
        on_iter(N, loss.item(), last=True)

    
    def train_student(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, optimizerG, student_lr, speed_up, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(distill_train_loader)
        scheduler.init(student_diffusion, student_lr, total_steps)
        checkpoint_to_continue = ""
        if checkpoint_to_continue != "":
            ckpt = torch.load(checkpoint_to_continue)
            scheduler.student_optimizer.load_state_dict(ckpt["scheduler"])
            del ckpt
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        student_diffusion.encoder.to(device)
        student_diffusion.encoder.train()
        pbar = tqdm(distill_train_loader)
        time_line = torch.round(torch.linspace(0, np.sqrt(999), 11) ** 2).to(device)
        N = 0
        L_tot = 0
        ema = ExponentialMovingAverage(student_diffusion.net_.parameters(), decay=0.9999)
        for img, label in pbar:
            scheduler.zero_grad()
            optimizerG.zero()
            img = img.to(device)
            index = torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            # time = speed_up * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            y = None
            extra_args = make_extra_args(img, label, device)
            # latent_z = student_diffusion.encoder(img)
            # latent_z = torch.randn(img.shape[0], 512, device=torch.device("cuda"))
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time_line, index, speed_up, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            optimizerG.step()
            moving_average(student_diffusion.net_, student_ema)
            ema.update(student_diffusion.net_.parameters())
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item(), ema, student_diffusion.encoder, optimizerG, scheduler)
        on_iter(N, loss.item(), ema, student_diffusion.encoder, optimizerG, scheduler, last=True)


    
    def train_G(self, distill_train_loader, teacher_diffusion, netD, student_diffusion, student_ema, optimizerG, optimizerD, schedulerG, schedulerD, device, speed_up, make_extra_args=make_none_args, on_iter=default_iter_callback):
        total_steps = len(distill_train_loader)
        print(total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        # pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0
        ema = ExponentialMovingAverage(student_diffusion.net_.parameters(), decay=0.9999)
        writer = SummaryWriter()
        for epoch in range(1000):
            print(epoch)
            pbar = tqdm(distill_train_loader)
            for img, label in pbar:
                img = img.to(device)
                time = speed_up * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
                extra_args = make_extra_args(img, label, device)
                errG, errD = teacher_diffusion.train(student_diffusion, netD, img, time, optimizerG, optimizerD, N, extra_args, speed_up=speed_up, device=device)
                L_tot += errG   
                N += 1
                # set_postfix_str('loss={:^7.3f},Cos={:^7.3f}'.format(loss.detach().cpu().numpy(),torch.mean(cos_delta_x)))
                # pbar.set_postfix_str(f"Loss: {L_tot / N} G Loss: {errG}, D Loss: {errD}")
                # pbar.set_description(f"Loss: {L_tot / N}")
                profix = 'Loss:{:^.7f} errG:{:^.7f} errD:{:^.7f}'.format(L_tot / N, errG, errD)
                pbar.set_postfix_str( profix )
                # pbar.set_description(f"G Loss: {errG}")
                # pbar.set_description(f"D Loss: {errD}")
                moving_average(student_diffusion.net_, student_ema)
                ema.update(student_diffusion.net_.parameters())
                on_iter(netD, optimizerG, optimizerD, schedulerG, schedulerD, N, errG, ema)
                if not N % 150:
                    writer.add_scalars('Loss', {'Loss/G':errG, 'Loss/D':errD}, global_step=N)
            # writer.add_scalar('Loss/G', errG, epoch)
            # writer.add_scalar('Loss/D', errG, epoch)
            schedulerG.step()
            schedulerD.step()
        on_iter(netD, optimizerG, optimizerD, schedulerG, schedulerD, N, errG, ema, last=True)




   