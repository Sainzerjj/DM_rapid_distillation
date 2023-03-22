import math
import numpy as np
import torch
import torch.nn.functional as F
from distill_enc.train_utils import save, distributions
from model.unet import BeatGANsEncoderConfig
def make_diffusion(model, n_timestep, time_scale, device):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
    return GaussianDiffusion(model, betas, time_scale=time_scale)

def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    elif schedule == "linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    else:
        raise Exception()
    return betas


def E_(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])
        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)
    else:
        return noise_fn(*shape, device=device)


class GaussianDiffusion:

    def __init__(self, net, betas, time_scale=1, sampler="ddpm"):
        super().__init__()
        self.encoder = None # BeatGANsEncoderConfig(
        #     image_size = 32,
        #     in_channels = 3,
        #     model_channels= 64,
        #     out_hid_channels = 512,
        #     out_channels = 512,
        #     num_res_blocks = 2,
        #     attention_resolutions = (16, ), # None, # (conf.enc_attn_resolutions
        #                            # or conf.attention_resolutions),
        #     dropout = 0.1,
        #     channel_mult= (1, 2, 2, 4, 4), # conf.enc_channel_mult or conf.channel_mult,
        #     use_time_condition=False,
        #     conv_resample = True,
        #     dims = 2,
        #     use_checkpoint = False,
        #     num_heads = 1,
        #     num_head_channels = -1,
        #     resblock_updown = True,
        #     use_new_attention_order = False,
        #     pool = 'adaptivenonzero',
        # ).make_model()

        self.net_ = net
        self.time_scale = time_scale
        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64, device=betas.device), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

        if sampler == "ddpm":
            self.p_sample = self.p_sample_ddpm
        else:
            self.p_sample = self.p_sample_clipped

    def inference(self, x, t, extra_args):
        return self.net_(x, t * self.time_scale, **extra_args)
    
    def inference_y(self, x, data, t):
        return self.net_(x, t * self.time_scale, y=None, x_start=data)
    
    def inference_G(self, x, t, extra_args):
        latent_z = torch.randn(x.shape[0], 512, device=torch.device("cuda"))
        return self.net_(x, t * self.time_scale, latent_z)
    
    def inference_z(self, x, t, latent_z):
        return self.net_(x, t * self.time_scale, latent_z)

    def p_loss(self, x_0, t, extra_args, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.get_alpha_sigma(x_0, t)
        z = alpha_t * x_0 + sigma_t * noise
        v_recon = self.inference_G(z.float(), t.float(), extra_args)
        v = alpha_t * noise - sigma_t * x_0
        return F.mse_loss(v_recon, v.float())
        # return F.mse_loss(v_recon, noise.float())
        # w = alpha_t**2 / sigma_t**2
        # return F.mse_loss(w * v_recon, w * x_0.float()) 
    
    def p_loss_z(self, x_0, t, extra_args, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.get_alpha_sigma(x_0, t)
        z = alpha_t * x_0 + sigma_t * noise
        v_recon = self.inference_y(z.float(), x_0.float(), t.float())
        v = alpha_t * noise - sigma_t * x_0
        x_recon = alpha_t * z - sigma_t * v
        save(x_recon,'pred_x0')
        return F.mse_loss(v_recon.pred, v.float())
        # return F.mse_loss(v_recon, noise.float())
        # w = alpha_t**2 / sigma_t**2
        # return F.mse_loss(w * v_recon, w * x_0.float()) 

    def q_posterior(self, x_0, x_t, t):
        mean = E_(self.posterior_mean_coef1, t, x_t.shape) * x_0 \
               + E_(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = E_(self.posterior_variance, t, x_t.shape)
        log_var_clipped = E_(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_mean_variance(self, x, t, latent_z, extra_args, clip_denoised):
        # v = self.inference_z(x.float(), t.float(), latent_z).double()
        v = self.inference_G(x.float(), t.float(), extra_args).double()
        # alpha_t, sigma_t = self.get_alpha_sigma(x, t)
        # x_recon = alpha_t * x - sigma_t * v
        x_recon = v
        # x_recon = self.inference(x.float(), t.float(), extra_args).double()
        
        # x_recon = self.inference_G(x.float(), t.float()).double()
        # if t[0]%200==0:
        # save(x_recon, "x/{}".format(t[0]))
        # x_recon = (x - sigma_t * v) / alpha_t
        if clip_denoised:
            x_recon = x_recon.clamp(min=-1, max=1)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    def p_sample_ddpm(self, x, t, latent_z, extra_args={}, clip_denoised=True, **kwargs):
        mean, _, log_var = self.p_mean_variance(x, t, latent_z, extra_args, clip_denoised)
        noise = torch.randn_like(x)

        # noise = idft2d(dft2d(torch.rand_like(x) * self.space_mask) * self.freq_mask).real
        # noise = noise.type(torch.cuda.FloatTensor)
        # fourior fast solve1   ``
        # alpha=5
        # freq_mask_path = "mask/cifar10_freq.npy"
        # space_mask_path = None
        # image_size = x.shape[-1]
        # freq_mask, space_mask = get_mask(freq_mask_path, space_mask_path, image_size, alpha)
        # noise = idft2d(dft2d(noise * space_mask) * freq_mask).real
        # noise = noise.type(torch.cuda.FloatTensor)

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    def p_sample_clipped(self, x, data, t, extra_args, eta=0, clip_denoised=True, clip_value=2):
        v = self.inference_y(x.float(), data, t)
        alpha, sigma = self.get_alpha_sigma(x, t)
        # index_tp1 = torch.full((data.shape[0],), time_line[index-1], dtype=torch.int64).cuda()
        # if clip_denoised:
        #     x = x.clip(-1, 1)
        # pred = v.pred
        pred = (x * alpha - v.pred * sigma)
        save(pred,'pred')
        # if clip_denoised:
        #     pred = pred.clip(-clip_value, clip_value)
        eps = (x - alpha * pred) / sigma
        # if clip_denoised:
        #     eps = eps.clip(-clip_value, clip_value)
        
        t_mask = (t > 0)
        if t_mask.any().item():
            if not t_mask.all().item():
                raise Exception()
            alpha_, sigma_ = self.get_alpha_sigma(x, (t-1).clip(min=0))
            ddim_sigma = eta * (sigma_ ** 2 / sigma ** 2).sqrt() * \
                         (1 - alpha ** 2 / alpha_ ** 2).sqrt()
            adjusted_sigma = (sigma_ ** 2 - ddim_sigma ** 2).sqrt()
            pred = pred * alpha_ + eps * adjusted_sigma
            if eta:
                pred += torch.randn_like(pred) * ddim_sigma
        return pred

    @torch.no_grad()
    def p_sample_loop(self, x, extra_args, eta=0):
        mode = self.net_.training
        self.net_.eval()
        for i in reversed(range(self.num_timesteps)):
            x = self.p_sample(
                x,
                torch.full((x.shape[0],), i, dtype=torch.int64).to(x.device),
                extra_args,
                eta=eta,
            )
        self.net_.train(mode)
        return x

    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma


class GaussianDiffusionDefault(GaussianDiffusion):

    def __init__(self, net, betas, time_scale=1, gamma=0.3):
        super.__init__(net, betas, time_scale)
        self.gamma = gamma
        self.encoder = None # BeatGANsEncoderConfig(
        #     image_size = 32,
        #     in_channels = 3,
        #     model_channels= 64,
        #     out_hid_channels = 512,
        #     out_channels = 512,
        #     num_res_blocks = 2,
        #     attention_resolutions = (16, ), # None, # (conf.enc_attn_resolutions
        #                            # or conf.attention_resolutions),
        #     dropout = 0.1,
        #     channel_mult= (1, 2, 2, 4, 4), # conf.enc_channel_mult or conf.channel_mult,
        #     use_time_condition=False,
        #     conv_resample = True,
        #     dims = 2,
        #     use_checkpoint = False,
        #     num_heads = 1,
        #     num_head_channels = -1,
        #     resblock_updown = True,
        #     use_new_attention_order = False,
        #     pool = 'adaptivenonzero',
        # ).make_model()
        # self.freq_mask = get_mask('./mask/cifar10_freq.npy', None, img_size=32, alpha=10)[0]
        # self.space_mask = get_mask('./mask/cifar10_freq.npy', None, img_size=32, alpha=10)[1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def distill_loss_v(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), t.float() + 1, extra_args).double()
            rec = (alpha * z - sigma * v).clip(-1, 1)
            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            eps_2 = (z - alpha_s * x_2) / sigma_s
            v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
            #    w = 1 + alpha_s**2 / sigma_s**2
        v = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args)
        # my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        return F.mse_loss(w * v.float(), w * v_2.float())
    
    def distill_loss_eps(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            eps = self.inference(z.float(), t.float() + 1, extra_args).double()
            rec = ((z - sigma * eps)/alpha).clip(-4, 4)
            # z_1 = alpha_1 * rec + sigma_1 * eps
            z_1 = self.p_sample_ddpm(z, t + 1, extra_args)
            eps_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = ((z_1 - sigma_1 * eps_1)/alpha_1).clip(-4, 4)
            # eps_2 = (z - alpha_s * x_2) / sigma_s
            # v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
            #    w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
                w = 1 + alpha_s**2 / sigma_s**2
        eps = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args)
        my_rec = ((z - sigma_s * eps)/alpha_s).clip(-1, 1)
        # v = alpha_s * eps - sigma_s * my_rec
        return F.mse_loss(w * my_rec.float(), w * x_2.float())

    def distill_loss_x(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
            # eps = idft2d(dft2d(torch.rand_like(x) * self.space_mask) * self.freq_mask).real
            # eps = eps.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            # v = self.inference(z.float(), t.float() + 1, extra_args).double()
            # rec = (alpha * z - sigma * v).clip(-1, 1)
            # eps_1 = (z - alpha * rec) / sigma
            # z_1 = alpha_1 * rec + sigma_1 * eps_1
            z_1 = self.p_sample_clipped(z, t + 1, extra_args)
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            # eps_2 = (z - alpha_s * x_2) / sigma_s
            # v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
            #    w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
                w = 1 + alpha_s**2 / sigma_s**2
        my_rec = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args)
        # v = alpha_s * eps - sigma_s * my_rec
        return F.mse_loss(w * my_rec.float(), w * x_2.float())
    
    def distill_loss(self, student_diffusion, x, time_line, index, speed_up, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
            # eps = idft2d(dft2d(torch.rand_like(x) * self.space_mask) * self.freq_mask).real
            # eps = eps.type(torch.cuda.FloatTensor)
        t = torch.tensor(time_line[index], dtype=torch.int64)
        tp1 = torch.tensor(time_line[index+1], dtype=torch.int64)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, tp1)
            z = alpha * x + sigma * eps
            # alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, index)
            alpha_s, sigma_s = alpha, sigma
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), tp1.float(), extra_args).double()
            rec = (alpha * z - sigma * v).clip(-1, 1)
            eps_1 = (z - alpha * rec) / sigma
            z_1 = alpha_1 * rec + sigma_1 * eps_1
            # z_1 = self.p_sample_ddpm(z, t + 99, extra_args)
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            # eps_2 = (z - alpha_s * x_2) / sigma_s
            # v_2 = alpha_s * eps_2 - sigma_s * x_2

            if self.gamma == 0:
                w = 1
            else:
            #    w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
                w = 1 + alpha_s**2 / sigma_s**2
        # v = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args)
        # my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        # mean, var, log_var = self.q_posterior(my_rec, z, t)
        # noise = torch.randn_like(z)
        # shape = [z.shape[0]] + [1] * (z.ndim - 1)
        # nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        # my_rec_2 = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        # my_rec = student_diffusion.net_(z.float(), t.float() * self.time_scale, latent, **extra_args)
        out = student_diffusion.net_(z.float(), t.float() * self.time_scale, y =None, x_start = x)
        my_rec = (alpha_s * z - sigma_s * out.pred).clip(-1, 1)
        save(my_rec,'3')
        # v = alpha_s * eps - sigma_s * my_rec
        return F.mse_loss(w * my_rec.float(), w * x_2.float())



    def train(self, student_diffusion, netD, x, t, optimizerG, optimizerD, N, extra_args, eps=None, speed_up=None, device=None):
        if eps is None:
            eps = torch.randn_like(x)
        for p in netD.parameters():  
            p.requires_grad = True

        skip_step = speed_up - 1
        netD.zero_grad()
        alpha, sigma = self.get_alpha_sigma(x, t + skip_step)
        alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
        alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // speed_up)

        z = alpha * x + sigma * eps
        v = self.inference(z.float(), t.float() + skip_step, extra_args).double()

        # eps_1 = self.inference(z.float(), t.float() + skip_step, extra_args).double()
        # rec = ((z - sigma * eps_1) / alpha).clip(-1,1)
        
        rec = (alpha * z - sigma * v).clip(-1, 1)
        # rec = self.inference(z.float(), t.float() + skip_step, extra_args).double()
        eps_1 = (z - alpha * rec) / sigma

        z_1 = alpha_1 * rec + sigma_1 * eps_1
        # z_1 = self.p_sample_clipped(z, t + 199, extra_args)
        # z_1 = alpha_1 * x + sigma_1 * eps
        # z_1.requires_grad = True
        v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
        # eps_2 = self.inference(z_1.float(), t.float(), extra_args).double()

        x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
        # x_2 = self.inference(z_1.float(), t.float(), extra_args).double()
        # x_2 = ((z_1 - sigma_1 * eps_2) / alpha_1).clip(-1,1)
        # eps_2 = (z - alpha_s * x_2) / sigma_s
        # v_2 = alpha_s * eps_2 - sigma_s * x_2
        # print(v_2.requires_grad)
        # v_2.requires_grad = True
        # save(rec,'hh1')
        # save(x_2,'hh2')
        # train with real
        # D_real = netD(v_2, t, v.detach()).view(-1)
        # w = torch.max(torch.ones_like(rec), alpha_s**2 / sigma_s**2)
        w = alpha_s**2 / sigma_s**2 + 1
        D_real = netD(w * x_2, t, w * rec.detach()).view(-1)
        # D_real = netD(z_1, t, z.detach()).view(-1)
        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()
        errD_real.backward(retain_graph=True)
        
        if N % 10 == 0:
            grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_2, create_graph=True
                    )[0]
            grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()
        
            r1_gamma = 0.05
            grad_penalty = r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

        # train with fake      

        # pred_v = student_diffusion.net_(z.float(), t.float(), **extra_args)
        # pred_x = (alpha_s * z - sigma_s * pred_v).clip(-1, 1)
        # mean, _, log_var = self.q_posterior(pred_x, z, t)
        # noise = torch.randn_like(z)
        # shape = [x.shape[0]] + [1] * (x.ndim - 1)
        # nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        # pred_z = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        # latent_z = torch.randn(x.shape[0], 100, device=device)
        # pred_x = student_diffusion.net_(z.float().detach(), t.float(), latent_z)
        # pred_x = student_diffusion.net_(z.float().detach(), t.float(), **extra_args)
        pred_v = student_diffusion.net_(z.float().detach(), t.float() * self.time_scale, **extra_args)
        pred_x = (alpha_s * z - sigma_s * pred_v).clip(-1, 1)
        # t_mask = (t > 0)
        # eps = ((z - alpha_s * pred_x) / sigma_s).clip(-1, 1)
        # if t_mask.any().item():
        #     alpha_, sigma_ = self.get_alpha_sigma(x, (t // speed_up - 1).clip(min=0))
        #     pred_z = pred_x * alpha_ + eps * sigma_
        # output = netD(pred_v, t, v.detach()).view(-1)
        output = netD(w * pred_x, t, w * rec.detach()).view(-1)
        # output = netD(pred_z, t, z.detach()).view(-1)
        
        errD_fake = F.softplus(output)
        errD_fake = errD_fake.mean()
        errD_fake.backward()
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        
    
        #update G
        for p in netD.parameters():
            p.requires_grad = False
        student_diffusion.net_.zero_grad()
        
    
        t = speed_up * torch.randint(0, student_diffusion.num_timesteps, (x.shape[0],), device=device)
        alpha, sigma = self.get_alpha_sigma(x, t + skip_step)
        alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // speed_up)
        # w = torch.max(torch.ones_like(rec), alpha_s**2 / sigma_s**2)
        w = alpha_s**2 / sigma_s**2 + 1
        eps = torch.randn_like(x)
        z = alpha * x + sigma * eps

        # v = self.inference(z.float(), t.float() + skip_step, extra_args).double()
        # rec = (alpha * z - sigma * v).clip(-1, 1)
        # eps_1 = self.inference(z.float(), t.float() + skip_step, extra_args).double()
        # rec = ((z - sigma * eps_1) / alpha).clip(-1,1)
        # rec = self.inference(z.float(), t.float() + skip_step, extra_args).double()

        # pred_v = student_diffusion.net_(z.float(), t.float(), **extra_args)
        # pred_x = (alpha_s * z - sigma_s * pred_v).clip(-1, 1)
        # mean, _, log_var = self.q_posterior(pred_x, z, t)
        # noise = torch.randn_like(z)
        # shape = [x.shape[0]] + [1] * (x.ndim - 1)
        # nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        # pred_z = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        # latent_z = torch.randn(x.shape[0], 100, device=device)
        # pred_x = student_diffusion.net_(z.float().detach(), t.float(), latent_z)

        # pred_x = student_diffusion.net_(z.float().detach(), t.float(), **extra_args)
        pred_v = student_diffusion.net_(z.float().detach(), t.float() * self.time_scale, **extra_args)
        pred_x = (alpha_s * z - sigma_s * pred_v).clip(-1, 1)
        # t_mask = (t > 0)
        # eps = ((z - alpha_s * pred_x) / sigma_s).clip(-1, 1)
        # if t_mask.any().item():
        #     alpha_, sigma_ = self.get_alpha_sigma(x, (t // speed_up - 1).clip(min=0))
        #     pred_z = pred_x * alpha_ + eps * sigma_

        # output = netD(pred_v, t, v.detach()).view(-1)
        output = netD(w * pred_x, t, w * rec.detach()).view(-1)
        # output = netD(pred_z, t, z.detach()).view(-1)
            
        errG = F.softplus(-output)
        errG = errG.mean()
        
        errG.backward()
        optimizerG.step()

        return errG.item(), errD.item()
        
        