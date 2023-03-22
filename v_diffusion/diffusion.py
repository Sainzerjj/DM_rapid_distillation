import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .utils import save_image_cuda
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T
try:
    from .functions import normal_kl, discretized_gaussian_loglik, flat_mean
except ImportError:
    import sys
    from pathlib import Path

    PROJ_DIR = str(Path(__file__).resolve().parents[1])
    if PROJ_DIR not in sys.path:
        sys.path.append(PROJ_DIR)
    from v_diffusion.functions import normal_kl, discretized_gaussian_loglik, flat_mean


def broadcast_to(
        arr, x,
        dtype=None, device=None, ndim=None):
    if x is not None:
        dtype = dtype or x.dtype
        device = device or x.device
        ndim = ndim or x.ndim
    out = torch.as_tensor(arr, dtype=dtype, device=device)
    return out.reshape((-1,) + (1,) * (ndim - 1))


def get_logsnr_schedule(schedule, logsnr_min: float = -20., logsnr_max: float = 20.):
    """
    schedule is named according to the relationship between alpha2 and t,
    i.e. alpha2 as a XX function of affine transformation of t (except for legacy)
    """

    logsnr_min, logsnr_max = torch.as_tensor(logsnr_min), torch.as_tensor(logsnr_max)
    if schedule == "linear":
        def logsnr2t(logsnr):
            return torch.sigmoid(logsnr)

        def t2logsnr(t):
            return torch.logit(t)
    elif schedule == "sigmoid":
        logsnr_range = logsnr_max - logsnr_min

        def logsnr2t(logsnr):
            return (logsnr_max - logsnr) / logsnr_range

        def t2logsnr(t):
            return logsnr_max - t * logsnr_range
    elif schedule == "cosine":
        def logsnr2t(logsnr):
            return torch.atan(torch.exp(-0.5 * logsnr)).div(0.5 * math.pi)
            
        def t2logsnr(t):
            return -2 * torch.log(torch.tan(t * math.pi * 0.5))
    elif schedule == "legacy":
        """
        continuous version of the (discrete) linear schedule used by \
          Ho, Jonathan, Ajay Jain, and Pieter Abbeel. \
            "Denoising diffusion probabilistic models." \
              Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
        """
        delta_max, delta_min = (
            torch.as_tensor(1 - 0.0001),
            torch.as_tensor(1 - 0.02))
        delta_max_m1 = torch.as_tensor(-0.0001)
        log_delta_max = torch.log1p(delta_max_m1)
        log_delta_min = torch.log1p(torch.as_tensor(-0.02))
        delta_range = delta_max - delta_min
        log_alpha_range = (delta_max * log_delta_max -
                           delta_min * log_delta_min) / delta_range - 1

        def schedule_fn(t):
            tau = delta_max - delta_range * t
            tau_m1 = delta_max_m1 - delta_range * t
            log_alpha = (
                    (delta_max * log_delta_max - tau * torch.log1p(tau_m1))
                    / delta_range - t).mul(-20. / log_alpha_range).add(-2.0612e-09)
            return log_alpha - stable_log1mexp(log_alpha)

        return schedule_fn
    else:
        raise NotImplementedError
    b = logsnr2t(logsnr_max)
    a = logsnr2t(logsnr_min) - b

    def schedule_fn(t):
        _a, _b = broadcast_to(a, t), broadcast_to(b, t)
        return t2logsnr(_a * t + _b)

    return schedule_fn


def stable_log1mexp(x):
    """
    numerically stable version of log(1-exp(x)), x<0
    """
    assert torch.all(x < 0.)
    return torch.where(
        x < -9,
        torch.log1p(torch.exp(x).neg()),
        torch.log(torch.expm1(x).neg()))

# SNR = log(alpha^2/sigma^2)   z_t = alpha * x + sigma * noise
def logsnr_to_posterior(logsnr_s, logsnr_t, var_type: str, intp_frac=None):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
    logr = logsnr_t - logsnr_s
    log_one_minus_r = stable_log1mexp(logr)
    mean_coef1 = (logr + log_alpha_st).exp()
    mean_coef2 = (log_one_minus_r + 0.5 * F.logsigmoid(logsnr_s)).exp()

    # strictly speaking, only when var_type == "small",
    # does `logvar` calculated here represent the logarithm
    # of the true posterior variance
    if var_type == "fixed_large":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
    elif var_type == "fixed_small":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
    elif var_type == "fixed_medium":
        # linear interpolation in log-space
        assert isinstance(intp_frac, (float, torch.Tensor))
        logvar = (
                intp_frac * (log_one_minus_r + F.logsigmoid(-logsnr_t)) +
                (1. - intp_frac) * (log_one_minus_r + F.logsigmoid(-logsnr_s))
        )
    else:
        raise NotImplementedError(var_type)

    return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


DEBUG = False


def logsnr_to_posterior_ddim(logsnr_s, logsnr_t, eta: float):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    if not DEBUG and eta == 1.:
        return logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small")
    else:
        if DEBUG:
            print("Debugging mode...")
        log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
        logr = logsnr_t - logsnr_s
        if eta == 0:
            log_one_minus_sqrt_r = stable_log1mexp(0.5 * logr)
            mean_coef1 = (F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)).mul(0.5).exp()
            mean_coef2 = (log_one_minus_sqrt_r + 0.5 * F.logsigmoid(logsnr_s)).exp()
            logvar = torch.as_tensor(-math.inf)
        else:
            log_one_minus_r = stable_log1mexp(logr)
            logvar = log_one_minus_r + F.logsigmoid(-logsnr_s) + 2 * math.log(eta)
            mean_coef1 = stable_log1mexp(
                logvar - F.logsigmoid(-logsnr_s))
            mean_coef1 += F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)
            mean_coef1 *= 0.5
            mean_coef2 = stable_log1mexp(mean_coef1 - log_alpha_st).add(
                0.5 * F.logsigmoid(logsnr_s))
            mean_coef1, mean_coef2 = mean_coef1.exp(), mean_coef2.exp()

        return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


@torch.jit.script
def pred_x0_from_eps(x_t, eps, logsnr_t):
    return x_t.div(torch.sigmoid(logsnr_t).sqrt()) - eps.mul(logsnr_t.neg().mul(.5).exp())


def pred_x0_from_x0eps(x_t, x0eps, logsnr_t):
    x_0, eps = x0eps.chunk(2, dim=1)
    _x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
    return x_0.mul(torch.sigmoid(-logsnr_t)) + _x_0.mul(torch.sigmoid(logsnr_t))


@torch.jit.script
def pred_eps_from_x0(x_t, x_0, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) - x_0.mul(logsnr_t.mul(.5).exp())


@torch.jit.script
def pred_v_from_x0eps(x_0, eps, logsnr_t):
    return -x_0.mul(torch.sigmoid(-logsnr_t).sqrt()) + eps.mul(torch.sigmoid(logsnr_t).sqrt())


@torch.jit.script
def pred_x0_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(logsnr_t).sqrt()) - v.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def pred_eps_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) + v.mul(torch.sigmoid(logsnr_t).sqrt())


def q_sample(x_0, logsnr_t, eps=None):
    if eps is None:
        eps = torch.randn_like(x_0)
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()) + eps.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def q_mean_var(x_0, logsnr_t):
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()), F.logsigmoid(-logsnr_t)


def raise_error_with_msg(msg):
    def raise_error(*args, **kwargs):
        raise NotImplementedError(msg)

    return raise_error


class GaussianDiffusion:
    def __init__(
            self,
            logsnr_fn,
            sample_timesteps,
            model_out_type,
            model_var_type,
            reweight_type,
            loss_type,
            intp_frac=None,
            w_guide=0.1,
            p_uncond=0.1,
            use_ddim=False
    ):
        self.logsnr_fn = logsnr_fn
        self.sample_timesteps = sample_timesteps

        self.model_out_type = model_out_type
        self.model_var_type = model_var_type
        # self.pre_out = 0
        # self.pre_x_0 = 0
        # from mse_target to re-weighting strategy
        # x0 -> constant
        # eps -> SNR
        # both -> truncated_SNR, i.e. max(1, SNR)
        self.reweight_type = reweight_type
        self.loss_type = loss_type
        self.intp_frac = intp_frac
        self.w_guide = w_guide
        self.p_uncond = p_uncond

        self.sel_attn_depth = 8
        self.sel_attn_block = "output"
        self.num_heads = 1
        self.blur_sigma = 3
    def t2logsnr(self, *ts, x=None):
        _broadcast_to = lambda t: broadcast_to(
            self.logsnr_fn(t), x=x)
        return tuple(map(_broadcast_to, ts))

    def q_posterior_mean_var(
            self, x_0, x_t, logsnr_s, logsnr_t, model_var_type=None, intp_frac=None):
        model_var_type = model_var_type or self.model_var_type
        intp_frac = self.intp_frac or intp_frac
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, var_type=model_var_type, intp_frac=intp_frac)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def q_posterior_mean_var_ddim(self, x_0, x_t, logsnr_s, logsnr_t):
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0.)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def p_mean_var(
            self, denoise_fn, x_t, s, t, y, clip_denoised, return_pred, use_ddim=False):

        out = denoise_fn(x_t, t, y=y)
        # save_image_cuda(out, "out.png", nrow=8, normalize=True, value_range=(-1., 1.))
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        # save_image_cuda(pred_x_0, "1.png", nrow=8, normalize=True, value_range=(-1., 1.))
        if use_ddim:
            model_mean, model_logvar = self.q_posterior_mean_var_ddim(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t)
        else:
            model_mean, model_logvar = self.q_posterior_mean_var(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t, intp_frac=intp_frac)

        if return_pred:
            return model_mean, model_logvar, pred_x_0
        else:
            return model_mean, model_logvar
    
    def p_mean_var_enc(
            self, denoise_fn, x_t, s, t, y, x_0, clip_denoised, return_pred, use_ddim=False):

        all_out = denoise_fn(x_t, t, y=y, x_start=x_0)
        out = all_out.pred
        attn_map = all_out.attention
        # save_image_cuda(out, "out.png", nrow=8, normalize=True, value_range=(-1., 1.))
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        pred_eps = pred_eps_from_v(x_t, out, logsnr_t)
        guide_scale = 1.3
        blur_sigma = 3
        mask_blurred = self.attention_masking(
                pred_x_0,
                t,
                attn_map,
                # prev_noise=pred_eps,
                x_real = x_0, 
                blur_sigma=blur_sigma,
            )
        # save_image_cuda(mask_blurred, "result_pred_x.png", nrow=8, normalize=True, value_range=(-1., 1.))
        mask_blurred = q_sample(mask_blurred, logsnr_t, eps=pred_eps)
        _, _, uncond_eps = self.p_pred_x_0_enc(denoise_fn, mask_blurred, t, y, pred_x_0, clip_denoised=True)
        guided_eps = uncond_eps + guide_scale * (pred_eps - uncond_eps)
        pred_x_0 = pred_x0_from_eps(x_t, guided_eps, logsnr_t)
        # save_image_cuda(pred_x_0, "2.png", nrow=8, normalize=True, value_range=(-1., 1.))
        if use_ddim:
            model_mean, model_logvar = self.q_posterior_mean_var_ddim(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t)
        else:
            model_mean, model_logvar = self.q_posterior_mean_var(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t, intp_frac=intp_frac)

        if return_pred:
            return model_mean, model_logvar, pred_x_0   # , guided_eps
        else:
            return model_mean, model_logvar
    
    def p_pred_x_0(
            self, denoise_fn, x_t, t, y, clip_denoised):
        
        out = denoise_fn(x_t, t, y=y)
        logsnr_t, = self.t2logsnr(t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        
        return pred_x_0
    

    def attention_masking(
        self, x, t, attn_map, x_real, blur_sigma, model_kwargs=None,
    ):
        """
        Apply the self-attention mask to produce bar{x_t}

        :param x: the predicted x_0 [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param attn_map: the attention map tensor at time t.
        :param prev_noise: the previously predicted epsilon to inject
            the same noise as x_t.
        :param blur_sigma: a sigma of Gaussian blur.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: the bar{x_t}
        """
        B, C, H, W = x.shape
        assert t.shape == (B,)
        
        if self.sel_attn_depth in [0, 1, 2] or self.sel_attn_block == "middle":
            attn_res = 8
        elif self.sel_attn_depth in [3, 4, 5]:
            attn_res = 16
        elif self.sel_attn_depth in [6, 7, 8]:
            attn_res = 32
        else:
            raise ValueError("sel_attn_depth must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]")
        
        # attn_res = 8
        # attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False)
        # attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).float()
        # attn_mask = F.interpolate(attn_mask, (H, W))
        # save_image_cuda(attn_mask, '1.png')

        # Generating attention mask
        attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).int().float()
        attn_mask = F.interpolate(attn_mask, (H, W))
        # save_image_cuda(attn_mask, '10.png')

        # Gaussian blur
        transform = T.GaussianBlur(kernel_size=15, sigma=blur_sigma)
        x_curr = transform(x)

        # Apply attention masking
        x_curr = x_curr * (attn_mask) + x * (1 - attn_mask)
        # x_curr = x * (attn_mask) + x_real * (1 - attn_mask)
        
        return x_curr

    def p_pred_x_0_enc(
            self, denoise_fn, x_t, t, y, x_start, clip_denoised):
        
        all_out = denoise_fn(x_t, t, y=y, x_start=x_start)
        out = all_out.pred
        attn_map = all_out.attention
        logsnr_t, = self.t2logsnr(t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))

        pred_eps = pred_eps_from_v(x_t, out, logsnr_t)
        return pred_x_0, attn_map, pred_eps
    # === sample ===
    def p_sample_step(
            self, denoise_fn, x_t, step, y, x_start=None,
            clip_denoised=True, return_pred=False, use_ddim=False):
        s, t = step.div(self.sample_timesteps), \
               step.add(1).div(self.sample_timesteps)
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        # delta_out = 0
        # delta_x_0 = 0
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t, s, t, y, x_start, 
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        # if t[0]!=1:
        #     delta_out = out - self.pre_out
        #     delta_x_0 = pred_x_0 - self.pre_x_0
        # self.pre_out = out
        # self.pre_x_0 = pred_x_0
        to_range_0_1 = lambda x: (x + 1.) / 2.
        save_image_cuda(pred_x_0, "1.png")
        # model_mean = torch.where(cond, model_mean, pred_x_0)
        if self.w_guide and y is not None:
            # classifier-free guidance
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                denoise_fn, x_t, s, t, torch.zeros_like(y),
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)

        noise = torch.randn_like(x_t)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample   #, pred_eps

    def p_sample_distill_step(
            self, denoise_fn, x_t, step, y, x_0,
            clip_denoised=True, return_pred=True, use_ddim=False):
        s, t = step.sub(1).div(self.sample_timesteps), \
               step.div(self.sample_timesteps)
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t, s, t, y, x_0, 
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        to_range_0_1 = lambda x: (x + 1.) / 2.
        # save_image(to_range_0_1(pred_x_0), "1.png", )
        # model_mean = torch.where(cond, model_mean, pred_x_0)
        if self.w_guide and y is not None:
            # classifier-free guidance
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                denoise_fn, x_t, s, t, torch.zeros_like(y),
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)

        noise = torch.randn_like(x_t)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(
            self, denoise_fn, shape, x_start=None, 
            noise=None, label=None, device="cpu", use_ddim=False):
        B = shape[0]
        t = torch.empty((B,), device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        if label is not None:
            label = label.to(device)
        # P = torch.ones((10))
        # # w = np.zeros([1001, 1000])
        # x_t = torch.load('x.npy').to(device)
        # a = []
        # b = []
        # c = []
        # d = []
        # e = []
        # f = []
        # g = []
        # h = []
        # w = []
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, x_start=x_start, use_ddim=use_ddim)
            
            # if ti==24 or ti%6==0:
            #     save_image_cuda(pred_eps, 'eps_{}.png'.format(ti))
            # x_tc = data_norm(pred_eps)
            # print(x_tc.max())
            # print(x_tc.min())
            
            # y1 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[30][3]).cpu().numpy()
            # y2 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[15][3]).cpu().numpy()
            # y3 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[3][3]).cpu().numpy()
            # y4 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[30][15]).cpu().numpy()
            # y5 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[15][15]).cpu().numpy()
            # y6 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[3][15]).cpu().numpy()
            # y7 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[30][30]).cpu().numpy()
            # y8 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[15][30]).cpu().numpy()
            # y9 = torch.mean(x_tc.mean(dim=0).mean(dim=0)[3][30]).cpu().numpy()
            # a.append(y1)
            # b.append(y2)
            # c.append(y3)
            # d.append(y4)
            # e.append(y5)
            # f.append(y6)
            # g.append(y7)
            # h.append(y8)
            # w.append(y9)
            
            # if ti!=999 and ti % 20 == 0:
            #     a.append(torch.norm(delta_out, p=2).cpu().numpy())
            #     b.append(torch.norm(delta_x_0, p=2).cpu().numpy())
            # if ti % 100 == 0 or ti == 999:
            #     delta = x_t - noise
            #     pred_y1 = test_sum(x_t)
            #     P = torch.cat([P, pred_y1])
            #     print(P)
            # to_range_0_1 = lambda x: (x + 5.) / 10.
            # y = to_range_0_1(x_t).reshape(B * 3 * 32 * 32, -1)
            # print(ti)
            # for j in range(0, y.shape[0]): 
            #     m = int(y[j] * 1000)
            #     w[m][ti]+=1
            # sns.heatmap(w, cmap="viridis_r")  #        YlGnBu_r
            # plt.xticks(np.arange(0, 1001, step=200))
            # plt.yticks(np.arange(0, 1001, 200))
            # plt.savefig('db_5.png', format='png', dpi=800)
            # plt.show()
            # plt.clf()

        # X = np.linspace(0,1000,50)
        # plt.subplot(2,1,1)
        # b.reverse()
        # plt.plot(X,b,c="blue",label="image",marker="*")
        # plt.subplot(2,1,2)
        # a.reverse()
        # plt.plot(X,a,c="red",label="noise",marker="*")
        # plt.xlabel("step", loc='center',fontsize=12)   # loc: 左中右 left center right
        
        # # Y轴标签
        # plt.ylabel("norm",loc='center',fontsize=12) 
        # plt.show()
        # plt.savefig('norm_x0.png', format='png', dpi=800)
        # plt.clf()
        # a.reverse()
        # b.reverse()
        # c.reverse()
        # d.reverse()
        # e.reverse()
        # f.reverse()
        # g.reverse()
        # h.reverse()
        # w.reverse()
        # X = np.linspace(0,1000,15)
        # plt.plot(X,a,c="r",linestyle="-",linewidth=3)
        # plt.plot(X,b,c="y",linestyle="-",linewidth=3)
        # plt.plot(X,c,c="b",linestyle="-",linewidth=3)
        # plt.plot(X,d,c="g",linestyle="-",linewidth=3)
        # plt.plot(X,e,c="c",linestyle="-",linewidth=3)
        # plt.plot(X,f,c="m",linestyle="-",linewidth=3)
        # plt.plot(X,g,c="orange",linestyle="-",linewidth=3)
        # plt.plot(X,h,c="gold",linestyle="-",linewidth=3)
        # plt.plot(X,w,c="peru",linestyle="-",linewidth=3)

        # plt.xlabel("step", loc='center',fontsize=12)   # loc: 左中右 left center right
        
        # # # Y轴标签
        # plt.ylabel("norm",loc='center',fontsize=12) 
        # plt.show()
        # plt.savefig('norm_noise.png', format='png', dpi=800)
        # plt.clf()
        return x_t.cpu()

    @torch.inference_mode()
    def p_sample_progressive(
            self, denoise_fn, shape,
            noise=None, label=None, device="cpu", use_ddim=False, pred_freq=50):
        B = shape[0]
        t = torch.empty(B, device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        L = self.sample_timesteps // pred_freq
        preds = torch.zeros((L, B) + shape[1:], dtype=torch.float32)
        idx = L
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, return_pred=True, use_ddim=use_ddim)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(
            self, denoise_fn, x_0, x_t, s, t, y, clip_denoised, return_pred):
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_0)
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, true_logvar = self.q_posterior_mean_var(
            x_0=x_0, x_t=x_t,
            logsnr_s=logsnr_s, logsnr_t=logsnr_t, model_var_type="fixed_small")
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t=x_t, s=s, t=t, y=y, x_0=x_0,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=False)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(
            x_0, pred_x_0, log_scale=0.5 * model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(s.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def from_model_out_to_pred(self, x_t, model_out, logsnr_t):
        assert self.model_out_type in {"x0", "eps", "both", "v"}
        if self.model_out_type == "v":
            v = model_out
            x_0 = pred_x0_from_v(x_t, v, logsnr_t)
            eps = pred_eps_from_v(x_t, v, logsnr_t)
        else:
            if self.model_out_type == "x0":
                x_0 = model_out
                eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
            elif self.model_out_type == "eps":
                eps = model_out
                x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
            elif self.model_out_type == "both":
                x_0, eps = model_out.chunk(2, dim=1)
            else:
                raise NotImplementedError(self.model_out_type)
            v = pred_v_from_x0eps(x_0, eps, logsnr_t)
        return {"constant": x_0, "snr": eps, "truncated_snr": (x_0, eps), "alpha2": v}

    def distill_losses(self, student_diffusion, teacher_denoise_fn, student_denoise_fn, 
                x_0, t, y, speed_up, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        with torch.no_grad():
            logsnr_t, = self.t2logsnr(t, x=x_0)
            x_t = q_sample(x_0, logsnr_t, eps=noise)
            t = t.mul(self.sample_timesteps)
            # save_image_cuda(x_t, "result_xt.png", nrow=8, normalize=True, value_range=(-1., 1.))
            x_tm1 = self.p_sample_distill_step(teacher_denoise_fn, x_t, step=t, y=None, 
                                speed_up=speed_up, clip_denoised=True, return_pred=False, use_ddim=True)
            # save_image_cuda(x_tm1, "result_xtm1.png", nrow=8, normalize=True, value_range=(-1., 1.))
            tm1 = t.sub(1).div(self.sample_timesteps)
            pred_x_0 = self.p_pred_x_0(teacher_denoise_fn, x_tm1, tm1, y, clip_denoised=True)
            save_image_cuda(pred_x_0, "result_pred_x.png", nrow=8, normalize=True, value_range=(-1., 1.))
            w = 1 + torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t)
            # w = torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t).sqrt()
        # calculate the loss
        # mse: re-weighted
        if self.loss_type == "mse":
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)
            t = t.div(self.sample_timesteps)
            model_out = student_diffusion.p_pred_x_0(student_denoise_fn, x_t, t, y=y, clip_denoised=True)
            save_image_cuda(model_out, "result.png", nrow=8, normalize=True, value_range=(-1., 1.))
            losses = flat_mean((w * pred_x_0 - w * model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)
        return losses
    
    def distill_losses_enc(self, student_diffusion, teacher_denoise_fn, student_denoise_fn, 
                x_0, t, y, speed_up, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        with torch.no_grad():
            logsnr_t, = self.t2logsnr(t, x=x_0)
            x_t = q_sample(x_0, logsnr_t, eps=noise)
            s = None
            if self.loss_type == "kl":
                t = torch.ceil(t * self.sample_timesteps)
                s = t.sub(1).div(self.sample_timesteps)
                t = t.div(self.sample_timesteps)
            # t = t.mul(self.sample_timesteps)
            # save_image_cuda(x_t, "result_xt.png", nrow=8, normalize=True, value_range=(-1., 1.))
            # x_tm1 = self.p_sample_distill_step(teacher_denoise_fn, x_t, step=t, y=None, 
            #                     x_0=x_0, clip_denoised=True, return_pred=False, use_ddim=True)
            # save_image_cuda(x_tm1, "result_xtm1.png", nrow=8, normalize=True, value_range=(-1., 1.))
            # tm1 = t.sub(1).div(self.sample_timesteps)
            tm1 = t.sub(0.1)
            logsnr_s, = self.t2logsnr(tm1, x=x_0)
            x_tm1 = q_sample(x_0, logsnr_s, eps=noise)
            # pred_x_0 = self.p_pred_x_0(teacher_denoise_fn, x_tm1, tm1, y, clip_denoised=True)
            pred_x_0, attn_map, pred_eps = self.p_pred_x_0_enc(teacher_denoise_fn, x_tm1, tm1, y=None, x_start=x_0, clip_denoised=True)
            save_image_cuda(pred_x_0, "result_pred_x_1.png", nrow=8, normalize=True, value_range=(-1., 1.))
            blur_sigma = self.blur_sigma
            mask_blurred = self.attention_masking(
                    pred_x_0,
                    tm1,
                    attn_map,
                    # prev_noise=pred_eps,
                    x_real=x_0,
                    blur_sigma=blur_sigma,
                )
            mask_blurred = q_sample(mask_blurred, logsnr_s, eps=pred_eps)
            _, _, uncond_eps = self.p_pred_x_0_enc(teacher_denoise_fn, mask_blurred, tm1, None, pred_x_0, clip_denoised=True)
            guided_eps = uncond_eps + self.w_guide * (pred_eps - uncond_eps)
            pred_x_0_target = pred_x0_from_eps(x_tm1, guided_eps, logsnr_s)
            save_image_cuda(pred_x_0_target, "result_pred_x_2.png", nrow=8, normalize=True, value_range=(-1., 1.))
            # pred_x_0 += self.w_guide * (pred_x_0 - pred_x_0_attn)
            # save_image_cuda(pred_x_0_attn, "result_pred_x_3.png", nrow=8, normalize=True, value_range=(-1., 1.))
            w = 1 + torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t)
            # w = torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t).sqrt()
        # calculate the loss
        # mse: re-weighted
        if self.loss_type == "mse":
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)
            # t = t.div(self.sample_timesteps)
            model_out, attn_map, pred_eps = student_diffusion.p_pred_x_0_enc(student_denoise_fn, x_t, t, y=None, x_start=x_0, clip_denoised=True)
            save_image_cuda(model_out, "result.png", nrow=8, normalize=True, value_range=(-1., 1.))
            losses = flat_mean((w * pred_x_0_target - w * model_out).pow(2))
        elif self.loss_type == "kl":
            # print("jjj")
            losses = self._loss_term_bpd(
                student_denoise_fn, x_0=x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=False, return_pred=False)
        else:
            raise NotImplementedError(self.loss_type)
        return losses

    def train_losses(self, denoise_fn, x_0, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        s = None
        if self.loss_type == "kl":
            t = torch.ceil(t * self.sample_timesteps)
            s = t.sub(1).div(self.sample_timesteps)
            t = t.div(self.sample_timesteps)

        # calculate the loss
        # kl: un-weighted
        # mse: re-weighted

        logsnr_t, = self.t2logsnr(t, x=x_0)
        x_t = q_sample(x_0, logsnr_t, eps=noise)
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(
                denoise_fn, x_0=x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=False, return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            assert self.reweight_type in {"constant", "snr", "truncated_snr", "alpha2"}
            target = {
                "constant": x_0,
                "snr": noise,
                "truncated_snr": (x_0, noise),
                "alpha2": pred_v_from_x0eps(x_0, noise, logsnr_t)
            }[self.reweight_type]
            
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)

            model_out = denoise_fn(x_t, t, y=y, x_start=x_0).pred
            predict = self.from_model_out_to_pred(
                x_t, model_out, logsnr_t
            )[self.reweight_type]
            

            # save_image_cuda(predict - target)
            # save_image_cuda(predict, "x.png")
            
            if isinstance(target, tuple):
                assert len(target) == 2
                losses = torch.maximum(*[
                    flat_mean((tgt - pred).pow(2))
                    for tgt, pred in zip(target, predict)])
            else:
                losses = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        B = x_0.shape[0]
        t = torch.ones([B, ], dtype=torch.float32)
        logsnr_t, = self.t2logsnr(t, x=x_0)
        T_mean, T_logvar = q_mean_var(x_0=x_0, logsnr_t=logsnr_t)
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, y, clip_denoised=True):
        B, T = x_0.shape, self.sample_timesteps
        s = torch.empty([B, ], dtype=torch.float32)
        t = torch.empty([B, ], dtype=torch.float32)
        losses = torch.zeros([B, T], dtype=torch.float32)
        mses = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            s.fill_(i / self.sample_timesteps)
            t.fill_((i + 1) / self.sample_timesteps)
            logsnr_t, = self.t2logsnr(t)
            x_t = q_sample(x_0, logsnr_t=logsnr_t)
            loss, pred_x_0 = self._loss_term_bpd(
                denoise_fn, x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=clip_denoised, return_pred=True)
            losses[:, i] = loss
            mses[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(losses, dim=1) + prior_bpd
        return total_bpd, losses, prior_bpd, mses


if __name__ == "__main__":
    DEBUG = True


    def test_logsnr_to_posterior():
        logsnr_schedule = get_logsnr_schedule("cosine")
        logsnr_s = logsnr_schedule(torch.as_tensor(0.))
        logsnr_t = logsnr_schedule(torch.as_tensor(1. / 1000))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))
        logsnr_s = logsnr_schedule(torch.as_tensor(999. / 1000))
        logsnr_t = logsnr_schedule(torch.as_tensor(1.))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))


    def test_logsnr_to_posterior_ddim():
        logsnr_schedule = get_logsnr_schedule("cosine")
        t = torch.linspace(0, 1, 1001, dtype=torch.float32)
        print(logsnr_schedule(t))
        logsnr_s = logsnr_schedule(t[:-1])
        logsnr_t = logsnr_schedule(t[1:])
        mean_coef1, mean_coef2, logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, "fixed_small")
        mean_coef1_, mean_coef2_, logvar_ = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=1.)
        print(
            torch.allclose(mean_coef1, mean_coef1_),
            torch.allclose(mean_coef2, mean_coef2_),
            torch.allclose(logvar, logvar_))


    def test_legacy():
        logsnr_schedule = get_logsnr_schedule("legacy")
        t = torch.linspace(0, 1, 1000, dtype=torch.float32)
        print(torch.sigmoid(logsnr_schedule(t))[::10])
        print(logsnr_schedule(t)[::10])
        t = torch.rand(10000, dtype=torch.float32)
        print(logsnr_schedule(t))

    # run tests
    TESTS = [test_logsnr_to_posterior, test_logsnr_to_posterior_ddim, test_legacy]
    TEST_INDICES = []
    for i in TEST_INDICES:
        TESTS[i]()


def data_norm(x):
    x_min = x.min()
    if x_min<0 :
        x+=torch.abs(x_min)
        x_min=x.min()
    x_max=x.max()
    dst=x_max-x_min
    norm=(x-x_min).true_divide(dst)
    return norm