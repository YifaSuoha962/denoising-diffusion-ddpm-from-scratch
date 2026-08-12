"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    # for 循环
    # betas = [beta_start + (beta_end - beta_start) * i / (T - 1) for i in range(T)]
    # return torch.tensor(betas, dtype=torch.float32)

    # pass

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    
    return 1.0 - betas

    # alpha = torch.ones_like(betas, dtype=betas.dtype, device=betas.device)
    # return alpha - betas
    
    # pass

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)
    # pass

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    # pass

    # by deepseek
    # return a[t].view(-1, 1, 1, 1)

    # offical solution
    # Note: gather 要求 index 为 torch.long 类型
    return a.gather(0, t.long()).reshape(-1, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    # pass
    
    ac_batch = extract_into_batch(alphas_cumprod, t, x0)
    # ac_batch = alphas_cumprod[t]
    # Note: torch 的广播机制是从最里层 (shape的最后一位) 开始的
    return torch.sqrt(ac_batch) * x0 + torch.sqrt(1.0 - ac_batch) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    return {'T':T, 'alphas':alphas, 'alphas_cumprod':alphas_cumprod, 'betas': betas, 
            'sqrt_alphas_cumprod':sqrt_alphas_cumprod, 'sqrt_one_minus_alphas_cumprod':sqrt_one_minus_alphas_cumprod} 
    # pass

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return F.mse_loss(noise_pred, noise)  # torch.mean((noise - noise_pred) ** 2)
    # pass

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    # pass
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    pred_noise = model(x_t, t)

    return noise_prediction_loss(pred_noise, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)

    # sinusoidal timestep embedding of shape (B, dim)
    assert dim % 2 == 0, "dim must be even"

    half = dim // 2
    # i / (half - 1), when half == 1 use exponent 0
    if half == 1:
        exponents = torch.zeros(1, device=t.device)
    else:
        exponents = torch.arange(
            half, device=t.device, dtype=torch.float32
        ) / (half - 1)
    freqs = 10000 ** (-exponents)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb
    # pass

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

