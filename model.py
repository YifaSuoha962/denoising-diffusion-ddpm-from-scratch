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
    args = torch.outer(t.float(), freqs)
    # args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb
    # pass

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(
    in_ch: int = 1,
    hidden: int = 16,
    time_dim: int = 16,
    seed: int = 0
) -> dict:
    # 固定随机种子
    torch.manual_seed(seed)

    params = {
        # conv 3x3: in_ch -> hidden
        'conv_in_w': (torch.randn(hidden, in_ch, 3, 3) * 0.02).requires_grad_(),
        'conv_in_b': torch.zeros(hidden, requires_grad=True),

        # timestep embedding: time_dim -> hidden
        'time_mlp_w': (torch.randn(hidden, time_dim) * 0.02).requires_grad_(),
        'time_mlp_b': torch.zeros(hidden, requires_grad=True),

        # middle conv: hidden -> hidden
        'conv_mid_w': (torch.randn(hidden, hidden, 3, 3) * 0.02).requires_grad_(),
        'conv_mid_b': torch.zeros(hidden, requires_grad=True),

        # output conv: hidden -> in_ch
        'conv_out_w': (torch.randn(in_ch, hidden, 3, 3) * 0.02).requires_grad_(),
        'conv_out_b': torch.zeros(in_ch, requires_grad=True),
    }

    return params

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise

    # 1. conv 3x3: in_ch -> hidden
    h = F.conv2d(
        x,
        params['conv_in_w'],
        params['conv_in_b'],
        padding=1
    )  # shape = [bz, c, h, w]

    # 2. timestep embedding
    time_dim = params['time_mlp_w'].shape[1]
    temb = timestep_embedding(t, time_dim)

    # time MLP: time_dim -> hidden
    temb = F.linear(
        temb,
        params['time_mlp_w'],
        params['time_mlp_b']
    )
    temb = F.relu(temb)

    # (B, hidden) -> (B, hidden, 1, 1)，然后广播到 H,W
    h = h + temb[:, :, None, None]      # Note: 和之前 alpha_cumprod 一样,要乘到每张图片上

    # 3. ReLU + middle conv + ReLU
    h = F.relu(h)

    h = F.conv2d(
        h,
        params['conv_mid_w'],
        params['conv_mid_b'],
        padding=1
    )
    h = F.relu(h)

    # 4. output conv: hidden -> in_ch
    out = F.conv2d(
        h,
        params['conv_out_w'],
        params['conv_out_b'],
        padding=1
    )

    return out

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    # 固定随机种子
    torch.manual_seed(seed)
    # 圆盘半径
    radius = size // 4
    # 初始化黑色背景
    images = torch.zeros(n, 1, size, size, dtype=torch.float32)
    # 像素坐标网格
    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )

    for i in range(n):
        # 随机圆心：(cy, cx)
        center = torch.randint(
            radius,
            size - radius,
            (2,)
        )

        cy, cx = center[0], center[1]

        # 判断每个像素是否位于圆盘内部
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

        # 圆盘像素设为 1
        images[i, 0][mask] = 1.0

    return images

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    # pass
    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule['T']
    t = torch.randint(0, T, (B,), device=x0.device)

    # sample noise 
    noise = torch.randn_like(x0)

    # compute loss 
    loss = diffusion_training_loss(lambda x, t: tiny_unet_forward(x, t, params),
                                     x0, t, noise, schedule['alphas_cumprod'])
    loss.backward()
    
    # 手动 SGD 更新参数
    new_params = {}
    for name, p in params.items():
        if p.grad is not None:
            p_new = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            p_new = p.clone().detach().requires_grad_(True)
        new_params[name] = p_new
    return new_params, float(loss.detach())

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    history = []
    cur_params = params

    for step in range(num_steps):
        # 每一步使用不同但可复现的随机种子
        step_seed = seed + step
        torch.manual_seed(step_seed)
        # 随机采样sample索引
        indices = torch.randint(0, len(dataset), (batch_size,))
        # 获取对应的 minibatch
        batch = dataset[indices]
        updated_params, cur_loss = ddpm_train_step(params=cur_params, 
                                                   x0=batch, 
                                                   schedule=schedule, 
                                                   lr=lr, 
                                                   seed=step_seed)
        history.append(cur_loss)
        cur_params = updated_params

    return cur_params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    x0_hat = (x_t - torch.sqrt(1-alphas_cumprod[t].reshape(-1, 1, 1, 1)) * eps) / torch.sqrt(alphas_cumprod[t].reshape(-1, 1, 1, 1))
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # 1. 从预测噪声恢复 x0，并限制到 [-1, 1]
    x0_hat = predict_x0_from_eps(
        x_t,
        t,
        eps,
        schedule['alphas_cumprod']
    ).clamp(-1.0, 1.0)

    # 2. 提取当前 timestep 对应的 alpha_t, beta_t, alpha_bar_t
    # 为了broadcast, 直接schedule['alphas']
    alpha_t = extract_into_batch(
        schedule['alphas'],
        t,
        x_t
    )

    beta_t = extract_into_batch(
        schedule['betas'],
        t,
        x_t
    )

    alpha_bar_t = extract_into_batch(
        schedule['alphas_cumprod'],
        t,
        x_t
    )

    # 3. 构造 alpha_bar_{t-1}
    #    t == 0 时按照约定 alpha_bar_{-1} = 1
    t_prev = torch.clamp(t - 1, min=0)

    alpha_bar_prev = extract_into_batch(
        schedule['alphas_cumprod'],
        t_prev,
        x_t
    )

    # 对 t == 0 的样本覆盖为 1
    is_t0 = (t == 0).reshape(-1, 1, 1, 1)

    alpha_bar_prev = torch.where(
        is_t0,
        torch.ones_like(alpha_bar_prev),
        alpha_bar_prev
    )

    # 4. posterior mean 的两个系数
    coef_x0 = (
        torch.sqrt(alpha_bar_prev)
        * beta_t
        / (1.0 - alpha_bar_t)
    )

    coef_xt = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )

    # 5. posterior mean
    mean = coef_x0 * x0_hat + coef_xt * x_t

    # 6. fixed variance: sigma_t^2 = beta_t
    variance = beta_t

    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # 1. 预测当前 x_t 中的噪声 epsilon
    eps = tiny_unet_forward(x_t, t, params)

    # 2. 计算反向过程 Gaussian 的 mean 和 variance
    mean, var, _ = ddpm_p_mean_variance(
        x_t,
        t,
        eps,
        schedule
    )

    # 3. 如果没有提供 noise，则采样标准高斯噪声
    if noise is None:
        noise = torch.randn_like(x_t)

    # t == 0 时最后一步不再加入随机噪声
    # shape: (B,) -> (B,1,1,1)
    nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)

    noise = noise * nonzero_mask

    # x_{t-1} = mean + sqrt(var) * noise
    x_prev = mean + torch.sqrt(var) * noise

    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    # 固定随机种子
    torch.manual_seed(seed)

    # 从纯高斯噪声开始
    x = torch.randn(shape)

    B = shape[0]
    T = schedule['T']

    # range(st, ed, step)
    for t in range(T-1, -1, -1):
        # Note-1: 当前 batch 中所有样本使用同一个 timestep
        t_batch = torch.full(
            (B,),
            t,
            dtype=torch.long,
            device=x.device
        )
        # denoise
        x = ddpm_p_sample(x, t_batch, params, schedule)

    return x

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

