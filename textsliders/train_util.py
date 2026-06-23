from typing import Optional, Union, Literal, List, Tuple

import torch

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, SchedulerMixin

from tqdm import tqdm


AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None  # if you want to change the cache dir, change this


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    # print(height, width)
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        generator=generator,
        device="cpu",
    )


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma

    return latents


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: List[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: List[str],
):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)
    
    

    return text_embeddings


def encode_prompts_slider(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: List[str],
    num_images_per_prompt: int = 1,
    sc: float = 1.0,
):

    text_tokens = text_tokenize(tokenizer, prompts)
    idx = text_tokens.argmax(-1)
    text_embeddings = text_encode(text_encoder, text_tokens)
    batch_indices = torch.arange(len(text_tokens))
    text_embeddings[batch_indices, idx, :] = sc * text_embeddings[batch_indices, idx, :]
        
    

    return text_embeddings


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompts_xl(
    tokenizers: List[CLIPTokenizer],
    text_encoders: List[SDXL_TEXT_ENCODER_TYPE],
    prompts: List[str],
    num_images_per_prompt: int = 1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def encode_prompts_xl_slider(
    tokenizers: List[CLIPTokenizer],
    text_encoders: List[SDXL_TEXT_ENCODER_TYPE],
    prompts: List[str],
    num_images_per_prompt: int = 1,
    sc: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool
    k = 0
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        
        idx = text_tokens_input_ids.argmax(-1)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )
        batch_indices = torch.arange(len(text_tokens_input_ids))
        if k == 0:
            text_embeds[batch_indices, idx, :] = sc * text_embeds[batch_indices, idx, :]
        
        text_embeds_list.append(text_embeds)
        k += 1

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    # print(unconditional.shape, conditional.shape)
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # print("Predict noise without reference images.")
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target

def predict_noise_with_reference(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.FloatTensor,  # 目标图像的潜在向量 [1, 4, H, W]
    text_embeddings: torch.FloatTensor,  # [2, 77, 768]
    ref_latents_noisy: Optional[torch.FloatTensor] = None,  # 带噪声的参考图像 [N, 4, H, W]
    guidance_scale: float = 7.5,
    **kwargs,
) -> torch.FloatTensor:
    """
    支持参考图像的噪声预测函数
    
    Args:
        unet: UNet模型
        scheduler: 调度器
        timestep: 当前时间步
        latents: 目标图像的潜在向量
        text_embeddings: 文本嵌入
        ref_latents_noisy: 带噪声的参考图像潜在向量
        guidance_scale: 引导强度
    """
    
    # 如果没有参考图像，使用标准predict_noise
    if ref_latents_noisy is None:
        # print("666")
        res=predict_noise(unet, scheduler, timestep, latents, text_embeddings, guidance_scale, **kwargs)
        # print("no_ref res:", res.shape)
        return res
    # print("Predict noise with reference images.")
    # 拼接目标图像和参考图像的潜在向量
    # print("latents:", latents.shape) #latents: torch.Size([1, 4, 96, 96])
    # print("ref_latents_noisy:", ref_latents_noisy.shape) #ref_latents_noisy: torch.Size([3, 4, 64, 64])
    combined_latents = torch.cat([latents, ref_latents_noisy], dim=0)
    
    # 扩展以支持classifier-free guidance
    latent_model_input = torch.cat([combined_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    
    # 预测噪声 (MRSA会在注意力层中自动处理参考图像信息)
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        **kwargs,
    ).sample
    
    # 执行引导
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    
    # 只返回目标图像的噪声预测 (第一个)
    # print("with_ref guided_target:", guided_target.shape)
    # print("with_ref guided_target[:1]:", guided_target[:1].shape)
    return guided_target[:1]  # [1, 4, H, W]

# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []
    # print("Diffusion without reference images.")
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:  # 移除tqdm
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents

@torch.no_grad()
def diffusion_with_reference(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # 目标图像的初始噪声
    text_embeddings: torch.FloatTensor,
    ref_latents_z_0: Optional[torch.FloatTensor] = None,  # 参考图像的原始潜在编码
    total_timesteps: int = 1000,
    start_timesteps: int = 0,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    """
    支持参考图像的扩散降噪过程
    
    Args:
        unet: UNet模型
        scheduler: 调度器  
        latents: 目标图像的初始噪声 [1, 4, H, W]
        text_embeddings: 文本嵌入 [2, 77, 768]
        ref_latents_z_0: 参考图像的原始潜在编码 [N, 4, H, W]
        total_timesteps: 总时间步数
        start_timesteps: 开始时间步
        guidance_scale: 引导强度
        generator: 随机数生成器
    """
    
    # 如果没有参考图像，使用标准diffusion
    if ref_latents_z_0 is None:
        # print("hhh")
        return diffusion(unet, scheduler, latents, text_embeddings, 
                        total_timesteps, start_timesteps, guidance_scale=guidance_scale, **kwargs)
    
    device = latents.device
    dtype = latents.dtype
    # print("Diffusion with reference images.")
    # 降噪循环
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:  # 移除tqdm
        
        # 为参考图像在当前时间步添加相应的噪声
        noise = torch.randn_like(ref_latents_z_0, device=device, dtype=dtype)
        
        if isinstance(timestep, int):
            timestep_tensor = torch.tensor([timestep], device=device)
        else:
            timestep_tensor = timestep
            
        ref_latents_noisy = scheduler.add_noise(ref_latents_z_0, noise, timestep_tensor)
        
        # print(latents.shape, ref_latents_noisy.shape)
        # 使用带参考图像的噪声预测
        noise_pred = predict_noise_with_reference(
            unet, scheduler, timestep, latents, text_embeddings,
            ref_latents_noisy=ref_latents_noisy, guidance_scale=guidance_scale, **kwargs
        )
        
        # 计算前一个噪声样本 x_t -> x_t-1 (只对目标图像)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    
    return latents

def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg


def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
    noise_pred = rescale_noise_cfg(
        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
    )

    return guided_target


@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: Tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> Tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width
