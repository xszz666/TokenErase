#!/usr/bin/env python
# coding: utf-8
"""
Stable Diffusion Inference with Custom Text Embeddings
Open-source version for artistic style generation
"""

import torch
from PIL import Image
import os
from tqdm import tqdm
import gc
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
import safetensors.torch


def flush():
    """Clear GPU cache"""
    torch.cuda.empty_cache()
    gc.collect()


def main():
    # Configuration
    pretrained_model_name_or_path = "/path/to/stable-diffusion-v1-5"
    learned_embeds_path = "/path/to/learned_embeds.safetensors"
    
    # Inference parameters
    prompts = [
        "a painting in Van Gogh style",
        "a painting in Picasso style",
    ]
    
    num_images_per_prompt = 5
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5
    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32
    
    # Initialize models
    print("Loading models...")
    noise_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        num_train_timesteps=1000
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet"
    )
    
    # Load custom embeddings
    print(f"Loading custom embeddings from {learned_embeds_path}...")
    new_token = "<custom>"
    loaded_embeds = safetensors.torch.load_file(learned_embeds_path)
    
    if new_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([new_token])
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)
    keyy = list(loaded_embeds.keys())[0]
    new_token_embed = loaded_embeds[keyy]
    
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight.data[new_token_id] = new_token_embed.clone()
    
    # Freeze models
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    
    # Create output directory
    output_path = "outputs/generated_images"
    os.makedirs(output_path, exist_ok=True)
    
    # Generate images
    print(f"Generating {len(prompts)} prompts x {num_images_per_prompt} images...")
    
    for prompt_idx, prompt in enumerate(prompts):
        full_prompt = f"{prompt}, {new_token}"
        
        for img_idx in range(num_images_per_prompt):
            print(f"\n[{prompt_idx + 1}/{len(prompts)}] Prompt: {prompt}")
            print(f"  Image {img_idx + 1}/{num_images_per_prompt}")
            
            generator = torch.manual_seed(seed + img_idx)
            
            # Tokenize prompt
            text_input = tokenizer(
                full_prompt, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Prepare unconditioned embeddings
            uncond_input = tokenizer(
                "", 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            
            # Initialize latents
            latents = torch.randn(
                (1, unet.in_channels, height // 8, width // 8), 
                generator=generator
            ).to(device)
            
            noise_scheduler.set_timesteps(ddim_steps)
            latents = latents * noise_scheduler.init_noise_sigma
            latents = latents.to(weight_dtype)
            
            # Denoising loop
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            
            for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
                concat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input, 
                        t, 
                        encoder_hidden_states=concat_text_embeddings
                    ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode image
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(images[0])
            
            # Save image
            filename = f"{output_path}/prompt_{prompt_idx:02d}_image_{img_idx:02d}_seed_{seed + img_idx}.png"
            pil_image.save(filename)
            print(f"  Saved: {filename}")
            
            flush()
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()