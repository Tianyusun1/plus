# File: stage2_generation/scripts/train_taiyi.py (V18.2: Balanced Soft-Control)

import argparse
import logging
import os
import math
import random
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# =========================================================
# [PATCH] 修复受限环境下的 PermissionError (完整保留)
# =========================================================
try:
    EnvironClass = os.environ.__class__
    _orig_setitem = EnvironClass.__setitem__
    _orig_delitem = EnvironClass.__delitem__

    def _safe_setitem(self, key, value):
        try:
            _orig_setitem(self, key, value)
        except PermissionError:
            pass
        except Exception as e:
            raise e

    def _safe_delitem(self, key):
        try:
            _orig_delitem(self, key)
        except PermissionError:
            pass
        except KeyError:
            pass
        except Exception as e:
            raise e

    EnvironClass.__setitem__ = _safe_setitem
    EnvironClass.__delitem__ = _safe_delitem
    
    def _safe_clear(self):
        keys = list(self.keys())
        for key in keys:
            self.pop(key, None)
            
    EnvironClass.clear = _safe_clear
    print("✅ Environment monkey-patch applied successfully.")
except Exception as e:
    print(f"⚠️ Failed to patch environment: {e}")

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
import itertools 

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)

# [NEW] Min-SNR Loss 计算辅助函数
def compute_snr(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    # 扩展 alpha 到 timestep 维度
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    # [修改] 输出目录更新为 v18_soft_balanced
    parser.add_argument("--output_dir", type=str, default="taiyi_shanshui_v18_soft_balanced")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=50) 
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    # [修改] 降低 LoRA 学习率至 5e-5，追求更细腻纹理，防止过拟合
    parser.add_argument("--learning_rate_lora", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    
    parser.add_argument("--lambda_struct", type=float, default=0.0)
    parser.add_argument("--lambda_energy", type=float, default=0.0)
    
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha_ratio", type=float, default=0.05)
    parser.add_argument("--smart_freeze", action="store_true", default=False)
    
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR 权重")
    parser.add_argument("--offset_noise_scale", type=float, default=0.1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        # [修改] 更新日志信息
        logger.info(f"🚀 V18.2 启动: 软硬平衡版 | Rank 128 | Mask Dropout 25% | Scale 0.85")

    # 1. 加载模型
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    # 2. 冻结策略
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) 
    
    lora_alpha = args.lora_rank * args.lora_alpha_ratio
    unet_lora_config = LoraConfig(
        r=args.lora_rank, 
        lora_alpha=lora_alpha, 
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    # ControlNet 策略
    if args.smart_freeze:
        if accelerator.is_main_process: logger.info("❄️ 警告: Smart Freeze 已开启，可能会丢失颜色语义！")
        controlnet.requires_grad_(False) 
        for n, p in controlnet.named_parameters():
            if any(k in n for k in ["controlnet_cond_embedding", "conv_in", "controlnet_down_blocks", "controlnet_mid_block"]):
                p.requires_grad = True
    else:
        if accelerator.is_main_process: logger.info("🔥 状态: ControlNet 全量解冻 - 学习语义布局映射")
        controlnet.requires_grad_(True)

    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, controlnet.parameters()), "lr": args.learning_rate},
        {"params": filter(lambda p: p.requires_grad, unet.parameters()), "lr": args.learning_rate_lora} 
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据加载
    raw_dataset = load_dataset("json", data_files=os.path.join(args.train_data_dir, "train.jsonl"))["train"]
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    if accelerator.is_main_process:
        logger.info(f"📊 数据集分布: 训练集 {len(train_dataset)} | 验证集 {len(test_dataset)}")

    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    cond_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), 
    ])

    null_prompt = tokenizer("", max_length=tokenizer.model_max_length, 
                            padding="max_length", truncation=True, return_tensors="pt")
    null_prompt_ids = null_prompt.input_ids[0]
    null_prompt_mask = null_prompt.attention_mask[0]

    def collate_fn(examples):
        pixel_values, cond_pixel_values, input_ids, attention_masks = [], [], [], []
        texts = []
        for example in examples:
            try:
                img_path = os.path.join(args.train_data_dir, example["image"])
                cond_path = os.path.join(args.train_data_dir, example["conditioning_image"])
                pixel_values.append(transform(Image.open(img_path).convert("RGB")))
                cond_pixel_values.append(cond_transform(Image.open(cond_path).convert("RGB")))
                
                caption = example["text"] 
                texts.append(caption)
                
                inputs = tokenizer(caption, max_length=tokenizer.model_max_length, 
                                 padding="max_length", truncation=True, return_tensors="pt")
                input_ids.append(inputs.input_ids[0])
                attention_masks.append(inputs.attention_mask[0])
                
            except Exception as e: continue
            
        if len(pixel_values) == 0: return None

        return {
            "pixel_values": torch.stack(pixel_values),
            "conditioning_pixel_values": torch.stack(cond_pixel_values),
            "input_ids": torch.stack(input_ids),
            "attention_masks": torch.stack(attention_masks),
            "texts": texts
        }

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500, 
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )

    controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 5. 训练循环
    global_step = 0
    for epoch in range(args.num_train_epochs):
        controlnet.train(); unet.train()
        for step, batch in enumerate(train_dataloader):
            if batch is None: continue 

            with accelerator.accumulate(controlnet, unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                if args.offset_noise_scale > 0:
                    noise += args.offset_noise_scale * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
                
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # [关键修改: 调整训练比例以强化布局控制] 
                rand_num = random.random()
                
                current_ids = batch["input_ids"]
                current_mask = batch["attention_masks"]
                cond_input = batch["conditioning_pixel_values"].to(dtype=torch.float16)

                if rand_num < 0.1:
                    # [Case A] 10% 丢弃 Text (保持新锚点 Mask 的引导)
                    current_ids = null_prompt_ids.repeat(len(batch["input_ids"]), 1).to(device)
                    current_mask = null_prompt_mask.repeat(len(batch["input_ids"]), 1).to(device)
                
                # [修改] 将丢弃 Mask 的阈值调整为 0.35 (即 25% 概率)
                # 提高纯文本训练比例，强迫 LoRA 脑补细节，避免过度依赖硬边
                elif rand_num < 0.35: 
                    # [Case B] 25% 丢弃 Mask -> 允许 LoRA 继续学习纯文本泛化
                    cond_input = torch.zeros_like(cond_input)
                
                # [Case C] 剩下的 65% 强制进行全监督训练 (Text + Anchor Mask)

                encoder_hidden_states = text_encoder(current_ids, attention_mask=current_mask)[0]
                
                down_res, mid_res = controlnet(noisy_latents, timesteps, encoder_hidden_states, cond_input, return_dict=False)
                
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, 
                    down_block_additional_residuals=[s.to(dtype=torch.float16) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=torch.float16)
                ).sample

                if args.snr_gamma == 0:
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                else:
                    snr = compute_snr(timesteps, scheduler)
                    base_weight = torch.stack([snr, args.snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]
                    
                    if scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = base_weight / (snr + 1e-5)
                    elif scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = base_weight / (snr + 1)
                    
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = list(filter(lambda p: p.requires_grad, controlnet.parameters())) + \
                                     list(filter(lambda p: p.requires_grad, unet.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                current_lr = optimizer.param_groups[1]["lr"] 
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | LoRA LR: {current_lr:.2e}")

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir / "controlnet_structure") 
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")

        if accelerator.is_main_process:
            controlnet.eval(); unet.eval()
            try:
                with torch.no_grad(), torch.autocast("cuda"):
                    pipe = StableDiffusionControlNetPipeline(
                        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet), controlnet=accelerator.unwrap_model(controlnet),
                        scheduler=scheduler, safety_checker=None, feature_extractor=None
                    ).to(device)
                    pipe.set_progress_bar_config(disable=True)
                    
                    # [修改] 增加负向提示词，修饰过硬的边界
                    val_neg = "hard edges, sticker-like, flat color, cartoon, split screen, low quality, bad anatomy, 真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"
                    
                    idx = random.randint(0, len(test_dataset)-1)
                    test_sample = test_dataset[idx]
                    
                    val_img_path = os.path.join(args.train_data_dir, test_sample["conditioning_image"])
                    val_cond_img = Image.open(val_img_path).convert("RGB").resize((args.resolution, args.resolution))
                    val_cond_tensor = cond_transform(val_cond_img).unsqueeze(0).to(device, dtype=torch.float16)
                    val_prompt = test_sample["text"]
                    
                    print(f"📷 正在验证 (Soft Balanced): {val_prompt}")

                    sample_img = pipe(
                        prompt=val_prompt, 
                        negative_prompt=val_neg, 
                        image=val_cond_tensor,
                        num_inference_steps=30, 
                        # [修改] 权重退回 0.85，给 LoRA 留出空间
                        controlnet_conditioning_scale=0.85, 
                        # [修改] 最后 30% 步数撤销控制，让边缘自然晕染
                        control_guidance_end=0.7 
                    ).images[0]
                    
                    sample_img.save(Path(args.output_dir) / f"val_epoch_{epoch+1}_step_{global_step}.png")
                    val_cond_img.save(Path(args.output_dir) / f"val_epoch_{epoch+1}_layout.png")
                    print(f"📷 验证图已保存。")
                    
                    del pipe
                    torch.cuda.empty_cache()
                    
            except Exception as e: 
                print(f"⚠️ 采样失败: {e}")
                if 'pipe' in locals(): del pipe
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(Path(args.output_dir) / "controlnet_structure")
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        print(f"✅ V18.2 训练完成 (Balanced Soft-Control Mode)。")

if __name__ == "__main__":
    main()