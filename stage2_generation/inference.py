import os
import glob
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler,
    UNet2DConditionModel
)
# [关键修改] 引入 PEFT，这是读取你训练权重的唯一钥匙
from peft import PeftModel
from utils.ink_mask import InkWashMaskGenerator

# ================= 配置区域 =================
# 1. 输入 TXT 文件夹路径
INPUT_TXT_DIR = "/home/610-sty/layout2paint3/outputs/layout/txt"

# 2. 输出结果保存路径 (建议改名区分)
OUTPUT_IMAGE_DIR = "./inference_peft_60000_results"

# 3. 模型路径
BASE_MODEL = "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
# 指向 V18 训练出的 Checkpoint
CHECKPOINT_DIR = "/home/610-sty/layout2paint3/outputs/taiyi_shanshui_v18_hardcore/checkpoint-65000"

CONTROLNET_PATH = os.path.join(CHECKPOINT_DIR, "controlnet_structure")
LORA_PATH = os.path.join(CHECKPOINT_DIR, "unet_lora")

# 4. 参数设置
WIDTH, HEIGHT = 512, 512
GUIDANCE_SCALE = 7.5
NUM_STEPS = 30

# [策略调整] 
# PEFT 融合模式下，LoRA 默认就是 1.0 (全开)。
# ControlNet 设为 0.8，给 V18 的强风格 LoRA 一点发挥空间，但也得管住构图。
CONTROLNET_SCALE = 0.8 

# ================= 解析函数 (保持不变) =================
def parse_layout_txt(file_path):
    layout_data = []
    poem = None
    filename = os.path.basename(file_path)
    poem_from_filename = os.path.splitext(filename)[0]
    poem = poem_from_filename

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("Poem:"):
            content = line.split(":", 1)[1].strip()
            if content: poem = content
            continue
        if "---" in line or "Class" in line and "Box" in line: continue
        parts = line.split('|')
        if len(parts) >= 4:
            try:
                cls_id = int(parts[0].strip())
                box_str = parts[2].strip().replace('(', '').replace(')', '')
                cx, cy, w, h = map(float, box_str.split(','))
                gestalt_str = parts[3].strip().replace('(', '').replace(')', '')
                bx, by, rot, flow = map(float, gestalt_str.split(','))
                item = [cls_id, cx, cy, w, h, bx, by, rot, flow]
                layout_data.append(item)
            except ValueError: continue
    return poem, layout_data

# ================= 主程序 =================
def main():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    # 1. 检查文件
    txt_files = glob.glob(os.path.join(INPUT_TXT_DIR, "*.txt"))
    if not txt_files:
        print(f"❌ 未找到 TXT 文件: {INPUT_TXT_DIR}")
        return
    
    print(f"📂 准备处理 {len(txt_files)} 个文件...")
    print("🚀 启动硬核 PEFT 加载模式...")

    mask_gen = InkWashMaskGenerator(width=WIDTH, height=HEIGHT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================================================
    # [核心修改] 手术式加载模型
    # ========================================================
    
    # 1. 单独加载底座 UNet
    print("   1. 加载 Base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL, subfolder="unet", torch_dtype=torch.float16
    )

    # 2. 使用 PEFT 强行挂载 LoRA
    print(f"   2. 挂载 PEFT LoRA: {LORA_PATH}")
    try:
        # 这步会读取 json 并匹配 target_modules (to_k, to_v...)
        unet = PeftModel.from_pretrained(unet, LORA_PATH)
        # 物理融合：把 LoRA 权重加到 UNet 权重里，变成一个普通的 UNet
        unet = unet.merge_and_unload()
        print("   ✅ LoRA 已成功物理熔合到 UNet！")
    except Exception as e:
        print(f"   ❌ LoRA 挂载失败，请检查路径或 peft 版本: {e}")
        return

    # 3. 加载 ControlNet
    print(f"   3. 加载 ControlNet: {CONTROLNET_PATH}")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float16)

    # 4. 组装 Pipeline (注入魔改后的 UNet)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        unet=unet,             # <--- 这里放的是带 LoRA 魂的 UNet
        controlnet=controlnet, 
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 专用负面提示词
    n_prompt = "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"

    # ========================================================
    
    success_count = 0
    for i, txt_path in enumerate(txt_files):
        print(f"\n[{i+1}/{len(txt_files)}] {os.path.basename(txt_path)}")
        
        poem_prompt, layout_data = parse_layout_txt(txt_path)
        if not layout_data: continue
        
        try:
            control_image = mask_gen.convert_boxes_to_mask(layout_data)
        except Exception as e:
            print(f"   ❌ Mask 生成失败: {e}")
            continue

        # 保存 Mask 对比
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        control_image.save(os.path.join(OUTPUT_IMAGE_DIR, f"{base_name}_mask.png"))

        # 推理
        try:
            image = pipe(
                prompt=poem_prompt,
                image=control_image,
                negative_prompt=n_prompt,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_conditioning_scale=CONTROLNET_SCALE,
                width=WIDTH,
                height=HEIGHT
            ).images[0]

            res_save_path = os.path.join(OUTPUT_IMAGE_DIR, f"{base_name}.png")
            image.save(res_save_path)
            print(f"   ✅ 完成: {res_save_path}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ 推理出错: {e}")

    print(f"\n🎉 全部完成！成功: {success_count} 张。结果在: {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()