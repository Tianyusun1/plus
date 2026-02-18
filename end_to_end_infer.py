# import os
# import sys
# import torch
# import yaml
# import argparse
# import random
# import copy
# import re
# import numpy as np
# from PIL import Image
# from transformers import BertTokenizer

# # --- Diffusers & PEFT Imports ---
# from diffusers import (
#     StableDiffusionControlNetPipeline, 
#     ControlNetModel, 
#     UniPCMultistepScheduler,
#     UNet2DConditionModel
# )
# from peft import PeftModel

# # ==========================================
# # 1. 环境与路径设置
# # ==========================================
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(current_script_path)
# sys.path.insert(0, project_root)

# try:
#     from models.poem2layout import Poem2LayoutGenerator
#     from inference.greedy_decode import greedy_decode_poem_layout
#     from stage2_generation.utils.ink_mask import InkWashMaskGenerator
# except ImportError as e:
#     print(f"[Error] 模块导入失败: {e}")
#     sys.exit(1)

# # ==========================================
# # [核心] 50句全风格测试集
# # ==========================================
# BATCH_POEMS = [
#     # --- 孤独与寒江 (Winter/Isolation) ---
#     "千山鸟飞绝，万径人踪灭。",
#     "孤舟蓑笠翁，独钓寒江雪。",
#     "柴门闻犬吠，风雪夜归人。",
#     "日暮苍山远，天寒白屋贫。",
    
#     # --- 壮丽山河 (Majestic Landscape) ---
#     "白日依山尽，黄河入海流。",
#     "两岸猿声啼不住，轻舟已过万重山。",
#     "飞流直下三千尺，疑是银河落九天。",
#     "日照香炉生紫烟，遥看瀑布挂前川。",
#     "星垂平野阔，月涌大江流。",
#     "无边落木萧萧下，不尽长江滚滚来。",
    
#     # --- 田园与人家 (Pastoral/Village) ---
#     "远上寒山石径斜，白云生处有人家。",
#     "绿树村边合，青山郭外斜。",
#     "枯藤老树昏鸦，小桥流水人家。",
#     "旧时王谢堂前燕，飞入寻常百姓家。",
#     "暧暧远人村，依依墟里烟。",
#     "种豆南山下，草盛豆苗稀。",
    
#     # --- 春日生机 (Spring/Lively) ---
#     "两只黄鹂鸣翠柳，一行白鹭上青天。",
#     "竹外桃花三两枝，春江水暖鸭先知。",
#     "几处早莺争暖树，谁家新燕啄春泥。",
#     "草长莺飞二月天，拂堤杨柳醉春烟。",
#     "乱花渐欲迷人眼，浅草才能没马蹄。",
#     "春色满园关不住，一枝红杏出墙来。",
#     "小荷才露尖尖角，早有蜻蜓立上头。",
    
#     # --- 秋意与夕阳 (Autumn/Sunset) ---
#     "一道残阳铺水中，半江瑟瑟半江红。",
#     "停车坐爱枫林晚，霜叶红于二月花。",
#     "落霞与孤鹜齐飞，秋水共长天一色。",
#     "月落乌啼霜满天，江枫渔火对愁眠。",
#     "青山横北郭，白水绕东城。",
    
#     # --- 边塞与大漠 (Frontier/Desert) ---
#     "大漠孤烟直，长河落日圆。",
#     "北风卷地白草折，胡天八月即飞雪。",
#     "黄沙百战穿金甲，不破楼兰终不还。",
#     "秦时明月汉时关，万里长征人未还。",
#     "大漠沙如雪，燕山月似钩。",
    
#     # --- 楼台与烟雨 (Architecture/Rain) ---
#     "南朝四百八十寺，多少楼台烟雨中。",
#     "清明时节雨纷纷，路上行人欲断魂。",
#     "天街小雨润如酥，草色遥看近却无。",
#     "水光潋滟晴方好，山色空蒙雨亦奇。",
#     "危楼高百尺，手可摘星辰。",
#     "欲穷千里目，更上一层楼。",
#     "黄鹤楼中吹玉笛，江城五月落梅花。",
    
#     # --- 静谧与禅意 (Zen/Quiet) ---
#     "人闲桂花落，夜静春山空。",
#     "深林人不知，明月来相照。",
#     "明月松间照，清泉石上流。",
#     "野旷天低树，江清月近人。",
#     "众鸟高飞尽，孤云独去闲。",
#     "采菊东篱下，悠然见南山。",
    
#     # --- 离别与愁绪 (Parting) ---
#     "劝君更尽一杯酒，西出阳关无故人。",
#     "孤帆远影碧空尽，唯见长江天际流。",
#     "春风又绿江南岸，明月何时照我还。",
#     "姑苏城外寒山寺，夜半钟声到客船。"
# ]

# # ==========================================
# # 2. 辅助函数
# # ==========================================

# def calculate_total_iou(boxes_tensor):
#     if boxes_tensor.size(0) < 2: return 0.0
#     x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
#     x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
#     y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
#     y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
    
#     n = boxes_tensor.size(0)
#     total_inter = 0.0
#     for i in range(n):
#         for j in range(i + 1, n):
#             xx1 = max(x1[i], x1[j]); yy1 = max(y1[i], y1[j])
#             xx2 = min(x2[i], x2[j]); yy2 = min(y2[i], y2[j])
#             w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
#             total_inter += w * h
#     return total_inter

# def apply_random_symmetry(layout, device='cpu', attempt_prob=0.5):
#     if not layout: return layout
#     boxes_data = [list(item[1:5]) for item in layout] 
#     boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32).to(device)
#     initial_iou = calculate_total_iou(boxes_tensor)
    
#     new_layout = copy.deepcopy(layout)
#     current_boxes = boxes_tensor.clone()
    
#     indices = list(range(len(layout)))
#     random.shuffle(indices)
    
#     for idx in indices:
#         if random.random() > attempt_prob: continue
#         original_item = new_layout[idx]
#         original_box = current_boxes[idx].clone()
        
#         new_cx = 1.0 - original_item[1]
#         item_list = list(original_item)
#         item_list[1] = new_cx
        
#         if len(item_list) >= 9:
#             item_list[5] = -item_list[5] # bias_x
#             item_list[7] = -item_list[7] # rotation
        
#         current_boxes[idx, 0] = new_cx
#         new_iou = calculate_total_iou(current_boxes)
        
#         if new_iou <= initial_iou + 1e-4: 
#             new_layout[idx] = tuple(item_list)
#             initial_iou = new_iou 
#         else:
#             current_boxes[idx] = original_box 
            
#     return new_layout

# def sanitize_filename(text):
#     """提取汉字作为文件名"""
#     safe_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
#     return safe_text[:10] if safe_text else "untitled_poem"

# # ==========================================
# # 3. 模型管线类
# # ==========================================

# class ShanshuiPipeline:
#     def __init__(self, args):
#         self.device = args.device
#         self.args = args
        
#         print("\n🚀 初始化全流程生成管线...")
#         self.layout_model, self.tokenizer = self._load_layout_model()
#         self.sd_pipe = self._load_sd_pipeline()
#         self.mask_generator = InkWashMaskGenerator(width=args.width, height=args.height)
        
#     def _load_layout_model(self):
#         print(f"   [Stage 1] 加载布局模型: {self.args.layout_config}")
#         with open(self.args.layout_config, "r") as f:
#             config = yaml.safe_load(f)
#         model_config = config['model']
        
#         tokenizer = BertTokenizer.from_pretrained(model_config['bert_path'])
        
#         model = Poem2LayoutGenerator(
#             bert_path=model_config['bert_path'],
#             num_classes=model_config['num_classes'],
#             hidden_size=model_config['hidden_size'],
#             bb_size=model_config['bb_size'],
#             decoder_layers=model_config['decoder_layers'],
#             decoder_heads=model_config['decoder_heads'],
#             dropout=model_config['dropout'],
#             latent_dim=model_config.get('latent_dim', 32)
#         )
        
#         checkpoint = torch.load(self.args.layout_checkpoint, map_location=self.device)
#         state_dict = checkpoint['model_state_dict']
#         new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
#         model.load_state_dict(new_state_dict)
#         model.to(self.device)
#         model.eval()
#         return model, tokenizer

#     def _load_sd_pipeline(self):
#         print(f"   [Stage 2] 加载 SD + ControlNet + LoRA...")
#         unet = UNet2DConditionModel.from_pretrained(
#             self.args.base_sd_path, subfolder="unet", torch_dtype=torch.float16
#         )
        
#         lora_path = os.path.join(self.args.sd_checkpoint_dir, "unet_lora")
#         try:
#             unet = PeftModel.from_pretrained(unet, lora_path)
#             unet = unet.merge_and_unload()
#             print("   ✅ LoRA 融合成功")
#         except Exception as e:
#             print(f"   ❌ LoRA 挂载失败: {e}")
#             sys.exit(1)
            
#         controlnet_path = os.path.join(self.args.sd_checkpoint_dir, "controlnet_structure")
#         controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
#         pipe = StableDiffusionControlNetPipeline.from_pretrained(
#             self.args.base_sd_path,
#             unet=unet,
#             controlnet=controlnet,
#             torch_dtype=torch.float16,
#             safety_checker=None
#         ).to(self.device)
        
#         pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#         if self.device == 'cuda':
#             pipe.enable_model_cpu_offload()
#         return pipe

#     def decode_latents_to_image(self, latents):
#         scaling_factor = self.sd_pipe.vae.config.scaling_factor
#         latents = 1 / scaling_factor * latents
#         with torch.no_grad():
#             image = self.sd_pipe.vae.decode(latents).sample
#         image = (image / 2 + 0.5).clamp(0, 1)
#         image = image.cpu().permute(0, 2, 3, 1).float().numpy()
#         image = (image * 255).round().astype("uint8")
#         return Image.fromarray(image[0])

#     def generate(self, poem_text, seed=None, save_intermediates_dir=None):
#         if seed is not None:
#             random.seed(seed)
#             torch.manual_seed(seed)
#             generator = torch.Generator(device=self.device).manual_seed(seed)
#         else:
#             generator = None

#         print(f"      正在推理: 【{poem_text[:15]}...】")
        
#         # 1. Layout
#         layout = greedy_decode_poem_layout(
#             model=self.layout_model, 
#             tokenizer=self.tokenizer, 
#             poem=poem_text,
#             max_elements=self.args.max_elements, 
#             device=self.device
#         )
        
#         if not layout:
#             print("      ⚠️ 无有效意象，跳过。")
#             return None, None
            
#         layout = apply_random_symmetry(layout, device=self.device, attempt_prob=0.6)

#         # 2. Mask
#         layout_list = [list(item) for item in layout]
#         control_mask = self.mask_generator.convert_boxes_to_mask(layout_list)

#         # 3. Diffusion
#         n_prompt = "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"
        
#         def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
#             if save_intermediates_dir and (step % 5 == 0 or step == self.args.steps - 1):
#                 image = self.decode_latents_to_image(latents)
#                 step_str = str(step).zfill(3)
#                 save_path = os.path.join(save_intermediates_dir, f"step_{step_str}.png")
#                 image.save(save_path)

#         callback = callback_fn if save_intermediates_dir else None
#         callback_steps = 1 if save_intermediates_dir else 0

#         image = self.sd_pipe(
#             prompt=poem_text,
#             image=control_mask,
#             negative_prompt=n_prompt,
#             num_inference_steps=self.args.steps,
#             guidance_scale=self.args.guidance,
#             controlnet_conditioning_scale=self.args.control_scale,
#             width=self.args.width,
#             height=self.args.height,
#             generator=generator,
#             callback=callback,
#             callback_steps=callback_steps
#         ).images[0]
        
#         return image, control_mask

# # ==========================================
# # 4. 主程序入口
# # ==========================================

# def main():
#     parser = argparse.ArgumentParser(description="Poem2Painting Batch 50 Inference")
    
#     # 路径参数
#     parser.add_argument('--layout_checkpoint', type=str, required=True, help="Stage 1 .pth")
#     parser.add_argument('--sd_checkpoint_dir', type=str, required=True, help="Stage 2 Dir")
#     parser.add_argument('--base_sd_path', type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
#     parser.add_argument('--layout_config', type=str, default="configs/default.yaml")
    
#     # 生成参数
#     parser.add_argument('--output_dir', type=str, default="outputs/batch_50_test", help="结果保存目录")
#     parser.add_argument('--width', type=int, default=512)
#     parser.add_argument('--height', type=int, default=512)
#     parser.add_argument('--steps', type=int, default=30)
#     parser.add_argument('--guidance', type=float, default=7.5)
#     parser.add_argument('--control_scale', type=float, default=1.0)
#     parser.add_argument('--max_elements', type=int, default=100)
#     parser.add_argument('--device', type=str, default="cuda")
#     parser.add_argument('--seed', type=int, default=None)
    
#     # 开关
#     parser.add_argument('--save_intermediates', action='store_true', help="保存中间过程")
#     parser.add_argument('--single_poem', type=str, default=None, help="如果设置，只跑这一句")
    
#     args = parser.parse_args()
    
#     # 初始化
#     pipeline = ShanshuiPipeline(args)
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # 确定要跑的列表
#     if args.single_poem:
#         tasks = [args.single_poem]
#         print(f"\n🎯 单句测试模式: {args.single_poem}")
#     else:
#         tasks = BATCH_POEMS
#         print(f"\n📚 批量测试模式: 共 {len(tasks)} 首诗")

#     print(f"📂 结果输出目录: {args.output_dir}\n")
#     print("="*60)

#     # 循环执行
#     success_count = 0
#     for i, poem in enumerate(tasks):
#         safe_name = sanitize_filename(poem)
#         prefix = f"{str(i+1).zfill(2)}_{safe_name}"
        
#         print(f"[{i+1}/{len(tasks)}] 处理: {prefix}")
        
#         # 中间过程目录
#         intermediates_dir = None
#         if args.save_intermediates:
#             intermediates_dir = os.path.join(args.output_dir, f"{prefix}_steps")
#             os.makedirs(intermediates_dir, exist_ok=True)

#         try:
#             final_img, mask_img = pipeline.generate(
#                 poem, 
#                 seed=args.seed,
#                 save_intermediates_dir=intermediates_dir
#             )
            
#             if final_img:
#                 save_path_img = os.path.join(args.output_dir, f"{prefix}_paint.png")
#                 save_path_mask = os.path.join(args.output_dir, f"{prefix}_mask.png")
                
#                 final_img.save(save_path_img)
#                 mask_img.save(save_path_mask)
#                 success_count += 1
#                 print(f"   ✅ 完成")
#             else:
#                 print(f"   ⚠️ 跳过 (空布局)")
                
#         except Exception as e:
#             print(f"   ❌ 失败: {e}")
#             import traceback
#             traceback.print_exc()
        
#         print("-" * 60)

#     print(f"\n🎉 全部完成! 成功: {success_count}/{len(tasks)}")
#     print(f"查看结果: {args.output_dir}")

# if __name__ == "__main__":
#     main()


import os
import sys
import torch
import yaml
import argparse
import random
import copy
import re
import numpy as np
from PIL import Image
from transformers import BertTokenizer

# --- Diffusers & PEFT Imports ---
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler,
    UNet2DConditionModel
)
from peft import PeftModel

# ==========================================
# 1. 环境与路径设置
# ==========================================
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    sys.exit(1)

# ==========================================
# [核心] 50句全风格测试集
# ==========================================
BATCH_POEMS = [
    # --- 孤独与寒江 (Winter/Isolation) ---
    "千山鸟飞绝，万径人踪灭。",
    "孤舟蓑笠翁，独钓寒江雪。",
    "柴门闻犬吠，风雪夜归人。",
    "日暮苍山远，天寒白屋贫。",
    
    # --- 壮丽山河 (Majestic Landscape) ---
    "白日依山尽，黄河入海流。",
    "两岸猿声啼不住，轻舟已过万重山。",
    "飞流直下三千尺，疑是银河落九天。",
    "日照香炉生紫烟，遥看瀑布挂前川。",
    "星垂平野阔，月涌大江流。",
    "无边落木萧萧下，不尽长江滚滚来。",
    
    # --- 田园与人家 (Pastoral/Village) ---
    "远上寒山石径斜，白云生处有人家。",
    "绿树村边合，青山郭外斜。",
    "枯藤老树昏鸦，小桥流水人家。",
    "旧时王谢堂前燕，飞入寻常百姓家。",
    "暧暧远人村，依依墟里烟。",
    "种豆南山下，草盛豆苗稀。",
    
    # --- 春日生机 (Spring/Lively) ---
    "两只黄鹂鸣翠柳，一行白鹭上青天。",
    "竹外桃花三两枝，春江水暖鸭先知。",
    "几处早莺争暖树，谁家新燕啄春泥。",
    "草长莺飞二月天，拂堤杨柳醉春烟。",
    "乱花渐欲迷人眼，浅草才能没马蹄。",
    "春色满园关不住，一枝红杏出墙来。",
    "小荷才露尖尖角，早有蜻蜓立上头。",
    
    # --- 秋意与夕阳 (Autumn/Sunset) ---
    "一道残阳铺水中，半江瑟瑟半江红。",
    "停车坐爱枫林晚，霜叶红于二月花。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    "月落乌啼霜满天，江枫渔火对愁眠。",
    "青山横北郭，白水绕东城。",
    
    # --- 边塞与大漠 (Frontier/Desert) ---
    "大漠孤烟直，长河落日圆。",
    "北风卷地白草折，胡天八月即飞雪。",
    "黄沙百战穿金甲，不破楼兰终不还。",
    "秦时明月汉时关，万里长征人未还。",
    "大漠沙如雪，燕山月似钩。",
    
    # --- 楼台与烟雨 (Architecture/Rain) ---
    "南朝四百八十寺，多少楼台烟雨中。",
    "清明时节雨纷纷，路上行人欲断魂。",
    "天街小雨润如酥，草色遥看近却无。",
    "水光潋滟晴方好，山色空蒙雨亦奇。",
    "危楼高百尺，手可摘星辰。",
    "欲穷千里目，更上一层楼。",
    "黄鹤楼中吹玉笛，江城五月落梅花。",
    
    # --- 静谧与禅意 (Zen/Quiet) ---
    "人闲桂花落，夜静春山空。",
    "深林人不知，明月来相照。",
    "明月松间照，清泉石上流。",
    "野旷天低树，江清月近人。",
    "众鸟高飞尽，孤云独去闲。",
    "采菊东篱下，悠然见南山。",
    
    # --- 离别与愁绪 (Parting) ---
    "劝君更尽一杯酒，西出阳关无故人。",
    "孤帆远影碧空尽，唯见长江天际流。",
    "春风又绿江南岸，明月何时照我还。",
    "姑苏城外寒山寺，夜半钟声到客船。"
]

# ==========================================
# 2. 辅助函数
# ==========================================

def calculate_total_iou(boxes_tensor):
    if boxes_tensor.size(0) < 2: return 0.0
    x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
    x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
    y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
    y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
    
    n = boxes_tensor.size(0)
    total_inter = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xx1 = max(x1[i], x1[j]); yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j]); yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            total_inter += w * h
    return total_inter

def apply_random_symmetry(layout, device='cpu', attempt_prob=0.5):
    if not layout: return layout
    boxes_data = [list(item[1:5]) for item in layout] 
    boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32).to(device)
    initial_iou = calculate_total_iou(boxes_tensor)
    
    new_layout = copy.deepcopy(layout)
    current_boxes = boxes_tensor.clone()
    
    indices = list(range(len(layout)))
    random.shuffle(indices)
    
    for idx in indices:
        if random.random() > attempt_prob: continue
        original_item = new_layout[idx]
        original_box = current_boxes[idx].clone()
        
        new_cx = 1.0 - original_item[1]
        item_list = list(original_item)
        item_list[1] = new_cx
        
        if len(item_list) >= 9:
            item_list[5] = -item_list[5] # bias_x
            item_list[7] = -item_list[7] # rotation
        
        current_boxes[idx, 0] = new_cx
        new_iou = calculate_total_iou(current_boxes)
        
        if new_iou <= initial_iou + 1e-4: 
            new_layout[idx] = tuple(item_list)
            initial_iou = new_iou 
        else:
            current_boxes[idx] = original_box 
            
    return new_layout

def sanitize_filename(text):
    """提取汉字作为文件名"""
    safe_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    return safe_text[:10] if safe_text else "untitled_poem"

# ==========================================
# 3. 模型管线类
# ==========================================

class ShanshuiPipeline:
    def __init__(self, args):
        self.device = args.device
        self.args = args
        
        print("\n🚀 初始化全流程生成管线...")
        self.layout_model, self.tokenizer = self._load_layout_model()
        self.sd_pipe = self._load_sd_pipeline()
        self.mask_generator = InkWashMaskGenerator(width=args.width, height=args.height)
        
    def _load_layout_model(self):
        print(f"   [Stage 1] 加载布局模型: {self.args.layout_config}")
        with open(self.args.layout_config, "r") as f:
            config = yaml.safe_load(f)
        model_config = config['model']
        
        tokenizer = BertTokenizer.from_pretrained(model_config['bert_path'])
        
        model = Poem2LayoutGenerator(
            bert_path=model_config['bert_path'],
            num_classes=model_config['num_classes'],
            hidden_size=model_config['hidden_size'],
            bb_size=model_config['bb_size'],
            decoder_layers=model_config['decoder_layers'],
            decoder_heads=model_config['decoder_heads'],
            dropout=model_config['dropout'],
            latent_dim=model_config.get('latent_dim', 32)
        )
        
        checkpoint = torch.load(self.args.layout_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def _load_sd_pipeline(self):
        print(f"   [Stage 2] 加载 SD + ControlNet + LoRA...")
        unet = UNet2DConditionModel.from_pretrained(
            self.args.base_sd_path, subfolder="unet", torch_dtype=torch.float16
        )
        
        lora_path = os.path.join(self.args.sd_checkpoint_dir, "unet_lora")
        try:
            unet = PeftModel.from_pretrained(unet, lora_path)
            unet = unet.merge_and_unload()
            print("   ✅ LoRA 融合成功")
        except Exception as e:
            print(f"   ❌ LoRA 挂载失败: {e}")
            sys.exit(1)
            
        controlnet_path = os.path.join(self.args.sd_checkpoint_dir, "controlnet_structure")
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.args.base_sd_path,
            unet=unet,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device == 'cuda':
            pipe.enable_model_cpu_offload()
        return pipe

    def decode_latents_to_image(self, latents):
        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        latents = 1 / scaling_factor * latents
        with torch.no_grad():
            image = self.sd_pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image[0])

    def generate(self, poem_text, seed=None, save_intermediates_dir=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        print(f"      正在推理: 【{poem_text[:15]}...】")
        
        # 1. Layout
        layout = greedy_decode_poem_layout(
            model=self.layout_model, 
            tokenizer=self.tokenizer, 
            poem=poem_text,
            max_elements=self.args.max_elements, 
            device=self.device
        )
        
        if not layout:
            print("      ⚠️ 无有效意象，跳过。")
            return None, None
            
        layout = apply_random_symmetry(layout, device=self.device, attempt_prob=0.6)

        # 2. Mask
        layout_list = [list(item) for item in layout]
        control_mask = self.mask_generator.convert_boxes_to_mask(layout_list)

        # 3. Diffusion
        # [Modified] 增强负向提示词，防止贴纸感
        n_prompt = (
            "sticker, cutout, collage, applique, harsh outlines, cartoon, flat color, "
            "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"
        )
        
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            if save_intermediates_dir and (step % 5 == 0 or step == self.args.steps - 1):
                image = self.decode_latents_to_image(latents)
                step_str = str(step).zfill(3)
                save_path = os.path.join(save_intermediates_dir, f"step_{step_str}.png")
                image.save(save_path)

        callback = callback_fn if save_intermediates_dir else None
        callback_steps = 1 if save_intermediates_dir else 0

        # [Modified] 加入 control_guidance_end=0.6 早停策略
        image = self.sd_pipe(
            prompt=poem_text,
            image=control_mask,
            negative_prompt=n_prompt,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.guidance,
            controlnet_conditioning_scale=self.args.control_scale,
            control_guidance_end=0.6, # <--- 关键修改：前60%步数控制构图，后40%自由晕染
            width=self.args.width,
            height=self.args.height,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps
        ).images[0]
        
        return image, control_mask

# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Poem2Painting Batch 50 Inference")
    
    # 路径参数
    parser.add_argument('--layout_checkpoint', type=str, required=True, help="Stage 1 .pth")
    parser.add_argument('--sd_checkpoint_dir', type=str, required=True, help="Stage 2 Dir")
    parser.add_argument('--base_sd_path', type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument('--layout_config', type=str, default="configs/default.yaml")
    
    # 生成参数
    parser.add_argument('--output_dir', type=str, default="outputs/batch_50_test", help="结果保存目录")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--control_scale', type=float, default=1.0)
    parser.add_argument('--max_elements', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=None)
    
    # 开关
    parser.add_argument('--save_intermediates', action='store_true', help="保存中间过程")
    parser.add_argument('--single_poem', type=str, default=None, help="如果设置，只跑这一句")
    
    args = parser.parse_args()
    
    # 初始化
    pipeline = ShanshuiPipeline(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要跑的列表
    if args.single_poem:
        tasks = [args.single_poem]
        print(f"\n🎯 单句测试模式: {args.single_poem}")
    else:
        tasks = BATCH_POEMS
        print(f"\n📚 批量测试模式: 共 {len(tasks)} 首诗")

    print(f"📂 结果输出目录: {args.output_dir}\n")
    print("="*60)

    # 循环执行
    success_count = 0
    for i, poem in enumerate(tasks):
        safe_name = sanitize_filename(poem)
        prefix = f"{str(i+1).zfill(2)}_{safe_name}"
        
        print(f"[{i+1}/{len(tasks)}] 处理: {prefix}")
        
        # 中间过程目录
        intermediates_dir = None
        if args.save_intermediates:
            intermediates_dir = os.path.join(args.output_dir, f"{prefix}_steps")
            os.makedirs(intermediates_dir, exist_ok=True)

        try:
            final_img, mask_img = pipeline.generate(
                poem, 
                seed=args.seed,
                save_intermediates_dir=intermediates_dir
            )
            
            if final_img:
                save_path_img = os.path.join(args.output_dir, f"{prefix}_paint.png")
                save_path_mask = os.path.join(args.output_dir, f"{prefix}_mask.png")
                
                final_img.save(save_path_img)
                mask_img.save(save_path_mask)
                success_count += 1
                print(f"   ✅ 完成")
            else:
                print(f"   ⚠️ 跳过 (空布局)")
                
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)

    print(f"\n🎉 全部完成! 成功: {success_count}/{len(tasks)}")
    print(f"查看结果: {args.output_dir}")

if __name__ == "__main__":
    main()