import os
import sys
import torch
import yaml
import argparse
import random
import copy
import re
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # 使用 PIL 替代 Matplotlib 以获得更好的像素级控制
from transformers import BertTokenizer

# ==========================================
# 1. 智能路径加载 & 基础设置
# ==========================================
current_script_path = os.path.abspath(__file__)
current_dir_name = os.path.basename(os.path.dirname(current_script_path))

if current_dir_name == 'scripts':
    project_root = os.path.dirname(os.path.dirname(current_script_path))
else:
    project_root = os.path.dirname(current_script_path)

sys.path.insert(0, project_root)
print(f"[Info] Project Root detected: {project_root}")

try:
    from models.poem2layout import Poem2LayoutGenerator
    # 确保 data.visualize 能被 greedy_decode 内部调用 (用于生成热力图)
    import data.visualize 
    from inference.greedy_decode import greedy_decode_poem_layout
except ImportError as e:
    print(f"\n[Critical Error] 无法导入项目模块: {e}")
    print("请确保在项目根目录或 scripts/ 目录下运行，并且 data/ 和 inference/ 文件夹完整。")
    sys.exit(1)

# ==========================================
# [配置] 类别与颜色定义
# ==========================================
CLASS_COLORS = {
    2: "red",      # mountain
    3: "blue",     # water
    4: "cyan",     # people
    5: "green",    # tree
    6: "yellow",   # building
    7: "magenta",  # bridge
    8: "purple",   # flower
    9: "orange",   # bird
    10: "brown"    # animal
}

CLASS_NAMES_EN = {
    2: "Mountain", 3: "Water", 4: "People", 5: "Tree",
    6: "Building", 7: "Bridge", 8: "Flower", 9: "Bird", 10: "Animal"
}

# ==========================================
# 2. 核心功能函数
# ==========================================

def calculate_total_iou(boxes_tensor):
    """计算所有框的总重叠面积"""
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
    """尝试水平翻转并进行碰撞检测"""
    if not layout: return layout
    boxes_data = [list(item[1:5]) for item in layout] 
    boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32).to(device)
    initial_iou = calculate_total_iou(boxes_tensor)
    
    indices = list(range(len(layout)))
    random.shuffle(indices)
    new_layout = copy.deepcopy(layout)
    current_boxes = boxes_tensor.clone()
    
    for idx in indices:
        if random.random() > attempt_prob: continue
        original_item = new_layout[idx]
        original_box = current_boxes[idx].clone()
        
        # 翻转逻辑: cx' = 1 - cx
        new_cx = 1.0 - original_item[1]
        
        # 翻转势态: Dir_X (Rotation Bias) 取反
        gestalt_params = list(original_item[5:])
        # 假设 9维格式: [cls, cx, cy, w, h, bx, by, rot, flow]
        # bx (bias_x) 需要翻转, rot (旋转角度) 也通常需要水平镜像
        if len(gestalt_params) >= 1: gestalt_params[0] = -gestalt_params[0] # bias_x
        if len(gestalt_params) >= 3: gestalt_params[2] = -gestalt_params[2] # rotation
        
        current_boxes[idx, 0] = new_cx
        new_iou = calculate_total_iou(current_boxes)
        
        if new_iou <= initial_iou + 1e-4: # 允许翻转
            item_list = list(original_item)
            item_list[1] = new_cx
            if len(item_list) >= 9: item_list[5:] = gestalt_params
            new_layout[idx] = tuple(item_list)
            initial_iou = new_iou 
        else: # 撤销
            current_boxes[idx] = original_box
    return new_layout

def sanitize_filename(text):
    """将诗句转换为合法的文件名 (保留部分字符)"""
    # 移除非法字符，保留汉字和字母数字
    safe_text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return safe_text[:15] if safe_text else "poem"

# ==========================================
# [关键] 黑色背景布局绘制 (基于 V8.0 visualize.py)
# ==========================================
def draw_layout_dark(layout, save_path, poem_text, img_size=(512, 512)):
    """
    绘制布局：黑色背景，无中文，包含势态可视化。
    Layout Item: (cls, cx, cy, w, h, bx, by, rot, flow)
    """
    try:
        # 1. 创建黑色背景
        img = Image.new('RGB', img_size, (20, 20, 20)) 
        draw = ImageDraw.Draw(img)
        W, H = img_size
        
        # 2. 字体加载 (使用默认字体避免中文乱码)
        try:
            # 尝试加载更清晰的英文字体，如果没有则使用默认
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        for item in layout:
            if len(item) < 5: continue
            
            cls_id = int(item[0])
            cx, cy, w, h = item[1], item[2], item[3], item[4]
            
            # 坐标转换
            pixel_cx, pixel_cy = cx * W, cy * H
            pixel_w, pixel_h = w * W, h * H
            xmin, ymin = pixel_cx - pixel_w/2, pixel_cy - pixel_h/2
            xmax, ymax = pixel_cx + pixel_w/2, pixel_cy + pixel_h/2
            
            color = CLASS_COLORS.get(cls_id, "white")
            cls_name = CLASS_NAMES_EN.get(cls_id, "UNK")
            
            # A. 绘制边框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            
            # B. 绘制势态 (Gestalt)
            info_text = ""
            if len(item) >= 9:
                # bx, by: Visual Center bias relative to half-size
                # rot: Rotation/Direction value
                # flow: Flow intensity
                bx, by, rot, flow = item[5], item[6], item[7], item[8]
                
                # --- 1. 视觉重心 (Visual Center) ---
                vc_x = pixel_cx + bx * (pixel_w / 2)
                vc_y = pixel_cy + by * (pixel_h / 2)
                r_dot = 3
                draw.ellipse([vc_x - r_dot, vc_y - r_dot, vc_x + r_dot, vc_y + r_dot], fill=color)
                
                # --- 2. 旋转方向 (Rotation/Direction) ---
                # 假设 rot 映射到角度，简单绘制一条指示线
                # 这里的 rot 可能是 normalized [-1, 1] 对应 [-90, 90] 度
                angle = rot * (math.pi / 2)
                line_len = min(pixel_w, pixel_h) * 0.4
                end_x = pixel_cx + math.sin(angle) * line_len
                end_y = pixel_cy - math.cos(angle) * line_len # y向下为正，减去cos模拟向上/下
                draw.line([(pixel_cx, pixel_cy), (end_x, end_y)], fill="white", width=1)
                
                info_text = f"|F:{flow:.1f}"

            # C. 绘制标签 (英文)
            label = f"{cls_name}{info_text}"
            
            # 确保文字不出界
            text_x = max(0, min(W - 80, xmin))
            text_y = max(0, ymin - 15 if ymin > 20 else ymax + 5)
            
            # 绘制文字背景 (可选，增加对比度)
            # draw.rectangle([text_x, text_y, text_x + len(label)*7, text_y + 12], fill=(0,0,0))
            draw.text((text_x, text_y), label, fill=color, font=font)

        # D. 底部绘制 Poem ID (用英文，避免乱码)
        draw.text((10, H - 20), f"Poem Hash: {hash(poem_text) % 10000}", fill="gray", font=font)
        
        img.save(save_path)
        
    except Exception as e:
        print(f"[Draw Error] {e}")

# ==========================================
# 3. 主程序
# ==========================================

TEST_POEMS = [
    "千山鸟飞绝，万径人踪灭。", "白日依山尽，黄河入海流。", "两岸猿声啼不住，轻舟已过万重山。",
    "远上寒山石径斜，白云生处有人家。", "水光潋滟晴方好，山色空蒙雨亦奇。", "孤帆远影碧空尽，唯见长江天际流。",
    "日照香炉生紫烟，遥看瀑布挂前川。", "青山横北郭，白水绕东城。", "明月松间照，清泉石上流。",
    "一道残阳铺水中，半江瑟瑟半江红。",
    "南朝四百八十寺，多少楼台烟雨中。", "小楼昨夜又东风，故国不堪回首月明中。", "危楼高百尺，手可摘星辰。",
    "欲穷千里目，更上一层楼。", "黄鹤楼中吹玉笛，江城五月落梅花。", "旧时王谢堂前燕，飞入寻常百姓家。",
    "折戟沉沙铁未销，自将磨洗认前朝。", "城阙辅三秦，风烟望五津。", "朱雀桥边野草花，乌衣巷口夕阳斜。",
    "深林人不知，明月来相照。",
    "离离原上草，一岁一枯荣。", "采菊东篱下，悠然见南山。", "墙角数枝梅，凌寒独自开。",
    "乱花渐欲迷人眼，浅草才能没马蹄。", "接天莲叶无穷碧，映日荷花别样红。", "竹外桃花三两枝，春江水暖鸭先知。",
    "人间四月芳菲尽，山寺桃花始盛开。", "小荷才露尖尖角，早有蜻蜓立上头。", "停车坐爱枫林晚，霜叶红于二月花。",
    "种豆南山下，草盛豆苗稀。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。", "柴门闻犬吠，风雪夜归人。", "路人借问遥招手，怕得鱼惊不应人。",
    "泥融飞燕子，沙暖睡鸳鸯。", "几处早莺争暖树，谁家新燕啄春泥。", "细雨鱼儿出，微风燕子斜。",
    "草长莺飞二月天，拂堤杨柳醉春烟。", "江上往来人，但爱鲈鱼美。", "独怜幽草涧边生，上有黄鹂深树鸣。",
    "枯藤老树昏鸦，小桥流水人家。",
    "北风卷地白草折，胡天八月即飞雪。", "天街小雨润如酥，草色遥看近却无。", "清明时节雨纷纷，路上行人欲断魂。",
    "月落乌啼霜满天，江枫渔火对愁眠。", "春风又绿江南岸，明月何时照我还。", "随风潜入夜，润物细无声。",
    "大漠孤烟直，长河落日圆。", "野旷天低树，江清月近人。", "忽如一夜春风来，千树万树梨花开。",
    "窗含西岭千秋雪，门泊东吴万里船。"
]

def load_model(config_path, checkpoint_path, device):
    print(f"Loading config from {config_path}...")
    if not os.path.exists(config_path):
        config_path = os.path.join(project_root, config_path)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    
    print(f"Initializing Tokenizer from {model_config['bert_path']}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_config['bert_path'])
    except Exception:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    print("Initializing Poem2LayoutGenerator...")
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

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default="configs/default.yaml")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_elements', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default="outputs/layout", help="Folder to save results")
    parser.add_argument('--no_symmetry', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)

    # 准备目录：
    # 1. Layout Text
    txt_dir = os.path.join(args.output_dir, "txt")
    # 2. Layout Image (Black BG)
    img_dir = os.path.join(args.output_dir, "img_black_bg")
    # 3. Heatmaps (Note: greedy_decode saves to 'outputs/heatmaps' by default, 
    # we can leave it there or inform user)
    
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.exists(args.config):
        possible_path = os.path.join(project_root, args.config)
        if os.path.exists(possible_path): args.config = possible_path

    model, tokenizer = load_model(args.config, args.checkpoint, args.device)
    
    print(f"\n>>> Starting Inference on {len(TEST_POEMS)} Poems <<<")
    print(f"    Layout Images (Black BG) -> {img_dir}")
    print(f"    Layout Text Data         -> {txt_dir}")
    print(f"    Attention Heatmaps       -> outputs/heatmaps/ (Auto-generated by greedy_decode)\n")
    
    for i, poem in enumerate(TEST_POEMS):
        safe_name = sanitize_filename(poem)
        print(f"[{i+1}/{len(TEST_POEMS)}] {poem[:10]}... -> {safe_name}")
        
        try:
            # 推理 (Heatmaps are generated internally here if visualize.py is present)
            layout = greedy_decode_poem_layout(
                model=model, tokenizer=tokenizer, poem=poem,
                max_elements=args.max_elements, device=args.device
            )
            
            if not args.no_symmetry and len(layout) > 0:
                layout = apply_random_symmetry(layout, device=args.device, attempt_prob=0.6)

            # 1. Save TXT (9-Dim Format)
            txt_path = os.path.join(txt_dir, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Poem: {poem}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Class':<6} | {'Name':<10} | {'Box (cx, cy, w, h)':<35} | {'Gestalt (bx, by, rot, flow)':<35}\n")
                f.write("-" * 100 + "\n")
                if not layout: f.write("  (No elements)\n")
                for item in layout:
                    cls_id = int(item[0])
                    box = item[1:5]
                    gestalt = item[5:]
                    
                    cls_name = CLASS_NAMES_EN.get(cls_id, "UNK")
                    box_str = f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f})"
                    
                    if len(gestalt) >= 4:
                        gest_str = f"({gestalt[0]:.3f}, {gestalt[1]:.3f}, {gestalt[2]:.3f}, {gestalt[3]:.3f})"
                    else:
                        gest_str = str(gestalt)
                        
                    f.write(f"{cls_id:<6} | {cls_name:<10} | {box_str:<35} | {gest_str:<35}\n")
            
            # 2. Save Layout Image (Black BG, English Labels)
            if layout:
                img_path = os.path.join(img_dir, f"{safe_name}.png")
                draw_layout_dark(layout, img_path, poem)

        except Exception as e:
            print(f"[Error] {poem}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll Completed!")

if __name__ == "__main__":
    main()