import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from typing import List, Union
import math
import random
import cv2 

class InkWashMaskGenerator:
    """
    [V10.6 自适应控制版] 有机墨迹生成器
    
    核心特性:
    1. Clean Edge: 降低外部轮廓的随机毛刺 (Roughness 0.1)，防止画面杂乱。
    2. Hybrid Control: 保持线框强控制。
    3. Textured Core: 核心带有噪点纹理。
    4. Adaptive Alpha: [NEW] 利用 Flow 值动态调节核心透明度，实现对 ControlNet 强度的像素级自适应控制。
    """
    
    CLASS_COLORS = {
        2: (255, 0, 0),   # 山 (Red)
        3: (0, 0, 255),   # 水 (Blue)
        4: (0, 255, 255), # 人 (Cyan)
        5: (0, 255, 0),   # 树 (Green)
        6: (255, 255, 0), # 建筑 (Yellow)
        7: (255, 0, 255), # 桥 (Bridge)
        8: (128, 0, 128), # 花 (Purple)
        9: (255, 165, 0), # 鸟 (Orange)
        10: (165, 42, 42) # 兽 (Brown)
    }
    
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        
    def _rotate_point(self, px, py, cx, cy, angle):
        """绕中心点旋转坐标"""
        theta = angle * math.pi / 2.0 
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        nx = cos_t * (px - cx) - sin_t * (py - cy) + cx
        ny = sin_t * (px - cx) + cos_t * (py - cy) + cy
        return nx, ny

    def _distort_box(self, x, y, w, h, rot=0.0, roughness=0.2):
        """
        生成不规则多边形顶点，模拟手绘/墨迹边缘
        """
        points = []
        segments = 8 
        
        cx, cy = x + w/2, y + h/2
        tl, tr = (x, y), (x + w, y)
        br, bl = (x + w, y + h), (x, y + h)
        
        def get_line_points(start, end):
            res = []
            vx, vy = end[0] - start[0], end[1] - start[1]
            for i in range(segments):
                alpha = i / segments
                px = start[0] + vx * alpha
                py = start[1] + vy * alpha
                
                noise_scale = math.sin(alpha * math.pi) * roughness * max(w, h) * 0.5
                perp_x, perp_y = -vy, vx
                norm = math.sqrt(perp_x**2 + perp_y**2) + 1e-6
                perp_x /= norm
                perp_y /= norm
                
                noise = (random.random() - 0.5) * 2 * noise_scale
                px += perp_x * noise
                py += perp_y * noise
                
                if rot != 0:
                    px, py = self._rotate_point(px, py, cx, cy, rot)
                
                res.append((px, py))
            return res

        points.extend(get_line_points(tl, tr))
        points.extend(get_line_points(tr, br))
        points.extend(get_line_points(br, bl))
        points.extend(get_line_points(bl, tl))
        
        return points

    def _apply_texture(self, field: np.ndarray, flow: float) -> np.ndarray:
        """
        根据 flow 值给场添加水墨纹理
        """
        if field.max() < 0.05: return field
        noise = np.random.uniform(0, 1, field.shape).astype(np.float32)
        
        if flow < 0: 
            dryness = abs(flow)
            threshold = 0.1 + 0.3 * dryness
            texture = (noise > threshold).astype(np.float32)
            field = field * (0.4 + 0.6 * texture)
            field = np.power(field, 0.8) 
        else:
            wetness = flow
            k_size = int(7 * wetness) * 2 + 1
            if k_size > 1:
                blurred = cv2.GaussianBlur(field, (k_size, k_size), 0)
                field = 0.3 * field + 0.7 * blurred
            texture = 1.0 - 0.2 * wetness * noise
            field = field * texture
        return np.clip(field, 0, 1)

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        """
        [V10.6] 转换布局为控制掩码 (集成 Ink-Dynamics Adaptive Control)
        """
        # 1. 初始化画布
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        anchor_canvas = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        anchor_draw = ImageDraw.Draw(anchor_canvas)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        valid_boxes = []
        for b in boxes:
            if len(b) < 5: continue
            b[3] = max(b[3], 0.06) 
            b[4] = max(b[4], 0.06)
            valid_boxes.append(b)
        
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            class_id = int(box[0])
            if class_id not in self.CLASS_COLORS: continue
            
            # 解析参数
            cx, cy, w, h = box[1], box[2], box[3], box[4]
            rot = box[7] if len(box) >= 8 else 0.0
            flow = box[8] if len(box) >= 9 else 0.0
            
            pixel_x = (cx - w/2) * self.width
            pixel_y = (cy - h/2) * self.height
            pixel_w = w * self.width
            pixel_h = h * self.height
            
            # -----------------------------------------------------------
            # A. 生成外部轮廓 (Outer Outline)
            # -----------------------------------------------------------
            # [修改] 降低 roughness (0.2 -> 0.1)，让边缘更整洁，减少画面噪点
            poly_points = self._distort_box(pixel_x, pixel_y, pixel_w, pixel_h, rot, roughness=0.1)
            
            # -----------------------------------------------------------
            # B. 绘制软势态层 (Soft Flow Layer) - 负责墨韵质感
            # -----------------------------------------------------------
            temp_img = Image.new('L', (self.width, self.height), 0)
            ImageDraw.Draw(temp_img).polygon(poly_points, fill=255)
            mask_np = np.array(temp_img).astype(np.uint8)
            
            dist_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
            max_val = dist_map.max()
            if max_val > 0:
                field = dist_map / max_val
                field = self._apply_texture(field, flow)
                
                color = self.CLASS_COLORS[class_id]
                object_layer = np.zeros_like(full_canvas)
                for c in range(3):
                    object_layer[:, :, c] = field * (color[c] / 255.0)
                
                alpha = np.clip(field * 1.8, 0, 1)
                alpha = np.expand_dims(alpha, axis=2)
                full_canvas = full_canvas * (1 - alpha) + object_layer * alpha
            
            # -----------------------------------------------------------
            # C. 绘制硬边锚点 (Hybrid Anchor)
            # -----------------------------------------------------------
            line_color = self.CLASS_COLORS[class_id] + (255,) 
            
            # C.1 绘制外线框 (Wireframe) - 保持轮廓约束
            anchor_draw.line(poly_points + [poly_points[0]], fill=line_color, width=3)
            
            # C.2 [核心升级] 纹理呼吸核心 (Textured Breathing Core)
            # 目的：不再使用纯色块，而是使用"噪点纹理+动态透明度"，
            # 强迫 ControlNet 脑补物体内部的纹理细节（如山石皴法）。
            
            core_ratio = 0.6  
            core_w = pixel_w * core_ratio
            core_h = pixel_h * core_ratio
            
            core_start_x = pixel_x + (pixel_w - core_w) / 2
            core_start_y = pixel_y + (pixel_h - core_h) / 2
            
            # 生成核心多边形 (Gestalt Aware: 随动旋转)
            core_points = self._distort_box(
                core_start_x, core_start_y, core_w, core_h, 
                rot=rot, roughness=0.1
            )
            
            # ================= [INNOVATION] =================
            # 墨韵感知自适应控制 (Ink-Dynamics Adaptive Control)
            # 利用 Alpha 透明度调制 ControlNet 的特征强度
            # ================================================
            
            # 1. 计算基础透明度 (Dynamic Alpha)
            if flow > 0: 
                # [湿笔 Wet Mode] -> 大幅降低 Alpha (降低权重)
                # 这种区域不仅显得湿润，而且 ControlNet 信号变弱(变灰)，
                # 允许 Diffusion Model 产生更多随机的晕染细节。
                # flow=1.0 (极湿) -> alpha=50 (非常弱的控制，SD自由发挥)
                # flow=0.0 (中性) -> alpha=200
                base_alpha = 50 + int((1.0 - flow) * 150) 
            else: 
                # [枯笔 Dry Mode] -> 保持高 Alpha (强权重)
                # 这种区域是画面的“骨架”，需要 ControlNet 强力锁定结构。
                base_alpha = 255 
            
            # 2. 创建临时图层绘制核心形状
            core_layer = Image.new('RGBA', (self.width, self.height), (0,0,0,0))
            core_draw = ImageDraw.Draw(core_layer)
            core_draw.polygon(core_points, fill=(255, 255, 255, 255))
            
            # 3. 生成噪点纹理 (Noise Texture)
            noise_intensity = 30 
            noise_arr = np.random.randint(-noise_intensity, noise_intensity, (self.height, self.width), dtype=np.int16)
            
            # 4. 融合颜色与纹理
            # 获取核心区域掩码
            core_mask_np = np.array(core_layer)[:, :, 3] > 0
            
            r, g, b = self.CLASS_COLORS[class_id]
            full_noise_layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            
            # 叠加 Alpha + Noise
            final_alpha_map = base_alpha + noise_arr
            final_alpha_map = np.clip(final_alpha_map, 50, 255).astype(np.uint8)
            
            full_noise_layer[:, :, 0] = r
            full_noise_layer[:, :, 1] = g
            full_noise_layer[:, :, 2] = b
            full_noise_layer[:, :, 3] = final_alpha_map
            
            # 5. 贴回 anchor_canvas
            full_noise_img = Image.fromarray(full_noise_layer, mode='RGBA')
            anchor_canvas.paste(full_noise_img, mask=core_layer)

        # 3. 最终合成
        soft_mask_np = np.clip(full_canvas * 255, 0, 255).astype(np.uint8)
        soft_mask_img = Image.fromarray(soft_mask_np).convert("RGBA")
        
        final_img = Image.alpha_composite(soft_mask_img, anchor_canvas).convert("RGB")

        # 移除模糊，保持纹理清晰
        # final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return final_img