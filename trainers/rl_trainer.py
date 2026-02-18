# File: trainers/rl_trainer.py (V12.0: Gestalt Physics Integration)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from trainers.loss import compute_kl_loss

# --- [V10.0 保留] 物理常识锚点 ---
CLASS_SIZE_PRIORS = {
    0: 0.30, 1: 0.30, 2: 0.35, 3: 0.30, 6: 0.20, 5: 0.15,
    7: 0.10, 4: 0.08, 10: 0.08, 8: 0.04, 9: 0.02
}

# === [V12.0 NEW] 势态物理先验 ===
# 格式: class_id: {param_name: (target_value, tolerance)}
GESTALT_PRIORS = {
    2: {  # Mountain (山) - 静止、枯笔、水平
        'flow': (-0.8, 0.2),      # 枯笔 [-1.0 to -0.6]
        'rotation': (0.0, 0.3),   # 水平 [-0.3 to 0.3]
    },
    3: {  # Water (水) - 流动、湿笔、允许偏移
        'flow': (0.6, 0.3),       # 湿笔 [0.3 to 0.9]
        'bias_x': (0.0, 0.5),     # 流向偏移 [-0.5 to 0.5]
    },
    4: {  # People (人) - 中性笔触、站立感
        'flow': (0.0, 0.4),       # 中性 [-0.4 to 0.4]
        'bias_y': (-0.2, 0.3),    # 重心略上 [-0.5 to 0.1]
    },
    5: {  # Tree (树) - 略枯、倾斜、重心略上
        'flow': (-0.3, 0.3),      # 略枯 [-0.6 to 0.0]
        'rotation': (0.2, 0.4),   # 倾斜 [-0.2 to 0.6]
        'bias_y': (-0.1, 0.3),    # 树冠 [-0.4 to 0.2]
    },
    6: {  # Building (建筑) - 静止、垂直
        'flow': (-0.5, 0.3),      # 枯笔 [-0.8 to -0.2]
        'rotation': (0.0, 0.2),   # 垂直 [-0.2 to 0.2]
    },
    7: {  # Bridge (桥) - 中性、水平跨度
        'flow': (0.1, 0.4),       # 略湿 [-0.3 to 0.5]
        'rotation': (0.0, 0.3),   # 水平 [-0.3 to 0.3]
        'bias_x': (0.0, 0.6),     # 横跨
    },
    8: {  # Flower (花) - 湿润、细腻
        'flow': (0.5, 0.3),       # 湿笔 [0.2 to 0.8]
    },
    9: {  # Bird (鸟) - 极湿、灵动、飞翔
        'flow': (0.8, 0.2),       # 极湿 [0.6 to 1.0]
        'bias_y': (-0.5, 0.3),    # 飞翔 [-0.8 to -0.2]
    },
    10: {  # Animal (动物) - 中性、重心低
        'flow': (0.0, 0.4),       # 中性 [-0.4 to 0.4]
        'bias_y': (0.2, 0.3),     # 重心低 [-0.1 to 0.5]
    },
}

class RLTrainer(LayoutTrainer):
    """
    [V12.0 Gestalt Enhanced] 完整集成势态物理约束
    
    新增功能:
    1. [Gestalt Physics] 类别-势态匹配奖励 (water必须flow>0.5, mountain必须flow<-0.5)
    2. [Gestalt Smoothness] 空间连续性奖励 (相邻物体态势平滑过渡)
    3. [Gestalt Extreme] 极值惩罚 (防止参数爆炸到±1.0边界)
    4. [Gestalt Supervision] 辅助监督损失 (防止RL破坏监督学习的势态知识)
    
    保留功能:
    - [V11.6] Tensor维度修复 + 重力沉底策略
    - [V10.0] 物理大小先验 + 空间关系约束
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        self.rl_lr = float(config['training'].get('rl_learning_rate', 1e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        self.success_scale = 2.0
        self.failure_scale = 0.1
        
        # --- 布局权重策略 ---
        self.w_iou = config['training']['reward_weights'].get('iou', 2.0)
        self.w_rel = config['training']['reward_weights'].get('relation', 4.0)
        self.w_physics = config['training']['reward_weights'].get('physics', 6.0)
        self.w_align = config['training']['reward_weights'].get('alignment', 4.0)
        self.w_hm_size = config['training']['reward_weights'].get('heatmap_size', 2.5)
        
        # 斥力与惩罚
        self.w_disp = config['training']['reward_weights'].get('dispersion', 7.0)
        self.w_overlap = config['training']['reward_weights'].get('overlap', -15.0)
        self.w_bound = config['training']['reward_weights'].get('boundary', -3.0)
        
        # [关键策略] 重力沉底，留白在顶
        self.w_balance = config['training']['reward_weights'].get('balance', 5.0)
        self.w_center = config['training']['reward_weights'].get('center', 1.5)

        # === [V12.0 NEW] 势态权重策略 ===
        self.w_gestalt_physics = config['training'].get('w_gestalt_physics', 8.0)
        self.w_gestalt_smooth = config['training'].get('w_gestalt_smooth', 3.0)
        self.w_gestalt_extreme = config['training'].get('w_gestalt_extreme', 5.0)
        self.w_gestalt_sup = config['training'].get('w_gestalt_sup', 0.3)

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V12.0] Gestalt Physics System Initialized.")
        print(f" -> Layout: Gravity Y=0.75 (Bottom) | Weight: {self.w_balance}")
        print(f" -> Gestalt: Physics Matching | Weight: {self.w_gestalt_physics}")
        print(f" -> Gestalt: Smoothness | Weight: {self.w_gestalt_smooth}")
        print(f" -> Gestalt: Extreme Penalty | Weight: {self.w_gestalt_extreme}")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        [V12.0 增强版] 计算奖励: 布局 (4D) + 势态 (4D)
        """
        B, T, full_dim = dynamic_layout.shape
        device = dynamic_layout.device
        
        # === 解包布局和势态 ===
        pred_boxes = dynamic_layout[..., :4]       # [B, T, 4]: [cx, cy, w, h]
        pred_gestalt = dynamic_layout[..., 4:8]    # [B, T, 4]: [bias_x, bias_y, rotation, flow]
        
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes'][..., :4] 
        kg_spatial_matrix = batch.get('kg_spatial_matrix')
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # ========== 布局奖励 (保持不变) ==========
        
        # 1. 物理大小
        r_physics = self._calculate_physics_size_reward(pred_boxes, kg_class_ids) * self.w_physics

        # 2. IoU
        r_iou = self._calculate_iou(pred_boxes, target_boxes) * loss_mask * self.w_iou

        # 3. 空间关系
        r_rel = torch.zeros(B, T, device=device)
        if kg_spatial_matrix is not None:
            r_rel = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) * self.w_rel

        # 4. 热力图对齐
        r_align = torch.zeros(B, T, device=device)
        r_hm_size = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            hi_res = F.interpolate(attention_maps, size=(256, 256), mode='bilinear', align_corners=False)
            potential_field = F.avg_pool2d(hi_res, kernel_size=5, stride=1, padding=2)
            r_align = self._calculate_attention_alignment(potential_field, pred_boxes) * self.w_align
            r_hm_size = self._calculate_heatmap_area_reward(potential_field, pred_boxes) * self.w_hm_size

        # 5. 分散与边界
        r_disp = self._calculate_dispersion_reward(pred_boxes, loss_mask) * self.w_disp
        r_bound = self._calculate_boundary_penalty(pred_boxes) * self.w_bound
        
        # 6. [重力系统]
        r_balance = self._calculate_vertical_balance_reward(pred_boxes, loss_mask) * self.w_balance
        r_center = self._calculate_horizontal_centering_reward(pred_boxes) * self.w_center

        # 7. 重叠惩罚
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        r_over = overlap_penalty * self.w_overlap 
        veto_factor = (1.0 - overlap_penalty * 5.0).clamp(min=0.0)
        
        # ========== [V12.0 NEW] 势态奖励 ==========
        
        # 8. 势态物理匹配 (核心!)
        r_gestalt_phy = self._calculate_gestalt_physics_reward(
            pred_gestalt, kg_class_ids, loss_mask
        ) * self.w_gestalt_physics
        
        # 9. 势态空间平滑
        r_gestalt_smooth = self._calculate_gestalt_smoothness_reward(
            pred_gestalt, pred_boxes, loss_mask
        ) * self.w_gestalt_smooth
        
        # 10. 势态极值惩罚
        r_gestalt_extreme = self._calculate_gestalt_extreme_penalty(
            pred_gestalt, loss_mask
        ) * self.w_gestalt_extreme
        
        # --- 汇总 (基础奖励 + 势态奖励) ---
        obj_rewards += (r_physics + r_align + r_balance + r_center + r_gestalt_phy) 
        obj_rewards += (r_iou + r_hm_size + r_rel + r_disp + r_gestalt_smooth) * veto_factor
        obj_rewards += r_bound + r_gestalt_extreme
        obj_rewards += r_over 

        self.last_reward_stats = {
            'Phy': r_physics.mean().item(),
            'Align': r_align.mean().item(),
            'Disp': r_disp.mean().item(),
            'Bal': r_balance.mean().item(), 
            'Over': r_over.mean().item(),
            'G_Phy': r_gestalt_phy.mean().item(),      # [NEW]
            'G_Smo': r_gestalt_smooth.mean().item(),   # [NEW]
            'G_Ext': r_gestalt_extreme.mean().item(),  # [NEW]
        }

        valid_count = loss_mask.sum(dim=1).clamp(min=1.0)
        return (obj_rewards * loss_mask).sum(dim=1) / valid_count

    def train_rl_epoch(self, epoch):
        self.model.train()
        total_reward = 0
        steps = 0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"RL Epoch {epoch}")):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            
            # Baseline
            self.model.eval()
            with torch.no_grad():
                b_out = self.model.forward_rl(batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                                             batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'), sample=False)
                reward_baseline = self.compute_reward(b_out[0], batch, attention_maps=b_out[2])
            
            # Sample
            self.model.train()
            s_out = self.model.forward_rl(batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                                         batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'), sample=True)
            reward_sample = self.compute_reward(s_out[0], batch, attention_maps=s_out[2])
            
            # Advantage
            raw_adv = reward_sample - reward_baseline
            std = raw_adv.std()
            norm_adv = (raw_adv - raw_adv.mean()) / (std + 1e-8) if std > 1e-6 else raw_adv
            final_adv = norm_adv * torch.where(norm_adv > 0, self.success_scale, self.failure_scale)
            
            # === [V11.6 FIX] 维度修复 ===
            seq_log_prob = s_out[1].sum(dim=1)  # [B, T] -> [B]
            rl_loss = -(seq_log_prob * final_adv).mean()
            
            # === [V12.0 NEW] 势态监督损失 (辅助) ===
            if step % 5 == 0:
                mu, logvar, dynamic_layout_sup, decoder_output, _, aux_outputs = self.model(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                    target_boxes=batch['target_boxes']
                )
                
                # 原有损失 (布局监督)
                loss_tuple = self.model.get_loss(
                    pred_cls=None, pred_bbox_ids=None, pred_boxes=dynamic_layout_sup, 
                    pred_count=None, layout_seq=None, layout_mask=batch['loss_mask'], 
                    num_boxes=batch['num_boxes'], target_coords_gt=batch['target_boxes'],
                    kg_spatial_matrix=batch.get('kg_spatial_matrix'), kg_class_weights=batch.get('kg_class_weights'),
                    kg_class_ids=batch['kg_class_ids'], decoder_output=decoder_output, 
                    gestalt_mask=batch.get('gestalt_mask'), aux_outputs=aux_outputs
                )
                
                # === [NEW] 势态监督损失 ===
                pred_gestalt_sup = dynamic_layout_sup[..., 4:8]  # [B, T, 4]
                target_gestalt = batch['target_boxes'][..., 4:8]  # [B, T, 4] from image
                gestalt_mask = batch.get('gestalt_mask')  # [B, T] validity
                
                if gestalt_mask is not None:
                    # 仅在有效区域计算势态损失
                    valid_mask = batch['loss_mask'] * gestalt_mask  # [B, T]
                    gestalt_loss = F.smooth_l1_loss(
                        pred_gestalt_sup, target_gestalt, reduction='none'
                    ) * valid_mask.unsqueeze(-1)  # [B, T, 4] * [B, T, 1]
                    
                    gestalt_loss = gestalt_loss.sum() / (valid_mask.sum().clamp(min=1.0) * 4.0)
                else:
                    gestalt_loss = 0.0
                
                # 组合损失
                total_combined_loss = (
                    rl_loss + 
                    0.2 * (loss_tuple[0] + (compute_kl_loss(mu, logvar) if mu is not None else 0.0)) +
                    self.w_gestalt_sup * gestalt_loss  # [NEW] 防止势态漂移
                )
            else:
                total_combined_loss = rl_loss

            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            if (step + 1) % 10 == 0:
                s = self.last_reward_stats
                print(f"[Step {step+1}] R:{reward_sample.mean().item():.2f} | "
                      f"Phy:{s.get('Phy',0):.2f} | Bal:{s.get('Bal',0):.2f} | Over:{s.get('Over',0):.2f} | "
                      f"G_Phy:{s.get('G_Phy',0):.2f} | G_Smo:{s.get('G_Smo',0):.2f}")

        # Plotting
        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward

    # ================= [V12.0 NEW] 势态奖励函数 =================
    
    def _calculate_gestalt_physics_reward(self, pred_gestalt, kg_class_ids, mask):
        """
        [核心功能] 类别-势态物理匹配奖励
        
        工作原理:
        1. 根据 GESTALT_PRIORS 定义的先验知识
        2. 检查每个物体的势态是否符合其类别的物理特性
        3. 例: water(cls=3) 的 flow 应该在 [0.3, 0.9]
        """
        B, T, _ = pred_gestalt.shape
        device = pred_gestalt.device
        rewards = torch.zeros(B, T, device=device)
        
        # 解包势态参数
        bias_x = pred_gestalt[..., 0]   # [-1, 1]
        bias_y = pred_gestalt[..., 1]
        rotation = pred_gestalt[..., 2]
        flow = pred_gestalt[..., 3]
        
        for b in range(B):
            for t in range(T):
                if mask[b, t] < 0.5:
                    continue
                
                cls_id = kg_class_ids[b, t].item()
                if cls_id not in GESTALT_PRIORS:
                    rewards[b, t] = 0.5  # 无约束类别给基础分
                    continue
                
                priors = GESTALT_PRIORS[cls_id]
                score = 0.0
                count = 0
                
                # 检查每个参数
                for param_name, (target, tolerance) in priors.items():
                    if param_name == 'flow':
                        value = flow[b, t].item()
                    elif param_name == 'rotation':
                        value = rotation[b, t].item()
                    elif param_name == 'bias_x':
                        value = bias_x[b, t].item()
                    elif param_name == 'bias_y':
                        value = bias_y[b, t].item()
                    else:
                        continue
                    
                    # 高斯奖励: 在 [target±tolerance] 内得高分
                    deviation = abs(value - target)
                    if deviation <= tolerance:
                        score += 1.0
                    else:
                        # 超出范围指数衰减
                        score += np.exp(-2.0 * ((deviation - tolerance) ** 2))
                    count += 1
                
                rewards[b, t] = score / count if count > 0 else 0.5
        
        return rewards
    
    def _calculate_gestalt_smoothness_reward(self, pred_gestalt, pred_boxes, mask):
        """
        空间平滑性奖励: 相邻物体的势态应该连续变化
        
        物理直觉:
        - 连续的山脉, rotation 应该逐渐变化
        - 河流的 flow 应该保持一致
        - 树林的 bias 应该有统一风向
        """
        B, T, _ = pred_gestalt.shape
        device = pred_gestalt.device
        rewards = torch.zeros(B, T, device=device)
        
        for b in range(B):
            valid_indices = torch.nonzero(mask[b] > 0.5).squeeze(1)
            if len(valid_indices) < 2:
                continue
            
            for i in range(len(valid_indices) - 1):
                idx_a = valid_indices[i]
                idx_b = valid_indices[i + 1]
                
                # 计算空间距离
                box_a = pred_boxes[b, idx_a]
                box_b = pred_boxes[b, idx_b]
                dist = torch.sqrt(
                    (box_a[0] - box_b[0])**2 + (box_a[1] - box_b[1])**2
                ).item()
                
                # 只对邻近物体(dist<0.3)计算平滑性
                if dist > 0.3:
                    continue
                
                # 势态差异
                gestalt_a = pred_gestalt[b, idx_a]
                gestalt_b = pred_gestalt[b, idx_b]
                gestalt_diff = torch.abs(gestalt_a - gestalt_b).mean().item()
                
                # 平滑奖励: 差异越小越好
                smooth_score = np.exp(-3.0 * gestalt_diff)
                rewards[b, idx_a] += smooth_score * 0.5
                rewards[b, idx_b] += smooth_score * 0.5
        
        return torch.clamp(rewards, 0.0, 1.0)
    
    def _calculate_gestalt_extreme_penalty(self, pred_gestalt, mask):
        """
        极值惩罚: 防止势态参数爆炸到边界
        
        合理范围:
        - bias_x/y: [-0.95, 0.95]
        - rotation: [-0.95, 0.95]
        - flow: [-1.0, 1.0]
        """
        # 计算超出合理范围的程度 (超过0.95就开始惩罚)
        extreme_violation = torch.clamp(
            torch.abs(pred_gestalt) - 0.95, min=0.0
        )
        
        # 指数惩罚
        penalty_per_dim = torch.exp(-5.0 * extreme_violation)
        
        # 平均所有维度
        penalty = penalty_per_dim.mean(dim=-1)
        
        return penalty * mask

    # ================= 布局辅助函数 (保持不变) =================
    
    def _calculate_relation_reward(self, pred_boxes, kg_spatial_matrix, kg_class_ids):
        """计算空间关系奖励 (Above/Below/Inside)"""
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        rewards = torch.zeros(B, T, device=device)
        
        for b in range(B):
            valid_indices = torch.nonzero(kg_class_ids[b] >= 2).squeeze(1)
            if len(valid_indices) < 2: continue

            for i in valid_indices:
                for j in valid_indices:
                    if i == j: continue
                    cid_i = kg_class_ids[b, i].item(); cid_j = kg_class_ids[b, j].item()
                    idx_i, idx_j = int(cid_i) - 2, int(cid_j) - 2
                    if not (0 <= idx_i < 9 and 0 <= idx_j < 9): continue
                    
                    rel = kg_spatial_matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue 
                    
                    box_a = pred_boxes[b, i]; box_b = pred_boxes[b, j]
                    current_reward = 0.0
                    
                    if rel in [1, 5]: # ABOVE: A.y < B.y
                        diff = box_a[1] - box_b[1]
                        if diff < 0: current_reward = 1.0
                        else: current_reward = torch.exp(-3.0 * diff)
                    elif rel == 2: # BELOW: A.y > B.y
                        diff = box_b[1] - box_a[1]
                        if diff < 0: current_reward = 1.0
                        else: current_reward = torch.exp(-3.0 * diff)
                    elif rel == 3: # INSIDE
                        a_x1, a_y1 = box_a[0]-box_a[2]/2, box_a[1]-box_a[3]/2
                        a_x2, a_y2 = box_a[0]+box_a[2]/2, box_a[1]+box_a[3]/2
                        b_x1, b_y1 = box_b[0]-box_b[2]/2, box_b[1]-box_b[3]/2
                        b_x2, b_y2 = box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2
                        violation = F.relu(b_x1 - a_x1) + F.relu(a_x2 - b_x2) + F.relu(b_y1 - a_y1) + F.relu(a_y2 - b_y2)
                        if violation < 1e-4: current_reward = 1.0
                        else: current_reward = torch.exp(-5.0 * violation)
                    
                    rewards[b, i] += current_reward
        return torch.clamp(rewards, max=1.0)

    def _calculate_vertical_balance_reward(self, pred_boxes, mask):
        """重力沉底 (Target Y=0.75)"""
        cy = pred_boxes[..., 1]
        weighted_sum_cy = (cy * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        mean_cy = weighted_sum_cy / count
        return torch.exp(-4.0 * (mean_cy.unsqueeze(1) - 0.75) ** 2)

    def _calculate_horizontal_centering_reward(self, pred_boxes):
        cx = pred_boxes[..., 0]
        dist_to_center = torch.abs(cx - 0.5)
        return torch.exp(-2.0 * dist_to_center ** 2)

    def _calculate_physics_size_reward(self, pred_boxes, kg_class_ids):
        B, T = pred_boxes.shape[:2]
        pred_area = (pred_boxes[..., 2] * pred_boxes[..., 3]).clamp(min=1e-6)
        target_priors = torch.full((B, T), 0.15, device=pred_boxes.device)
        for cid, prior_area in CLASS_SIZE_PRIORS.items():
            target_priors = torch.where(kg_class_ids == cid, torch.tensor(prior_area, device=pred_boxes.device), target_priors)
        return torch.exp(-1.0 * (torch.abs(torch.log(pred_area) - torch.log(target_priors)) ** 2))

    def _calculate_heatmap_area_reward(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        max_vals = attn_maps.view(B, T, -1).max(dim=-1)[0].view(B, T, 1, 1)
        active_pixels = (attn_maps > (max_vals * 0.4)).float().sum(dim=[-1, -2])
        heatmap_coverage = (active_pixels / (H * W)).clamp(min=0.01, max=0.45)
        pred_area = (boxes[..., 2] * boxes[..., 3]).clamp(min=1e-6)
        return torch.exp(-1.5 * torch.abs(torch.log(pred_area) - torch.log(heatmap_coverage)))

    def _calculate_attention_alignment(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        device = boxes.device
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing='ij')
        x1, y1 = (boxes[..., 0]-boxes[..., 2]/2).view(B,T,1,1), (boxes[..., 1]-boxes[..., 3]/2).view(B,T,1,1)
        x2, y2 = (boxes[..., 0]+boxes[..., 2]/2).view(B,T,1,1), (boxes[..., 1]+boxes[..., 3]/2).view(B,T,1,1)
        mask = (grid_x.view(1,1,H,W) >= x1) & (grid_x.view(1,1,H,W) <= x2) & (grid_y.view(1,1,H,W) >= y1) & (grid_y.view(1,1,H,W) <= y2)
        align = (attn_maps * mask.float()).sum(dim=[-1, -2]) / (attn_maps.sum(dim=[-1, -2]) + 1e-6)
        return torch.pow(align, 2)

    def _calculate_dispersion_reward(self, pred_boxes, mask):
        B, T, _ = pred_boxes.shape
        dist_mat = torch.cdist(pred_boxes[..., :2], pred_boxes[..., :2], p=2)
        m1, m2 = mask.unsqueeze(1).bool(), mask.unsqueeze(2).bool()
        v_mask = m1 & m2 & (~torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0))
        sum_dist = (dist_mat * v_mask.float()).sum(dim=2)
        return torch.clamp((sum_dist / v_mask.float().sum(dim=2).clamp(min=1.0)) * 4.0, max=1.0)

    def _calculate_boundary_penalty(self, pred_boxes):
        cx, cy = pred_boxes[..., 0], pred_boxes[..., 1]
        pen = F.relu(0.05 - cx) + F.relu(cx - 0.95) + F.relu(0.05 - cy) + F.relu(cy - 0.98)
        return torch.clamp(pen * 5.0, max=2.0)
    
    def _calculate_iou(self, pred, target):
        px1, py1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        px2, py2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        tx1, ty1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        tx2, ty2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        ix = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0) * (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
        return ix / (pred[..., 2]*pred[..., 3] + target[..., 2]*target[..., 3] - ix + 1e-6)

    def _calculate_overlap_penalty(self, pred_boxes):
        B, T, _ = pred_boxes.shape
        x1, y1 = pred_boxes[..., 0]-pred_boxes[..., 2]/2, pred_boxes[..., 1]-pred_boxes[..., 3]/2
        x2, y2 = pred_boxes[..., 0]+pred_boxes[..., 2]/2, pred_boxes[..., 1]+pred_boxes[..., 3]/2
        area = pred_boxes[..., 2] * pred_boxes[..., 3]
        ix = (torch.min(x2.unsqueeze(2), x2.unsqueeze(1)) - torch.max(x1.unsqueeze(2), x1.unsqueeze(1))).clamp(min=0)
        iy = (torch.min(y2.unsqueeze(2), y2.unsqueeze(1)) - torch.max(y1.unsqueeze(2), y1.unsqueeze(1))).clamp(min=0)
        iou = (ix * iy) / (area.unsqueeze(2) + area.unsqueeze(1) - ix * iy + 1e-6)
        iou.masked_fill_(torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0), 0.0)
        return F.relu(iou.max(dim=2)[0] - 0.05)

    def _plot_reward_history(self):
        if not self.reward_history: return
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history, marker='o', color='b', label='Avg Reward')
            plt.xlabel('Epochs')
            plt.ylabel('Reward Value')
            plt.title('RL Training Progress (V12.0: Gestalt Enhanced)')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.plot_path_reward)
            plt.close()
        except Exception: pass