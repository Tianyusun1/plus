# trainers/loss.py (V10.0: Text-to-Gestalt Enhanced Loss Functions - FIXED)

import torch
import torch.nn.functional as F

# =============================================================================
# [NEW V10.0] 语义态势先验表
# =============================================================================
# 定义每个类别的态势统计特征，用于约束模型从文本学习语义
# 格式: cls_id: (flow_mean, flow_std, rotation_mean, rotation_std, bias_weight)
SEMANTIC_GESTALT_PRIORS = {
    # flow: 负值=枯笔干涩, 正值=湿润洇散
    # rotation: 笔画倾斜角度 (归一化到 [-1, 1])
    # bias_weight: 允许的最大重心偏移量
    
    2: (-0.3, 0.25, 0.0, 0.3, 0.5),   # 山(mountain): 干涩厚重, 可有斜向皴法, 重心偏移小
    3: (0.5, 0.30, 0.0, 0.6, 0.7),    # 水(water): 湿润流动, 方向多变, 重心偏移大
    4: (0.0, 0.25, 0.0, 0.5, 0.6),    # 人(people): 中性, 动态姿态
    5: (-0.1, 0.30, 0.0, 0.7, 0.5),   # 树(tree): 略干, 枝干倾斜, 重心可偏
    6: (-0.5, 0.20, 0.0, 0.2, 0.3),   # 建筑(building): 干涩工整, 垂直, 重心居中
    7: (-0.2, 0.25, 0.0, 0.4, 0.4),   # 桥(bridge): 略干, 有倾斜
    8: (0.3, 0.30, 0.0, 0.6, 0.6),    # 花(flower): 湿润柔美, 方向灵活
    9: (0.2, 0.30, 0.0, 0.8, 0.7),    # 鸟(bird): 灵动, 方向多变, 重心偏移大
    10: (0.0, 0.25, 0.0, 0.5, 0.5),   # 动物(animal): 中性
}

# =============================================================================
# 基础布局损失 (保持不变)
# =============================================================================
def layout_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask, coord_loss_weight=1.0):
    """
    计算布局生成的混合损失 (Supervised Reconstruction Loss)。
    适用于基础几何布局 (Class + 4D Box)。
    
    Args:
        pred_cls: [B, num_elements, num_classes] 预测的类别 logits
        pred_coord: [B, num_elements, 4] 预测的坐标 (cx, cy, w, h)
        target_layout_seq: [B, S] (S = 5 * num_elements) 真实布局序列
        target_layout_mask: [B, S] 真实布局掩码
        coord_loss_weight: 坐标损失权重
    Returns:
        total_loss, cls_loss, coord_loss
    """
    batch_size, seq_len = target_layout_seq.shape
    num_elements = seq_len // 5

    reshaped_target = target_layout_seq.view(batch_size, num_elements, 5)
    target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements]
    target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements, 4]

    reshaped_mask = target_layout_mask.view(batch_size, num_elements, 5)
    cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements]

    # Classification loss
    cls_loss = F.cross_entropy(pred_cls, target_cls_ids, reduction='none') # [B, num_elements]
    cls_loss = cls_loss * cls_mask.float()
    cls_loss = cls_loss.sum() / cls_mask.sum().clamp(min=1)

    # Coordinate loss
    coord_loss = F.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, num_elements, 4]
    coord_loss = coord_loss * cls_mask.unsqueeze(-1).float()
    coord_loss = coord_loss.sum() / (cls_mask.sum().clamp(min=1) * 4)

    total_loss = cls_loss + coord_loss_weight * coord_loss
    return total_loss, cls_loss, coord_loss

# =============================================================================
# [V10.0 重写] 态势损失: 三重约束 (FIXED 索引错误)
# =============================================================================
def compute_gestalt_loss(
    pred_gestalt, 
    target_gestalt, 
    loss_mask, 
    gestalt_mask=None, 
    kg_class_ids=None, 
    pred_coords=None
):
    """
    [V10.0 升级版] 计算视觉态势 (Gestalt) 的综合损失。
    
    核心改进:
    1. 像素级监督 (Pixel-Level Supervision): 当有有效GT时，作为基础约束
    2. 语义先验 (Semantic Prior): **强化**类别级约束，让模型学习"云飘逸、山厚重"
    3. 空间平滑 (Spatial Smoothness): 相邻物体态势一致性
    
    关键变化:
    - 语义先验从"硬规则"改为"软约束" (用Hinge Loss而不是完全禁止)
    - 增加细粒度的类别先验 (不再只区分static/dynamic，而是每类都有具体数值)
    - 降低空间平滑权重，避免过度抑制个性

    Args:
        pred_gestalt: [B, N, 4] 预测态势 (bias_x, bias_y, rotation, flow)
        target_gestalt: [B, N, 4] GT 态势 (从像素提取)
        loss_mask: [B, N] 基础布局掩码 (Padding Mask)
        gestalt_mask: [B, N] (Optional) 态势有效性掩码 (1=Valid, 0=Failed)
        kg_class_ids: [B, N] (Optional) 物体类别 ID，用于语义先验
        pred_coords: [B, N, 4] (Optional) 预测坐标，用于空间平滑
        
    Returns:
        total_gestalt_loss: scalar (总损失)
        loss_pixel: scalar (像素监督损失，用于日志)
        loss_semantic: scalar (语义先验损失，用于日志)
        loss_smoothness: scalar (空间平滑损失，用于日志)
    """
    device = pred_gestalt.device
    B, N, _ = pred_gestalt.shape
    
    # =========================================================================
    # 1. 像素级监督损失 (Pixel-Level Supervision)
    # =========================================================================
    # 只在有有效GT时计算 (gestalt_mask=1 表示该样本的态势提取成功)
    loss_pixel = torch.tensor(0.0, device=device)
    
    if gestalt_mask is not None:
        valid_mask = loss_mask * gestalt_mask  # [B, N]
        num_valid = valid_mask.sum().clamp(min=1)
        
        if num_valid > 0:
            # Smooth L1 Loss (对异常值更鲁棒)
            reg_loss = F.smooth_l1_loss(pred_gestalt, target_gestalt, reduction='none')  # [B, N, 4]
            loss_pixel = (reg_loss.mean(dim=-1) * valid_mask).sum() / num_valid
    else:
        # 如果没有gestalt_mask，退化为全部样本都用
        valid_mask = loss_mask
        num_valid = valid_mask.sum().clamp(min=1)
        reg_loss = F.smooth_l1_loss(pred_gestalt, target_gestalt, reduction='none')
        loss_pixel = (reg_loss.mean(dim=-1) * valid_mask).sum() / num_valid
    
    # =========================================================================
    # 2. 语义先验损失 (Semantic Prior) - [FIXED] 修正索引方式
    # =========================================================================
    # 目的: 强制模型学习 "云应该湿润、山应该干涩" 这种类别级特征
    # 即使训练集中某些样本的像素GT不准确，语义先验也能拉回正确方向
    
    loss_semantic = torch.tensor(0.0, device=device)
    
    if kg_class_ids is not None:
        # 初始化先验张量
        priors = torch.zeros((B, N, 5), device=device)
        has_prior = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        # [FIX] 展平后操作，避免高维索引问题
        kg_ids_flat = kg_class_ids.reshape(-1)      # [B*N]
        priors_flat = priors.reshape(-1, 5)         # [B*N, 5]
        has_prior_flat = has_prior.reshape(-1)      # [B*N]
        
        for cid, (f_mean, f_std, r_mean, r_std, b_weight) in SEMANTIC_GESTALT_PRIORS.items():
            mask = (kg_ids_flat == cid)  # [B*N]
            if mask.sum() > 0:
                priors_flat[mask, 0] = f_mean
                priors_flat[mask, 1] = f_std
                priors_flat[mask, 2] = r_mean
                priors_flat[mask, 3] = r_std
                priors_flat[mask, 4] = b_weight
                has_prior_flat[mask] = True
        
        # 恢复形状
        priors = priors_flat.reshape(B, N, 5)
        has_prior = has_prior_flat.reshape(B, N)
        
        # 只对有先验定义的类别计算语义损失
        semantic_mask = has_prior & (loss_mask > 0)  # [B, N]
        
        if semantic_mask.sum() > 0:
            # 提取预测值
            pred_bias = pred_gestalt[..., :2]  # [B, N, 2] (bias_x, bias_y)
            pred_rot = pred_gestalt[..., 2]    # [B, N]
            pred_flow = pred_gestalt[..., 3]   # [B, N]
            
            # -----------------------------------------------------------------
            # A. Flow约束: 应该在 [mean - std, mean + std] 软区间内
            # -----------------------------------------------------------------
            flow_mean = priors[..., 0]
            flow_std = priors[..., 1]
            flow_lower = flow_mean - flow_std
            flow_upper = flow_mean + flow_std
            
            # Hinge Loss: 只惩罚超出合理范围的预测
            flow_violation = torch.maximum(
                F.relu(flow_lower - pred_flow),  # 低于下界的惩罚
                F.relu(pred_flow - flow_upper)   # 高于上界的惩罚
            )
            loss_flow = (flow_violation * semantic_mask.float()).sum() / semantic_mask.sum().clamp(min=1)
            
            # -----------------------------------------------------------------
            # B. Rotation约束: 引导到合理区间
            # -----------------------------------------------------------------
            rot_mean = priors[..., 2]
            rot_std = priors[..., 3]
            
            # 软约束: 鼓励rotation接近均值，但允许std范围内的偏差
            rot_target = torch.clamp(pred_rot, rot_mean - rot_std, rot_mean + rot_std)
            loss_rot = F.smooth_l1_loss(pred_rot, rot_target, reduction='none')
            loss_rot = (loss_rot * semantic_mask.float()).sum() / semantic_mask.sum().clamp(min=1)
            
            # -----------------------------------------------------------------
            # C. Bias约束: 重心偏移不应过大 (类别相关)
            # -----------------------------------------------------------------
            bias_weight = priors[..., 4]  # [B, N]
            bias_magnitude = torch.sqrt((pred_bias ** 2).sum(dim=-1))  # [B, N]
            
            # 惩罚超过类别允许阈值的偏移
            # 例如: 建筑(bias_weight=0.3)不应有大偏移，鸟(bias_weight=0.7)可以偏移更多
            bias_penalty = F.relu(bias_magnitude - bias_weight)
            loss_bias = (bias_penalty * semantic_mask.float()).sum() / semantic_mask.sum().clamp(min=1)
            
            # 汇总语义损失
            loss_semantic = loss_flow + loss_rot + 0.5 * loss_bias
    
    # =========================================================================
    # 3. 空间平滑损失 (Spatial Smoothness)
    # =========================================================================
    # 目的: 相邻物体的态势应该相似 (例如同一片水域的多个"水"元素)
    # 注意: 权重降低到0.3，避免过度抑制合理的态势差异
    
    loss_smoothness = torch.tensor(0.0, device=device)
    
    if pred_coords is not None:
        centers = pred_coords[..., :2]  # [B, N, 2] (cx, cy)
        dist_mat = torch.cdist(centers, centers, p=2)  # [B, N, N] 欧式距离
        
        # 定义邻居: 距离 < 0.25 (调大阈值，因为水墨画物体可能较远)
        is_neighbor = (dist_mat < 0.25) & (dist_mat > 1e-4)
        
        # 确保只计算有效物体对
        valid_objs = loss_mask.unsqueeze(1) * loss_mask.unsqueeze(2)  # [B, N, N]
        neighbor_mask = is_neighbor & (valid_objs > 0)
        
        if neighbor_mask.sum() > 0:
            # 计算态势差异 (只看rotation和flow，bias允许不同)
            # 原因: bias是位置相关的，不应强制一致
            gestalt_dynamic = pred_gestalt[..., 2:]  # [B, N, 2] (rotation, flow)
            
            # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
            gestalt_diff = gestalt_dynamic.unsqueeze(2) - gestalt_dynamic.unsqueeze(1)
            gestalt_diff_sq = (gestalt_diff ** 2).sum(dim=-1)  # [B, N, N]
            
            loss_smoothness = (gestalt_diff_sq * neighbor_mask.float()).sum() / neighbor_mask.sum().clamp(min=1)
    
    # =========================================================================
    # 4. 总损失加权
    # =========================================================================
    # 权重配置说明:
    # - w_pixel = 1.0: 像素监督是基础，但不是唯一真理
    # - w_semantic = 2.0: **提高到2.0**，强化文本语义学习
    # - w_smooth = 0.3: **降低到0.3**，避免过度平滑
    
    w_pixel = 1.0
    w_semantic = 2.0    # 从0.5提升到2.0！
    w_smooth = 0.3      # 从0.5降低到0.3
    
    total_gestalt_loss = (
        w_pixel * loss_pixel + 
        w_semantic * loss_semantic + 
        w_smooth * loss_smoothness
    )
    
    # 返回细分损失，用于训练日志和调试
    return total_gestalt_loss, loss_pixel, loss_semantic, loss_smoothness

# =============================================================================
# [NEW V10.0] 文本-态势对齐损失
# =============================================================================
def compute_text_gestalt_alignment_loss(
    gestalt_text,      # [B, N, 4] 文本推理的态势
    gestalt_decoder,   # [B, N, 4] Decoder预测的态势
    kg_class_ids,      # [B, N]
    loss_mask          # [B, N]
):
    """
    [NEW V10.0] 文本-态势对齐损失
    目的: 强制文本推理的态势符合类别先验
    
    策略: 对每个类别，计算其文本推理态势与先验均值的距离
    """
    device = gestalt_text.device
    B, N = kg_class_ids.shape
    
    loss_align = torch.tensor(0.0, device=device)
    count = 0
    
    # 对每个类别，计算其态势的聚类损失
    for cid, (f_mean, f_std, r_mean, r_std, _) in SEMANTIC_GESTALT_PRIORS.items():
        mask = (kg_class_ids == cid) & (loss_mask > 0)
        if mask.sum() < 2:  # 至少要有2个样本
            continue
        
        # 该类别的文本态势应该接近先验均值
        text_gestalt_cls = gestalt_text[mask]  # [K, 4]
        
        # 构造目标: bias用0（因为位置相关），rotation和flow用先验均值
        target = torch.tensor([0.0, 0.0, r_mean, f_mean], device=device)
        target = target.unsqueeze(0).expand_as(text_gestalt_cls)
        
        loss_cls = F.mse_loss(text_gestalt_cls, target)
        loss_align += loss_cls
        count += 1
    
    # 如果没有足够的类别样本，返回0
    if count == 0:
        return loss_align
    
    return loss_align / count

# =============================================================================
# KL散度损失 (保持不变)
# =============================================================================
def compute_kl_loss(mu, logvar, free_bits=0.0):
    """
    [New] 健壮的 KL 散度计算函数，支持 Free Bits 策略。
    
    Args:
        mu: [B, latent_dim] 均值
        logvar: [B, latent_dim] 对数方差
        free_bits: float, KL 散度的最小阈值（Hinge Loss）。
                   如果 KL < free_bits，则 Loss 为 0。
                   这可以防止后验分布过早坍缩到先验，保留一定的编码信息。
                   
    Returns:
        kl_loss: scalar (averaged over batch)
    """
    # 1. 计算每个样本的 KL 散度
    # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_per_sample = torch.sum(kld_element, dim=1) # [B]
    
    # 2. 应用 Free Bits (Hinge Loss)
    if free_bits > 0.0:
        free_bits_tensor = torch.tensor(free_bits, device=mu.device)
        kld_per_sample = torch.max(kld_per_sample, free_bits_tensor)
        
    # 3. 对 Batch 求平均
    kl_loss = torch.mean(kld_per_sample)
    
    return kl_loss

# =============================================================================
# [NEW V10.0] 辅助函数: 获取动态Loss权重 (用于分阶段训练)
# =============================================================================
def get_gestalt_loss_weights(epoch, strategy='progressive'):
    """
    根据训练阶段返回不同的Loss权重
    
    Args:
        epoch: 当前epoch
        strategy: 'progressive' 或 'fixed'
    
    Returns:
        dict: {'pixel': float, 'semantic': float, 'smooth': float}
    """
    if strategy == 'fixed':
        return {'pixel': 1.0, 'semantic': 2.0, 'smooth': 0.3}
    
    elif strategy == 'progressive':
        # 分阶段训练策略
        if epoch < 10:
            # Stage 1: 主要学像素特征
            return {'pixel': 1.0, 'semantic': 0.5, 'smooth': 0.1}
        elif epoch < 30:
            # Stage 2: 增强语义学习
            return {'pixel': 1.0, 'semantic': 2.0, 'smooth': 0.3}
        else:
            # Stage 3: 进一步强化语义，降低像素依赖
            return {'pixel': 0.5, 'semantic': 3.0, 'smooth': 0.3}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# =============================================================================
# [NEW V10.0] 诊断函数: 分析语义先验覆盖率
# =============================================================================
def analyze_semantic_prior_coverage(kg_class_ids, loss_mask):
    """
    统计batch中有多少物体被语义先验覆盖
    
    Args:
        kg_class_ids: [B, N] 类别ID
        loss_mask: [B, N] 有效性掩码
    
    Returns:
        dict: 统计信息
    """
    device = kg_class_ids.device
    total_objects = loss_mask.sum().item()
    
    covered_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    for cid in SEMANTIC_GESTALT_PRIORS.keys():
        covered_mask = covered_mask | (kg_class_ids == cid)
    
    covered_objects = (covered_mask & (loss_mask > 0)).sum().item()
    coverage_rate = covered_objects / max(total_objects, 1)
    
    return {
        'total_objects': total_objects,
        'covered_objects': covered_objects,
        'coverage_rate': coverage_rate
    }