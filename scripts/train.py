# scripts/train.py (V10.0: Enhanced Text-to-Gestalt Learning)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os
import argparse 

# 获取当前脚本 (train.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (train.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
import trainers 
from trainers.rl_trainer import RLTrainer 
from trainers.loss import analyze_semantic_prior_coverage, get_gestalt_loss_weights  # [NEW V10.0]
from collections import Counter
import numpy as np 

# --- 辅助函数：计算类别权重 (解决类别偏差问题) ---
def compute_class_weights(dataset, num_classes: int, max_weight_ratio: float = 3.0):
    """
    计算数据集内所有布局元素的类别频率，并返回反向频率权重。
    返回 (num_classes + 1) 个权重，其中索引 0 对应 EOS。
    """
    element_class_counts = Counter()
    
    # 遍历整个数据集计算所有元素实例的类别计数
    for sample in dataset.data:
        # boxes 是 List[(cls_id, cx, cy, w, h)]，cls_id 是 2.0 - 10.0
        for cls_id_float, _, _, _, _ in sample['boxes']:
            # 映射到内部 ID 0-8 (对应元素 2-10)
            internal_element_id = int(cls_id_float) - 2
            if 0 <= internal_element_id < num_classes:
                element_class_counts[internal_element_id] += 1
                
    total_count = sum(element_class_counts.values())
    
    # 最终权重数组大小必须是 9 (元素) + 1 (EOS) = 10
    final_num_classes = num_classes + 1 
    
    if total_count == 0:
        print("[Warning] No valid elements found in dataset for weight calculation. Using uniform weights.")
        return torch.ones(final_num_classes)

    # 初始化权重: size 10 (索引 0 for EOS, 索引 1-9 for elements)
    weights = torch.zeros(final_num_classes) 
    
    # 计算 9 个元素 (内部 ID 0-8) 的权重，并将它们存储在索引 1-9
    for i in range(num_classes): # i goes from 0 to 8 (internal element ID)
        frequency = element_class_counts.get(i, 0) / total_count
        
        # 将元素的内部 ID i (0-8) 映射到新的索引 i+1 (1-9)
        new_index = i + 1 
        
        if frequency > 0:
            # 采用 log(1.0 + x) 或 log(1.02 + x) 来平滑和反转频率
            weights[new_index] = 1.0 / np.log(1.02 + frequency) 
        else:
            # 对于稀有/不存在的类别，赋予最高的权重
            weights[new_index] = 1.0 / np.log(1.02 + 1e-6) # 赋予最大权重
            
    # 为 EOS 类 (索引 0) 分配权重
    # 我们使用计算出的元素权重的平均值作为基线
    avg_element_weight = weights[1:].sum() / num_classes if num_classes > 0 else 1.0
    weights[0] = avg_element_weight # 将平均权重分配给 EOS (索引 0)
            
    # 权重钳位和标准化
    # 对所有 10 个类别计算平均权重
    avg_weight = weights.mean()
    max_allowed_weight = avg_weight * max_weight_ratio
    weights = torch.clamp(weights, max=max_allowed_weight)
    
    # 重新归一化，确保总和是 final_num_classes (10)
    weights = weights / weights.sum() * final_num_classes
    
    return weights.float()
# ------------------------------------------

# =============================================================================
# [NEW V10.0] 数据集统计分析函数
# =============================================================================
def analyze_dataset_statistics(dataset):
    """
    分析数据集的态势提取质量和语义先验覆盖率
    """
    print("\n" + "="*70)
    print(">>> DATASET STATISTICS ANALYSIS <<<")
    print("="*70)
    
    # 1. 态势提取有效率统计
    valid_gestalt_count = 0
    total_object_count = 0
    class_gestalt_stats = {cid: {'valid': 0, 'total': 0} for cid in range(2, 11)}
    
    print("\n[1/3] Analyzing Gestalt Extraction Coverage...")
    for i in range(min(len(dataset), 1000)):  # 采样前1000个样本
        sample = dataset[i]
        loss_mask = sample['loss_mask']
        gestalt_mask = sample['gestalt_mask']
        kg_class_ids = sample['kg_class_ids']
        
        for j in range(len(kg_class_ids)):
            if loss_mask[j] > 0:
                cid = kg_class_ids[j].item()
                if cid in class_gestalt_stats:
                    class_gestalt_stats[cid]['total'] += 1
                    if gestalt_mask[j] > 0:
                        class_gestalt_stats[cid]['valid'] += 1
                
                total_object_count += 1
                if gestalt_mask[j] > 0:
                    valid_gestalt_count += 1
    
    overall_coverage = 100 * valid_gestalt_count / max(total_object_count, 1)
    print(f"  Overall Gestalt Coverage: {valid_gestalt_count}/{total_object_count} ({overall_coverage:.1f}%)")
    
    print("\n  Per-Class Gestalt Coverage:")
    class_names = {2: "mountain", 3: "water", 4: "people", 5: "tree", 
                   6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"}
    for cid in sorted(class_gestalt_stats.keys()):
        stats = class_gestalt_stats[cid]
        if stats['total'] > 0:
            ratio = 100 * stats['valid'] / stats['total']
            name = class_names.get(cid, f"cls_{cid}")
            print(f"    {name:12s} (ID {cid}): {stats['valid']:4d}/{stats['total']:4d} ({ratio:5.1f}%)")
    
    # 2. 语义先验覆盖率
    print("\n[2/3] Analyzing Semantic Prior Coverage...")
    batch = dataset[0]
    coverage_stats = analyze_semantic_prior_coverage(
        batch['kg_class_ids'].unsqueeze(0), 
        batch['loss_mask'].unsqueeze(0)
    )
    print(f"  Semantic Prior Coverage: {coverage_stats['covered_objects']}/{coverage_stats['total_objects']} "
          f"({100*coverage_stats['coverage_rate']:.1f}%)")
    
    # 3. 态势数值分布统计
    print("\n[3/3] Analyzing Gestalt Value Distributions...")
    gestalt_values = {'bias_x': [], 'bias_y': [], 'rotation': [], 'flow': []}
    
    for i in range(min(len(dataset), 500)):
        sample = dataset[i]
        target_boxes = sample['target_boxes']  # [N, 8]
        loss_mask = sample['loss_mask']        # [N]
        gestalt_mask = sample['gestalt_mask']  # [N]
        
        valid_mask = (loss_mask > 0) & (gestalt_mask > 0)
        if valid_mask.sum() > 0:
            valid_gestalt = target_boxes[valid_mask, 4:]  # [K, 4]
            gestalt_values['bias_x'].extend(valid_gestalt[:, 0].tolist())
            gestalt_values['bias_y'].extend(valid_gestalt[:, 1].tolist())
            gestalt_values['rotation'].extend(valid_gestalt[:, 2].tolist())
            gestalt_values['flow'].extend(valid_gestalt[:, 3].tolist())
    
    for key, values in gestalt_values.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"  {key:12s}: mean={mean_val:6.3f}, std={std_val:6.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")
    
    print("="*70 + "\n")
    
    return {
        'gestalt_coverage': overall_coverage,
        'semantic_coverage': coverage_stats['coverage_rate'],
        'class_stats': class_gestalt_stats
    }


def main():
    # [NEW] 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Train or RL-Finetune Poem2Layout")
    parser.add_argument('--rl_tuning', action='store_true', help="Enable Reinforcement Learning fine-tuning mode")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to pretrained model checkpoint (required for RL tuning)")
    parser.add_argument('--skip_analysis', action='store_true', help="Skip dataset statistics analysis")  # [NEW V10.0]
    parser.add_argument('--gestalt_strategy', type=str, default='progressive', 
                       choices=['fixed', 'progressive'], 
                       help="Gestalt loss weighting strategy")  # [NEW V10.0]
    args = parser.parse_args()

    # 1. Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Init tokenizer and load FULL dataset for weight calculation
    model_config = config['model']
    train_config = config['training'] # 获取 training 配置块
    
    # 重新初始化数据集以确保数据完整性
    dataset = PoegraphLayoutDataset(
        xlsx_path=model_config['xlsx_path'],
        labels_dir=model_config['labels_dir'],
        bert_model_path=model_config['bert_path'],
        max_layout_length=model_config['max_layout_length'],
        max_text_length=model_config['max_text_length']
    )
    
    # ==========================================================================
    # [NEW V10.0] 数据集质量分析
    # ==========================================================================
    if not args.skip_analysis:
        dataset_stats = analyze_dataset_statistics(dataset)
        
        # 根据统计结果给出建议
        if dataset_stats['gestalt_coverage'] < 50.0:
            print("⚠️  [WARNING] Gestalt extraction coverage is low (<50%)!")
            print("    → Consider adjusting thresholds in VisualGestaltExtractor")
            print("    → Or increase semantic prior weight to compensate\n")
        
        if dataset_stats['semantic_coverage'] < 80.0:
            print("⚠️  [WARNING] Many objects are not covered by semantic priors!")
            print("    → Consider adding more class IDs to SEMANTIC_GESTALT_PRIORS\n")
    
    # --- 计算类别权重 ---
    num_element_classes = model_config['num_classes'] # 9
    class_weights_tensor = compute_class_weights(dataset, num_element_classes)
    
    print(f"Calculated Class Weights (Internal 0:EOS, 1-9:Elements 2-10): {class_weights_tensor.tolist()}")
    # ---------------------------

    # 3. Init model (传入所有损失权重，包括新增的 Gestalt Loss Weight)
    print(f"\nInitializing model with latent_dim={model_config.get('latent_dim', 32)}...")
    model = Poem2LayoutGenerator(
        bert_path=model_config['bert_path'],
        num_classes=num_element_classes, # 实际元素类别数 (9)
        # --- BBox Discrete Parameters (Legacy) ---
        num_bbox_bins=model_config.get('num_bbox_bins', 1000),
        bbox_embed_dim=model_config.get('bbox_embed_dim', 24),
        # --------------------------------------
        hidden_size=model_config['hidden_size'],
        bb_size=model_config['bb_size'],
        decoder_layers=model_config['decoder_layers'],
        decoder_heads=model_config['decoder_heads'],
        dropout=model_config['dropout'],
        
        # === CVAE 参数 ===
        latent_dim=model_config.get('latent_dim', 32),
        
        # --- 传入所有损失权重 ---
        coord_loss_weight=model_config.get('coord_loss_weight', 0.0),
        iou_loss_weight=model_config.get('iou_loss_weight', 1.0), 
        reg_loss_weight=model_config.get('reg_loss_weight', 1.0),    
        cls_loss_weight=model_config.get('cls_loss_weight', 0.0),    
        count_loss_weight=model_config.get('count_loss_weight', 0.0),
        area_loss_weight=model_config.get('area_loss_weight', 1.0),
        
        # 核心逻辑权重
        relation_loss_weight=model_config.get('relation_loss_weight', 5.0),
        overlap_loss_weight=model_config.get('overlap_loss_weight', 3.0),
        size_loss_weight=model_config.get('size_loss_weight', 2.0),
        
        # 审美权重
        alignment_loss_weight=model_config.get('alignment_loss_weight', 0.0),
        balance_loss_weight=model_config.get('balance_loss_weight', 0.0),
        
        # [V5.4] 聚类损失权重
        clustering_loss_weight=model_config.get('clustering_loss_weight', 1.0),

        # [NEW V6.0] 一致性损失权重
        consistency_loss_weight=model_config.get('consistency_loss_weight', 1.0),
        
        # [NEW V10.0] 视觉态势损失权重 (Visual Gestalt Supervision)
        # 从配置文件读取，默认 2.0
        gestalt_loss_weight=model_config.get('gestalt_loss_weight', 2.0),
        
        class_weights=class_weights_tensor 
        # -----------------------------------------
    )
    
    # ==========================================================================
    # [NEW V10.0] 将训练策略传递给config
    # ==========================================================================
    config['gestalt_strategy'] = args.gestalt_strategy
    print(f"Gestalt Loss Strategy: {args.gestalt_strategy}")

    # 4. Split dataset and init data loaders
    total_size = len(dataset)
    train_size = int(0.8 * total_size) # 80%
    val_size = int(0.1 * total_size)   # 10%
    test_size = total_size - train_size - val_size # 剩余为 10%

    # 执行 80/10/10 随机划分
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"\nDataset split: Train={train_size}, Validation={val_size}, Test={test_size}")

    # [NOTE] Batch Size 读取自配置文件
    batch_size = train_config['batch_size']
    print(f"Using Batch Size: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=layout_collate_fn,
        num_workers=4, # 大批量数据建议开启多线程加载
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=layout_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=layout_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # --- 获取 tokenizer 和一个固定样例 ---
    tokenizer = dataset.tokenizer 
    # 从验证集中选择第一个样例
    example_idx_in_full_dataset = val_dataset.indices[0]
    example_poem = dataset.data[example_idx_in_full_dataset]
    
    # **打印固定推理样例的 KG 向量和空间矩阵**
    print("\n" + "="*60)
    print(">>> INFERENCE EXAMPLE CONFIGURATION <<<")
    print("="*60)
    print(f"Poem: '{example_poem['poem']}'")
    print(f"GT Boxes: {example_poem['boxes']}")
    print("\n--- Knowledge Graph Debug ---")
    
    # 1. 视觉向量
    kg_vector_example = dataset.pkg.extract_visual_feature_vector(example_poem['poem'])
    print(f"KG Vector (0:mountain(2), 1:water(3), ..., 8:animal(10)):")
    print(f"  {kg_vector_example.tolist()}")
    
    # 2. [NEW] 空间矩阵
    kg_spatial_matrix_example = dataset.pkg.extract_spatial_matrix(example_poem['poem'])
    print("\nSpatial Matrix (9x9):")
    print(kg_spatial_matrix_example)
    print("="*60 + "\n")

    # 5. Logic Branch: RL Tuning OR Supervised Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.rl_tuning:
        print("\n" + "="*70)
        print(">>> ENTERING RL FINE-TUNING MODE (SCST) <<<")
        print("="*70 + "\n")
        
        # 1. 必须加载预训练模型
        if args.checkpoint is None:
            raise ValueError("RL tuning requires a pretrained checkpoint! Use --checkpoint")
        
        print(f"Loading pretrained model from {args.checkpoint}...")
        # map_location 确保在 CPU/GPU 间迁移兼容
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. 读取 RL 配置参数
        rl_lr = float(train_config.get('rl_learning_rate', 5e-6))
        rl_epochs = int(train_config.get('rl_epochs', 50))
        
        print(f"RL Config -> Learning Rate: {rl_lr:.2e} (float) | Epochs: {rl_epochs} (int)")

        # 3. 初始化 RLTrainer
        trainer = RLTrainer(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # 4. 强制覆盖优化器的学习率
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = rl_lr
        
        # [NEW] 初始化最佳奖励记录
        best_reward = -float('inf')

        # 5. 开始 RL 训练循环
        for epoch in range(rl_epochs):
            # [MODIFIED] 接收 train_rl_epoch 返回的 avg_reward
            avg_reward = trainer.train_rl_epoch(epoch)
            
            # [NEW] 可视化：每�� RL 结束生成一张样例图，直观看到模型变化
            trainer._run_inference_example(epoch)
            
            # === [NEW] 保存逻辑 A: 保存最棒的模型 (Best Reward) ===
            # 这是 infer.py 优先加载的模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_save_path = os.path.join(train_config['output_dir'], "rl_best_reward.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'avg_reward': avg_reward,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'rl_config': {'lr': rl_lr}
                }, best_save_path)
                print(f"🌟 [New Best] Avg Reward {avg_reward:.4f} achieved! Model saved to {best_save_path}")

            # === [MODIFIED] 保存逻辑 B: 每 10 个 Epoch 保存一次 ===
            if (epoch + 1) % 100 == 0:
                rl_save_path = os.path.join(train_config['output_dir'], f"rl_finetuned_epoch_{epoch+1}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'avg_reward': avg_reward,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'rl_config': {'lr': rl_lr}
                }, rl_save_path)
                print(f"💾 [Checkpoint] Epoch {epoch+1} saved to {rl_save_path}")
                
    else:
        # 原有的监督训练逻辑
        print("\n" + "="*70)
        print(">>> STARTING SUPERVISED TRAINING (Text-to-Gestalt Enhanced) <<<")
        print("="*70)
        print(f"Total Epochs: {train_config['epochs']} | Batch Size: {batch_size}")
        print(f"Gestalt Loss Strategy: {args.gestalt_strategy}")
        print("="*70 + "\n")
        
        trainer = trainers.LayoutTrainer(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        trainer.train()

if __name__ == "__main__":
    main()