#!/bin/bash

# ================= 配置区域 =================
# GPU 动态调度 (按需开启，单卡通常不需要改)
# export CUDA_VISIBLE_DEVICES=0 

# 优化显存分配策略 (防止 OOM，保持开启)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# [关键修复] 自动定位项目根目录
# 1. 获取脚本所在的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. 推断项目根目录 (假设脚本在 stage2_generation 目录下，根目录则是上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. 强制切换工作目录到项目根目录
cd "$PROJECT_ROOT"
echo "📂 工作目录已自动切换至: $(pwd)"

# [缓存设置]
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"

# [核心修改 1: 数据路径]
# 指向正确的数据集目录 (请确认此处是否为您最新的 v9_2 数据集)
DATA_DIR="$PROJECT_ROOT/taiyi_energy_dataset_v9_2" 

# [核心修改 2: 输出路径]
# 改为 V19_breath，对应我们的"呼吸感"策略
OUTPUT_DIR="$PROJECT_ROOT/outputs/taiyi_shanshui_v19_breath"

# [基础模型路径] (保持本地绝对路径)
MODEL_NAME="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# Accelerate 配置文件路径
ACCELERATE_CONFIG="stage2_generation/configs/accelerate_config.yaml"

# ===========================================

# 1. 安全检查
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "❌ 错误: 在 $DATA_DIR 中找不到 train.jsonl"
  echo "👉 请检查数据路径是否正确"
  exit 1
fi

# 2. 检查 Accelerate 配置
if [ ! -f "$ACCELERATE_CONFIG" ]; then
  echo "⚠️ 生成默认配置..."
  mkdir -p $(dirname "$ACCELERATE_CONFIG")
  cat > "$ACCELERATE_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
fi

# 3. 启动训练 (V19.0)
echo "========================================================"
echo "🚀 启动 Stage 2 V19.0 训练 (Rank 64 | Tuned LR | 强Mask+柔LoRA)"
echo "   数据源: $DATA_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo "   策略: 强控制力 + 水墨韵味平衡"
echo "========================================================"

# [核心参数调整]
accelerate launch --config_file "$ACCELERATE_CONFIG" --mixed_precision="fp16" stage2_generation/scripts/train_taiyi.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  \
  --num_train_epochs=50 \
  --checkpointing_steps=5000 \
  --mixed_precision="fp16" \
  \
  --learning_rate=2e-5 \
  --learning_rate_lora=5e-5 \
  \
  --lora_rank=64 \
  --lora_alpha_ratio=1.0 \
  \
  --lambda_struct=0.0 \
  --lambda_energy=0.0 \
  \
  --snr_gamma=5.0 \
  --offset_noise_scale=0.05

echo "✅ 训练结束。请检查验证图是否兼具构图控制与水墨笔触！"