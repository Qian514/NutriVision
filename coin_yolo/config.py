"""
训练配置：所有路径、超参数集中在此处管理。
"""

from pathlib import Path

# ── 项目根目录 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # Nutrition/

# ── 数据源（贴图生成的原始数据） ──────────────────────────────────────────────
# 可以把多个来源文件夹都放进来，dataset_utils 会统一处理
RAW_DATA_DIRS = [
    ROOT / "coin_training_data",          # generate_coin_data.py 的输出
    ROOT / "Data" / "dish_coin_output",   # generate_coin_data_fixed.py 的输出
]

# ── YOLO 数据集目录 ────────────────────────────────────────────────────────────
DATASET_DIR  = ROOT / "coin_yolo" / "dataset"
TRAIN_IMG    = DATASET_DIR / "images" / "train"
VAL_IMG      = DATASET_DIR / "images" / "val"
TRAIN_LABEL  = DATASET_DIR / "labels" / "train"
VAL_LABEL    = DATASET_DIR / "labels" / "val"
DATA_YAML    = ROOT / "coin_yolo" / "data.yaml"

# ── 类别 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["coin"]      # class_id = 0

# ── 数据集划分比例 ─────────────────────────────────────────────────────────────
VAL_RATIO    = 0.15          # 15% 作为验证集

# ── 训练超参数 ─────────────────────────────────────────────────────────────────
PRETRAINED   = "yolov8n.pt"  # 预训练权重（n / s / m / l / x，越大越准但越慢）
EPOCHS       = 50
BATCH_SIZE   = 16
IMAGE_SIZE   = 640
LR0          = 0.01          # 初始学习率
LRF          = 0.01          # 最终学习率 = LR0 * LRF
PATIENCE     = 15            # 早停轮数

# ── 输出 ──────────────────────────────────────────────────────────────────────
RUNS_DIR     = ROOT / "coin_yolo" / "runs"
RUN_NAME     = "coin_v1"
