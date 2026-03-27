"""
YOLO 微调训练入口。

用法：
  python train.py              # 使用 config.py 中的默认参数
  python train.py --epochs 80  # 覆盖单个参数

训练结果保存在 coin_yolo/runs/<run_name>/
"""

import argparse
import sys
from pathlib import Path

# 把 coin_yolo 目录加入 path，保证 import config 可用
sys.path.insert(0, str(Path(__file__).parent))

import config
import dataset_utils


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 for coin detection")
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--batch",       type=int,   default=config.BATCH_SIZE)
    p.add_argument("--imgsz",       type=int,   default=config.IMAGE_SIZE)
    p.add_argument("--lr0",         type=float, default=config.LR0)
    p.add_argument("--lrf",         type=float, default=config.LRF)
    p.add_argument("--patience",    type=int,   default=config.PATIENCE)
    p.add_argument("--pretrained",  type=str,   default=config.PRETRAINED)
    p.add_argument("--name",        type=str,   default=config.RUN_NAME)
    p.add_argument("--skip-prepare",action="store_true",
                   help="跳过数据集准备（已经执行过 dataset_utils.py 时使用）")
    p.add_argument("--device",      type=str,   default="",
                   help="训练设备，留空自动选择（云GPU填 0，Mac CPU填 cpu）")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. 准备数据集 ─────────────────────────────────────────────────────────
    if not args.skip_prepare:
        print("=" * 50)
        print("Step 1: 准备数据集")
        print("=" * 50)
        dataset_utils.prepare_dataset()
    else:
        print("[跳过] 数据集准备")

    if not config.DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml 不存在: {config.DATA_YAML}，请先运行 dataset_utils.py")

    # ── 2. 加载模型 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 2: 加载预训练模型")
    print("=" * 50)

    from ultralytics import YOLO
    model = YOLO(args.pretrained)
    print(f"已加载: {args.pretrained}")

    # ── 3. 训练 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 3: 开始训练")
    print("=" * 50)

    train_kwargs = dict(
        data      = str(config.DATA_YAML),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        lr0       = args.lr0,
        lrf       = args.lrf,
        patience  = args.patience,
        project   = str(config.RUNS_DIR),
        name      = args.name,
        exist_ok  = True,
        verbose   = True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # ── 4. 验证 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 4: 在验证集上评估")
    print("=" * 50)
    metrics = model.val()
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

    best_weights = config.RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\n最佳权重已保存到: {best_weights}")


if __name__ == "__main__":
    main()
