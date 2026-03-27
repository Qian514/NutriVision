"""
数据集工具：
  1. 从多个原始数据目录收集图片/标注对
  2. 随机划分 train / val
  3. 复制到 YOLO 标准目录结构
  4. 生成 data.yaml
"""

import shutil
import random
import yaml
from pathlib import Path

import config


def collect_pairs(raw_dirs: list[Path]) -> list[tuple[Path, Path]]:
    """
    从多个目录中收集 (image, label) 对。
    要求：同目录下同名的 .jpg/.png 与 .txt 配对。
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs = []

    for d in raw_dirs:
        if not d.exists():
            print(f"[跳过] 目录不存在: {d}")
            continue
        for img_path in d.rglob("*"):
            if img_path.suffix.lower() not in exts:
                continue
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                print(f"  [警告] 找不到标注文件，跳过: {img_path.name}")
                continue
            pairs.append((img_path, label_path))

    print(f"[收集] 共找到 {len(pairs)} 对有效数据")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float = config.VAL_RATIO,
    seed: int = 42,
) -> tuple[list, list]:
    """随机打乱后按比例划分为 train / val。"""
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_n:], shuffled[:val_n]


def copy_to_dataset(
    train_pairs: list[tuple[Path, Path]],
    val_pairs:   list[tuple[Path, Path]],
):
    """将文件复制到 YOLO 标准目录结构，存在则覆盖。"""
    dirs = [
        config.TRAIN_IMG, config.VAL_IMG,
        config.TRAIN_LABEL, config.VAL_LABEL,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    def _copy(pairs, img_dir, lbl_dir):
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            shutil.copy2(lbl_src, lbl_dir / lbl_src.name)

    _copy(train_pairs, config.TRAIN_IMG, config.TRAIN_LABEL)
    _copy(val_pairs,   config.VAL_IMG,   config.VAL_LABEL)

    print(f"[数据集] train={len(train_pairs)}  val={len(val_pairs)}")


def write_yaml():
    """生成 YOLO 格式的 data.yaml。"""
    data = {
        "path":  str(config.DATASET_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(config.CLASS_NAMES),
        "names": config.CLASS_NAMES,
    }
    with open(config.DATA_YAML, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"[yaml] 已写入 {config.DATA_YAML}")


def prepare_dataset():
    """一键完成数据集准备（收集 → 划分 → 复制 → 生成 yaml）。"""
    pairs = collect_pairs(config.RAW_DATA_DIRS)
    if not pairs:
        raise RuntimeError("没有找到任何数据，请先运行数据生成脚本。")

    train_pairs, val_pairs = split_pairs(pairs)
    copy_to_dataset(train_pairs, val_pairs)
    write_yaml()
    return train_pairs, val_pairs


if __name__ == "__main__":
    prepare_dataset()
