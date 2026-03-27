"""
将硬币贴到 Data/dish_coin/ 各子文件夹的图片中。

文件夹名称对应硬币位置：
  left-top      → 左上角
  left-botttom  → 左下角（原始文件夹名含笔误）
  right-top     → 右上角
  right-bottom  → 右下角

输出到 Data/dish_coin_output/，同时生成 YOLO 格式标注 .txt。
"""

import cv2
import random
import numpy as np
from pathlib import Path

from generate_coin_data import load_coin, paste_coin, save_yolo_label, COIN_SCALE_RANGE, MARGIN

INPUT_ROOT = Path("Data/dish_coin")
OUTPUT_ROOT = Path("Data/dish_coin_output")
COIN_PNG = "coin.png"

# 文件夹名 → paste_coin 使用的 corner 参数
FOLDER_TO_CORNER = {
    "left-top":     "top_left",
    "left-botttom": "bottom_left",   # 原始文件夹名有笔误
    "right-top":    "top_right",
    "right-bottom": "bottom_right",
}

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def process_folder(folder: Path, corner: str, coin_bgra_orig: np.ndarray):
    images = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    print(f"\n[{folder.name}] → {corner}，共 {len(images)} 张")

    out_dir = OUTPUT_ROOT / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        bg = cv2.imread(str(img_path))
        if bg is None:
            print(f"  [跳过] 无法读取 {img_path.name}")
            continue

        H, W = bg.shape[:2]
        scale = random.uniform(*COIN_SCALE_RANGE)
        target_size = int(min(H, W) * scale)
        coin_bgra = cv2.resize(coin_bgra_orig, (target_size, target_size))

        result = paste_coin(bg, coin_bgra, corner)

        save_path = out_dir / f"{img_path.stem}_coin.jpg"
        cv2.imwrite(str(save_path), result)

        # 计算 bbox
        ch, cw = coin_bgra.shape[:2]
        if corner == "top_left":
            x, y = MARGIN, MARGIN
        elif corner == "top_right":
            x, y = W - cw - MARGIN, MARGIN
        elif corner == "bottom_left":
            x, y = MARGIN, H - ch - MARGIN
        else:  # bottom_right
            x, y = W - cw - MARGIN, H - ch - MARGIN

        bbox = (x, y, x + cw, y + ch)
        save_yolo_label(str(save_path), bbox, (W, H))

        print(f"  [完成] {img_path.name} → bbox={bbox}")


def main():
    coin_bgra = load_coin(COIN_PNG)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder_name, corner in FOLDER_TO_CORNER.items():
        folder = INPUT_ROOT / folder_name
        if not folder.exists():
            print(f"[跳过] 文件夹不存在: {folder}")
            continue
        process_folder(folder, corner, coin_bgra)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
