"""
训练数据生成：向菜品图片中贴合一元硬币

将 coin.png（透明背景）合成到菜品图片的随机角落，
用于生成 YOLO 硬币检测的训练数据。
"""

import cv2
import random
import numpy as np
from pathlib import Path


COIN_PNG = "coin.png"

# 硬币占图片短边的比例，模拟不同拍摄距离
COIN_SCALE_RANGE = (0.15, 0.25)

# 硬币距图片边缘的间距（像素）
MARGIN = 10


def load_coin(coin_path: str) -> np.ndarray:
    """加载硬币 PNG（含 alpha 通道），返回 BGRA 图像。"""
    coin = cv2.imread(coin_path, cv2.IMREAD_UNCHANGED)
    if coin is None:
        raise FileNotFoundError(f"找不到硬币图片: {coin_path}")
    if coin.shape[2] == 3:
        # 没有 alpha 通道，加一个全不透明的
        alpha = np.full(coin.shape[:2], 255, dtype=np.uint8)
        coin = np.dstack([coin, alpha])
    return coin


def paste_coin(bg: np.ndarray, coin_bgra: np.ndarray, corner: str, margin: int = MARGIN) -> np.ndarray:
    """
    将硬币贴到背景图的指定角落。

    Args:
        bg:        背景图 (BGR)
        coin_bgra: 硬币图 (BGRA)
        corner:    "top_left" / "top_right" / "bottom_left" / "bottom_right"
        margin:    距边缘像素数

    Returns:
        合成后的 BGR 图像
    """
    H, W = bg.shape[:2]
    ch, cw = coin_bgra.shape[:2]

    if corner == "top_left":
        x, y = margin, margin
    elif corner == "top_right":
        x, y = W - cw - margin, margin
    elif corner == "bottom_left":
        x, y = margin, H - ch - margin
    else:  # bottom_right
        x, y = W - cw - margin, H - ch - margin

    # 边界保护
    x = max(0, min(x, W - cw))
    y = max(0, min(y, H - ch))

    result = bg.copy()
    roi = result[y:y+ch, x:x+cw]

    coin_bgr = coin_bgra[:, :, :3]
    alpha = coin_bgra[:, :, 3:4] / 255.0

    roi[:] = (coin_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    return result


def save_yolo_label(image_save_path: str, bbox_xyxy: tuple, image_size: tuple):
    """
    保存 YOLO 格式标注文件（与图片同名的 .txt）。
    格式：class_id cx cy w h （均归一化到 0-1）
    class_id = 0 （硬币）
    """
    W, H = image_size
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H

    label_path = Path(image_save_path).with_suffix(".txt")
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def add_coin_to_image(
    image_path: str,
    save_path: str,
    coin_path: str = COIN_PNG,
    corner: str = None,
) -> dict:
    """
    向图片中添加硬币并保存。

    Returns:
        包含 corner 和 coin bbox 的 dict，用于后续生成 YOLO 标注
    """
    bg = cv2.imread(image_path)
    if bg is None:
        raise FileNotFoundError(f"找不到图片: {image_path}")

    coin_bgra = load_coin(coin_path)

    # 按比例缩放硬币
    H, W = bg.shape[:2]
    scale = random.uniform(*COIN_SCALE_RANGE)
    target_size = int(min(H, W) * scale)
    coin_bgra = cv2.resize(coin_bgra, (target_size, target_size))

    # 随机选角落
    if corner is None:
        corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    result = paste_coin(bg, coin_bgra, corner)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, result)

    # 计算硬币 bbox（用于 YOLO 标注）
    ch, cw = coin_bgra.shape[:2]
    if corner == "top_left":
        x, y = MARGIN, MARGIN
    elif corner == "top_right":
        x, y = W - cw - MARGIN, MARGIN
    elif corner == "bottom_left":
        x, y = MARGIN, H - ch - MARGIN
    else:
        x, y = W - cw - MARGIN, H - ch - MARGIN

    bbox = (x, y, x + cw, y + ch)

    # 保存 YOLO 格式标注
    save_yolo_label(save_path, bbox, (W, H))

    return {"corner": corner, "bbox_xyxy": bbox, "image_size": (W, H)}


def batch_generate(input_dir: str, output_dir: str, coin_path: str = COIN_PNG):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    print(f"共 {len(images)} 张图片")

    for img_path in images:
        save_path = Path(output_dir) / f"{img_path.stem}_coin.jpg"
        try:
            info = add_coin_to_image(str(img_path), str(save_path), coin_path)
            print(f"  [完成] {img_path.name} → {info['corner']}  bbox={info['bbox_xyxy']}")
        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")


if __name__ == "__main__":
    info = add_coin_to_image(
        image_path="Data/release_data/test/000094.jpg",
        save_path="coin_training_data/test_coin.jpg",
    )
    print(f"完成：{info}")
