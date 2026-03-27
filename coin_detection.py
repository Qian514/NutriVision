"""
硬币检测模块

流程：
  1. 预处理（灰度、模糊、边缘检测）
  2. 轮廓检测 → 筛选近似椭圆的轮廓
  3. 拟合椭圆 → 取长轴作为参考直径
  4. 输出 px_per_mm 比例尺

默认参照：1元人民币，直径 25mm
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


COIN_DIAMETER_MM = {
    "1yuan": 25.0,
    "5jiao": 20.5,
    "1jiao": 19.0,
}


@dataclass
class CoinDetectionResult:
    found: bool
    px_per_mm: float = 0.0
    ellipse: tuple = None        # cv2.fitEllipse 的原始输出 ((cx,cy),(a,b),angle)
    major_axis_px: float = 0.0
    coin_type: str = ""


def detect_coin(
    image_input,
    coin_type: str = "1yuan",
    min_area: int = 1000,        # 轮廓最小面积，过滤噪声
    circularity_thresh: float = 0.75,  # 圆度阈值（4π·面积/周长²），越接近1越圆
) -> CoinDetectionResult:
    """
    Args:
        image_input: 图片路径（str）或已读取的 BGR numpy 数组
        coin_type:   硬币类型，见 COIN_DIAMETER_MM
        min_area:    轮廓最小面积
        circularity_thresh: 圆度筛选阈值

    Returns:
        CoinDetectionResult
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"找不到图片: {image_input}")
    else:
        img = image_input

    real_diameter_mm = COIN_DIAMETER_MM[coin_type]

    # 1. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # 闭运算：连接断开的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 2. 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # 圆度：完美圆形 = 1.0
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < circularity_thresh:
            continue

        # 需要至少5个点才能拟合椭圆
        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (minor_ax, major_ax), angle = ellipse
        # 确保 major_ax >= minor_ax
        major_ax = max(minor_ax, major_ax)
        minor_ax = min(minor_ax, major_ax)

        # 长短轴比：太扁的椭圆不像硬币（倾斜拍摄时比值一般不超过2）
        axis_ratio = major_ax / minor_ax if minor_ax > 0 else 99
        if axis_ratio > 2.5:
            continue

        candidates.append({
            "ellipse": ellipse,
            "major_axis_px": major_ax,
            "circularity": circularity,
            "area": area,
            "contour": cnt,
        })

    if not candidates:
        return CoinDetectionResult(found=False)

    # 取面积最小的候选（硬币远小于盘子等其他圆形物体）
    best = min(candidates, key=lambda c: c["area"])
    major_axis_px = best["major_axis_px"]
    px_per_mm = major_axis_px / real_diameter_mm

    return CoinDetectionResult(
        found=True,
        px_per_mm=px_per_mm,
        ellipse=best["ellipse"],
        major_axis_px=major_axis_px,
        coin_type=coin_type,
    )


def visualize_coin(image_input, result: CoinDetectionResult, save_path: str = None):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_rgb)
    ax.axis("off")

    if not result.found:
        ax.set_title("Coin not detected")
    else:
        # 绘制拟合椭圆
        from matplotlib.patches import Ellipse
        (cx, cy), (minor_ax, major_ax), angle = result.ellipse
        major_ax = max(minor_ax, major_ax)
        minor_ax = min(minor_ax, major_ax)
        ellipse_patch = Ellipse(
            (cx, cy), major_ax, minor_ax,
            angle=-angle,  # matplotlib 角度方向与 cv2 相反
            edgecolor="lime", facecolor="none", linewidth=2.5,
        )
        ax.add_patch(ellipse_patch)
        ax.plot(cx, cy, "+", color="lime", markersize=12, markeredgewidth=2)
        ax.set_title(
            f"Coin detected ({result.coin_type})  |  "
            f"major axis: {result.major_axis_px:.1f}px  |  "
            f"scale: {result.px_per_mm:.3f} px/mm"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    IMAGE = "test_coin.jpg"

    result = detect_coin(IMAGE, coin_type="1yuan")

    if result.found:
        print(f"检测成功")
        print(f"  长轴像素: {result.major_axis_px:.1f} px")
        print(f"  比例尺:   {result.px_per_mm:.4f} px/mm")
        print(f"  1cm = {result.px_per_mm * 10:.1f} px")
    else:
        print("未检测到硬币，请检查图片或调整参数")

    visualize_coin(IMAGE, result, save_path="coin_detection_result.png")
