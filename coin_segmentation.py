"""
硬币精细分割模块

流程：
  1. YOLO 检测硬币 → 得到 bounding box
  2. 以 bbox 作为 SAM 的 prompt → 精确分割硬币 mask
  3. 由 mask 像素面积推算比例尺（面积法，对相机倾斜鲁棒）
  4. 同时用椭圆拟合得到直径法比例尺（作为校验）

比例尺说明：
  - px_per_mm   : 线性比例尺（px/mm），由面积法推导 sqrt(px²/mm²)
  - px2_per_mm2 : 面积比例尺（px²/mm²），直接用于食物面积换算

1元硬币参数：直径 25mm，面积 = π×12.5² ≈ 490.87 mm²
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from pathlib import Path


COIN_REAL_AREA_MM2 = {
    "1yuan": np.pi * 12.25 ** 2,   
}

COIN_DIAMETER_MM = {
    "1yuan": 22.5,
}


@dataclass
class CoinSegResult:
    found: bool
    # ── 检测结果 ──────────────────────────────────────────────────────────────
    bbox_xyxy: tuple = None          # YOLO 检测框 (x1,y1,x2,y2)
    yolo_conf: float = 0.0
    # ── 分割结果 ──────────────────────────────────────────────────────────────
    mask: np.ndarray = None          # bool mask，与原图同尺寸
    pixel_area: int = 0              # 硬币区域像素数
    # ── 比例尺（面积法，主用） ────────────────────────────────────────────────
    px2_per_mm2: float = 0.0         # 面积比例尺：px² / mm²
    px_per_mm: float = 0.0           # 线性比例尺：sqrt(px2_per_mm2)
    # ── 比例尺（椭圆直径法，校验用） ─────────────────────────────────────────
    ellipse: tuple = None            # cv2.fitEllipse 输出
    px_per_mm_ellipse: float = 0.0
    # ── 元信息 ────────────────────────────────────────────────────────────────
    coin_type: str = "1yuan"


class CoinSegmentor:
    """
    YOLO + SAM 联合硬币分割器。

    Args:
        yolo_weights:  训练好的 YOLO 权重路径（.pt）
        sam_checkpoint: SAM 权重路径（sam_vit_b_01ec64.pth 等）
        sam_model_type: "vit_b" / "vit_l" / "vit_h"
        conf_thresh:   YOLO 置信度阈值
    """

    def __init__(
        self,
        yolo_weights: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        conf_thresh: float = 0.25,
    ):
        # ── 加载 YOLO ─────────────────────────────────────────────────────────
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_weights)
        self.conf_thresh = conf_thresh

        # ── 加载 SAM ──────────────────────────────────────────────────────────
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to("cpu")   # Mac MPS 不支持 float64
        self.sam_predictor = SamPredictor(sam)

        print(f"[CoinSegmentor] YOLO: {yolo_weights}")
        print(f"[CoinSegmentor] SAM : {sam_checkpoint} ({sam_model_type})")

    def segment(self, image_input, coin_type: str = "1yuan") -> CoinSegResult:
        """
        Args:
            image_input: 图片路径（str/Path）或 BGR numpy 数组
            coin_type:   "1yuan" / "5jiao" / "1jiao"

        Returns:
            CoinSegResult
        """
        # ── 读图 ──────────────────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise FileNotFoundError(f"找不到图片: {image_input}")
        else:
            bgr = image_input.copy()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── Step 1: YOLO 检测 ─────────────────────────────────────────────────
        yolo_results = self.yolo(bgr, conf=self.conf_thresh, verbose=False)
        boxes = yolo_results[0].boxes

        if len(boxes) == 0:
            return CoinSegResult(found=False, coin_type=coin_type)

        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        yolo_conf = float(boxes.conf[best_idx])

        # ── Step 2: SAM 精细分割（以 YOLO bbox 为 prompt） ────────────────────
        self.sam_predictor.set_image(rgb)
        bbox_prompt = np.array([x1, y1, x2, y2])
        masks, scores, _ = self.sam_predictor.predict(
            box=bbox_prompt,
            multimask_output=True,   # 生成3个候选，取最高分
        )
        best_mask_idx = int(scores.argmax())
        mask = masks[best_mask_idx].astype(bool)  # H×W bool

        # ── Step 3: 面积法计算比例尺 ──────────────────────────────────────────
        pixel_area = int(mask.sum())
        real_area_mm2 = COIN_REAL_AREA_MM2[coin_type]
        px2_per_mm2 = pixel_area / real_area_mm2
        px_per_mm = np.sqrt(px2_per_mm2)

        # ── Step 4: 椭圆拟合法（校验） ────────────────────────────────────────
        # 如果检验失败，说明有可能将其他物品检测成了硬币，此时应该要求用户重新拍照或调整硬币位置。
        ellipse = None
        px_per_mm_ellipse = 0.0
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                (cx, cy), (ax1, ax2), angle = ellipse
                major_axis_px = max(ax1, ax2)
                px_per_mm_ellipse = major_axis_px / COIN_DIAMETER_MM[coin_type]

        return CoinSegResult(
            found=True,
            bbox_xyxy=(x1, y1, x2, y2), 
            yolo_conf=yolo_conf,
            mask=mask,
            pixel_area=pixel_area,
            px2_per_mm2=px2_per_mm2,
            px_per_mm=px_per_mm,
            ellipse=ellipse,
            px_per_mm_ellipse=px_per_mm_ellipse,
            coin_type=coin_type,
        )

    def visualize(self, image_input, result: CoinSegResult, save_path: str = None):
        """可视化：原图 | bbox | SAM mask 叠加。"""
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
        else:
            bgr = image_input.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：YOLO bbox
        ax0 = axes[0]
        ax0.imshow(rgb)
        if result.found:
            x1, y1, x2, y2 = result.bbox_xyxy
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax0.add_patch(rect)
            ax0.set_title(f"YOLO Detection  conf={result.yolo_conf:.2f}")
        else:
            ax0.set_title("No coin detected")
        ax0.axis("off")

        # 右图：SAM mask 叠加
        ax1 = axes[1]
        ax1.imshow(rgb)
        if result.found and result.mask is not None:
            overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
            overlay[result.mask] = [1, 1, 0, 0.45]   # 黄色半透明
            ax1.imshow(overlay)

            # 绘制椭圆轮廓
            if result.ellipse is not None:
                (cx, cy), (ax_w, ax_h), angle = result.ellipse
                ellipse_patch = mpatches.Ellipse(
                    (cx, cy), max(ax_w, ax_h), min(ax_w, ax_h),
                    angle=-angle, edgecolor="red", facecolor="none", linewidth=2
                )
                ax1.add_patch(ellipse_patch)

            title = (
                f"SAM Mask  |  pixel area: {result.pixel_area} px\n"
                f"Area method : {result.px_per_mm:.3f} px/mm\n"
                f"Ellipse method: {result.px_per_mm_ellipse:.3f} px/mm"
            )
            ax1.set_title(title, fontsize=9)
        ax1.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[保存] {save_path}")
        else:
            plt.show()
        plt.close()


# ── 便捷函数 ──────────────────────────────────────────────────────────────────

def compute_food_area_mm2(food_pixel_area: int, coin_result: CoinSegResult) -> float:
    """
    利用硬币分割结果将食物像素面积换算为真实面积（mm²）。

    Args:
        food_pixel_area: SAM 分割出的食物 mask 像素数
        coin_result:     CoinSegResult（需 found=True）

    Returns:
        食物真实投影面积（mm²）
    """
    if not coin_result.found or coin_result.px2_per_mm2 == 0:
        raise ValueError("硬币未检测到，无法换算面积")
    return food_pixel_area / coin_result.px2_per_mm2


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--image",   required=True)
    p.add_argument("--yolo",    required=True, help="YOLO 权重路径")
    p.add_argument("--sam",     required=True, help="SAM checkpoint 路径")
    p.add_argument("--save",    default=None,  help="结果图保存路径")
    p.add_argument("--coin",    default="1yuan")
    args = p.parse_args()

    segmentor = CoinSegmentor(
        yolo_weights=args.yolo,
        sam_checkpoint=args.sam,
    )
    result = segmentor.segment(args.image, coin_type=args.coin)

    if result.found:
        print(f"硬币检测成功")
        print(f"  YOLO 置信度     : {result.yolo_conf:.3f}")
        print(f"  像素面积        : {result.pixel_area} px")
        print(f"  比例尺（面积法）: {result.px_per_mm:.4f} px/mm")
        print(f"  比例尺（椭圆法）: {result.px_per_mm_ellipse:.4f} px/mm")
        print(f"  1cm = {result.px_per_mm * 10:.1f} px")
    else:
        print("未检测到硬币")

    segmentor.visualize(args.image, result, save_path=args.save)
