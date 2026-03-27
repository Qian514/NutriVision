"""
餐食营养分析主流程

Pipeline 步骤：
  1. 食物识别  — Qwen-VL 识别菜名 + 食材列表
  2. 硬币分割  — YOLO 检测 + SAM 精割 → px²/mm² 比例尺
  3. 食物分割  — 交互式 SAM，用户手动框选食物区域（排除盘子）
  4. 区域过滤  — 排除与硬币重叠的区域
  5. 面积换算  — 各食物区域像素面积 → 真实投影面积（mm²）

  （后续）
  6. 智能体推理 — 根据菜名 + 食材 + 面积 → 估算重量 → 查营养库

输出：PipelineResult

运行：
python pipeline.py \
  --image Data/test_coin.jpg \
  --sam weights/sam_vit_b_01ec64.pth \
  --yolo weights/best.pt \
  --save-img out/pipline_result.jpg \
  --save-json out/result.json \
  -- no-llm  # 可选，跳过 LLM 食物识别，节省 API 费用
"""

import sys
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.rcParams["font.family"] = ["STHeiti"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field

from food_detection import create_detector, FoodDetectionResult
from interactive_segment import InteractiveSegmentor
from coin_segmentation import CoinSegmentor, CoinSegResult, compute_food_area_mm2


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FoodRegion:
    """一个食材分割区域，附带真实面积。"""
    label: str                          # 食材名（暂时为空，待分类模型填写）
    mask: np.ndarray = field(repr=False)
    bbox: tuple = None                  # (x, y, w, h)
    pixel_area: int = 0
    real_area_mm2: float = 0.0          # 换算后的真实投影面积


@dataclass
class PipelineResult:
    image_path: str
    # ── Step 1 ─────────────────────────────────────────────────────────────
    dish: str = ""
    ingredients: list[str] = field(default_factory=list)
    cooking_method: str = ""
    # ── Step 2 ─────────────────────────────────────────────────────────────
    coin: CoinSegResult = None
    # ── Step 3-5 ───────────────────────────────────────────────────────────
    food_regions: list[FoodRegion] = field(default_factory=list)
    # ── 汇总 ───────────────────────────────────────────────────────────────
    total_food_area_mm2: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"图片      : {self.image_path}",
            f"菜品      : {self.dish}",
            f"食材      : {', '.join(self.ingredients) or '未识别'}",
            f"烹饪方式  : {self.cooking_method}",
            f"硬币检测  : {'成功' if self.coin and self.coin.found else '失败'}",
        ]
        if self.coin and self.coin.found:
            lines.append(f"  比例尺  : {self.coin.px_per_mm:.3f} px/mm")
            lines.append(f"  面积法  : {self.coin.px2_per_mm2:.3f} px²/mm²")
        lines.append(f"食物区域数: {len(self.food_regions)}")
        lines.append(f"总投影面积: {self.total_food_area_mm2:.1f} mm²  "
                     f"({self.total_food_area_mm2/100:.1f} cm²)")
        lines.append("="*50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

class NutritionPipeline:
    """
    Args:
        sam_checkpoint:  SAM 权重路径（.pth）
        yolo_weights:    硬币 YOLO 权重路径（.pt）
        sam_model_type:  "vit_b" / "vit_l" / "vit_h"
        coin_iou_thresh: 与硬币 mask 的 IoU 超过此值则视为硬币区域，从食物中排除
        min_food_area:   食物区域最小像素面积（过滤噪声）
        use_llm:         是否调用 Qwen-VL 进行食物识别（False 时跳过，节省 API 费用）
    """

    def __init__(
        self,
        sam_checkpoint: str,
        yolo_weights: str,
        sam_model_type: str = "vit_b",
        coin_iou_thresh: float = 0.3, # 与硬币 mask 的 IoU 超过此值则视为硬币区域，从食物中排除
        min_food_area: int = 500, # 食物区域最小像素面积（过滤噪声）
        use_llm: bool = True, # 是否调用 Qwen-VL 进行食物识别（False 时跳过，节省 API 费用）
    ):
        self.coin_iou_thresh = coin_iou_thresh
        self.min_food_area = min_food_area
        self.use_llm = use_llm

        print("[Pipeline] 加载模型...")

        # 食物识别（LLM，可选）
        if use_llm:
            self.food_detector = create_detector("llm") # 参数表示使用 LLM 模型，还有深度学习模型（未实现）

        # 硬币分割（YOLO + SAM）
        self.coin_segmentor = CoinSegmentor(
            yolo_weights=yolo_weights,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
        )

        # 食物分割（交互式 SAM）
        self.food_segmentor = InteractiveSegmentor(
            checkpoint=sam_checkpoint,
            model_type=sam_model_type,
        )

        print("[Pipeline] 模型加载完成")

    def run(self, image_path: str) -> PipelineResult:
        """
        Args:
            image_path: 输入图片路径（图中应放有硬币作为参照）

        Returns:
            PipelineResult
        """
        result = PipelineResult(image_path=image_path)
        t0 = time.time()

        # ── Step 1: 食物识别 ──────────────────────────────────────────────
        print(f"\n[Step 1] 食物识别...")
        if self.use_llm:
            detection: FoodDetectionResult = self.food_detector.detect(image_path)
            result.dish = detection.dish
            result.ingredients = detection.ingredients
            result.cooking_method = detection.cooking_method
            print(f"  菜品: {result.dish}")
            print(f"  食材: {result.ingredients}")
        else:
            result.dish = "（跳过 LLM）"
            result.ingredients = "（跳过 LLM）"
            result.cooking_method = "（跳过 LLM）"
            print("  （已跳过 LLM，dish/ingredients 为空）")

        # ── Step 2: 硬币检测与分割 ────────────────────────────────────────
        print(f"\n[Step 2] 硬币检测与分割...")
        coin: CoinSegResult = self.coin_segmentor.segment(image_path)
        result.coin = coin

        if coin.found:
            print(f"  比例尺（面积法）: {coin.px_per_mm:.3f} px/mm")
            print(f"  比例尺（椭圆法）: {coin.px_per_mm_ellipse:.3f} px/mm")
            # 两种方法差距超过 15% 时给出警告
            if coin.px_per_mm_ellipse > 0:
                diff = abs(coin.px_per_mm - coin.px_per_mm_ellipse) / coin.px_per_mm
                if diff > 0.15:
                    print(f"  ⚠️  两种比例尺差异 {diff*100:.1f}%，硬币分割可能不准")
        else:
            print("  ⚠️  未检测到硬币，面积换算将跳过")

        # ── Step 3: 食物分割（交互式）──────────────────────────────────────
        print(f"\n[Step 3] 食物分割（交互式 SAM）...")
        print("  请在弹出窗口中框选食物区域（排除盘子），完成后按 Q 关闭")
        self.food_segmentor.run(image_path)
        raw_regions = self.food_segmentor.regions
        print(f"  共标注 {len(raw_regions)} 个区域")

        # ── Step 4: 排除硬币区域 ──────────────────────────────────────────
        print(f"\n[Step 4] 过滤硬币区域...")
        food_regions_raw = self._filter_coin_regions(raw_regions, coin)
        print(f"  过滤后剩余 {len(food_regions_raw)} 个食物区域")

        # ── Step 5: 面积换算 ──────────────────────────────────────────────
        print(f"\n[Step 5] 面积换算...")
        food_regions = []
        for region in food_regions_raw:
            mask = region["mask"]
            pixel_area = int(mask.sum())
            bbox = self._mask_to_bbox(mask)
            real_area = compute_food_area_mm2(pixel_area, coin) if coin.found else 0.0
            food_regions.append(FoodRegion(
                label="",
                mask=mask,
                bbox=bbox,
                pixel_area=pixel_area,
                real_area_mm2=real_area,
            ))

        result.food_regions = food_regions
        result.total_food_area_mm2 = sum(r.real_area_mm2 for r in food_regions)

        elapsed = time.time() - t0
        print(f"\n[Pipeline] 完成，耗时 {elapsed:.1f}s")
        print(result.summary())

        return result

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> tuple:
        """从 mask 计算 (x, y, w, h) 边界框。"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    def _filter_coin_regions(
        self,
        regions: list[dict],
        coin: CoinSegResult,
    ) -> list[dict]:
        """
        移除与硬币 mask 高度重叠的区域。
        若硬币未检测到，则返回全部区域。
        """
        if not coin.found or coin.mask is None:
            return regions

        coin_mask = coin.mask
        coin_area = coin_mask.sum()

        filtered = []
        for region in regions:
            mask = region["mask"]
            intersection = (mask & coin_mask).sum()
            iou = intersection / coin_area if coin_area > 0 else 0
            if iou > self.coin_iou_thresh:
                print(f"    排除区域 area={mask.sum()}px（与硬币 IoU={iou:.2f}）")
                continue
            filtered.append(region)

        return filtered


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize_pipeline(result: PipelineResult, save_path: str = None):
    """
    可视化整体 pipeline 结果：
    - 原图 + 硬币 mask（黄色）+ 食物 mask（彩色）
    """
    bgr = cv2.imread(result.image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)

    # 硬币区域（黄色高亮）
    if result.coin and result.coin.found and result.coin.mask is not None:
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[result.coin.mask] = [1, 1, 0, 0.5]
        ax.imshow(overlay)
        # 绘制 YOLO bbox
        x1, y1, x2, y2 = result.coin.bbox_xyxy
        rect = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

    # 食物区域（彩色）
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(result.food_regions), 1)))
    for i, region in enumerate(result.food_regions):
        color = colors[i % len(colors)]
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[region.mask] = [*color[:3], 0.4]
        ax.imshow(overlay)

        # 标注面积
        x, y, w, h = region.bbox
        label = f"{region.real_area_mm2:.0f}mm²" if region.real_area_mm2 > 0 else f"{region.pixel_area}px"
        ax.text(
            x + w/2, y + h/2, label,
            color="white", fontsize=6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc=color[:3], alpha=0.7),
        )

    title_lines = [
        f"Dish: {result.dish or 'N/A'}  |  "
        f"Scale: {result.coin.px_per_mm:.3f} px/mm" if (result.coin and result.coin.found) else "Dish: N/A  |  No coin",
        f"Food regions: {len(result.food_regions)}  |  "
        f"Total area: {result.total_food_area_mm2/100:.1f} cm²",
    ]
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.axis("off")

    # 图例
    patches = [mpatches.Patch(color="yellow", alpha=0.7, label="Coin")]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[可视化] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def save_result_json(result: PipelineResult, save_path: str):
    """将结构化结果保存为 JSON（不含 numpy 数组）。"""
    data = {
        "image_path": result.image_path,
        "dish": result.dish,
        "ingredients": result.ingredients,
        "cooking_method": result.cooking_method,
        "coin": {
            "found": result.coin.found if result.coin else False,
            "px_per_mm": result.coin.px_per_mm if (result.coin and result.coin.found) else 0,
            "px2_per_mm2": result.coin.px2_per_mm2 if (result.coin and result.coin.found) else 0,
            "pixel_area": result.coin.pixel_area if (result.coin and result.coin.found) else 0,
        },
        "food_regions": [
            {
                "label": r.label,
                "pixel_area": r.pixel_area,
                "real_area_mm2": round(r.real_area_mm2, 2),
                "bbox": r.bbox,
            }
            for r in result.food_regions
        ],
        "total_food_area_mm2": round(result.total_food_area_mm2, 2),
        "total_food_area_cm2": round(result.total_food_area_mm2 / 100, 2),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 已保存: {save_path}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="餐食营养分析 Pipeline")
    p.add_argument("--image",   required=True,             help="输入图片路径")
    p.add_argument("--sam",     required=True,             help="SAM 权重路径 (.pth)")
    p.add_argument("--yolo",    required=True,             help="硬币 YOLO 权重路径 (.pt)")
    p.add_argument("--no-llm",  action="store_true",       help="跳过 Qwen-VL 食物识别（省钱调试用）")
    p.add_argument("--save-img",default="pipeline_result.jpg", help="结果图保存路径")
    p.add_argument("--save-json",default="pipeline_result.json", help="JSON 结果保存路径")
    args = p.parse_args()

    pipeline = NutritionPipeline(
        sam_checkpoint=args.sam,
        yolo_weights=args.yolo,
        use_llm=not args.no_llm,
    )

    result = pipeline.run(args.image)
    visualize_pipeline(result, save_path=args.save_img)
    save_result_json(result, save_path=args.save_json)
