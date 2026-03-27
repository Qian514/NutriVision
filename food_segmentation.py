"""
食物分割模块（SAM 自动模式）

当前功能：用 SAM 自动分割图片中所有区域，可视化结果。
后续扩展：对每个 mask 裁剪 → 分类模型识别食材。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SegmentedRegion:
    mask: np.ndarray          # bool 数组，shape=(H, W)
    bbox: tuple               # (x, y, w, h)
    area: int                 # 像素面积
    crop: np.ndarray = field(default=None, repr=False)  # 裁剪出的小图 (RGB)
    label: str = ""           # 食材名称，由后续分类模型填写


@dataclass
class SegmentationResult:
    regions: list[SegmentedRegion]
    image: np.ndarray = field(default=None, repr=False)  # 原图 (RGB)

    def __len__(self):
        return len(self.regions)


# ---------------------------------------------------------------------------
# SAM 分割器
# ---------------------------------------------------------------------------

class SAMSegmentor:
    """
    使用 SAM 自动模式分割图片中的所有区域。

    权重下载（选一个）：
      vit_b (375MB，最轻量): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
      vit_l (1.2GB):         https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
      vit_h (2.6GB):         https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        min_area: int = 500,       # 过滤掉面积过小的 mask（噪声）
        max_masks: int = 50,       # 最多保留的 mask 数量
    ):
        """
        Args:
            checkpoint: SAM 权重文件路径（.pth）
            model_type: "vit_b" / "vit_l" / "vit_h"
            min_area:   像素面积阈值，小于此值的 mask 会被过滤
            max_masks:  保留面积最大的前 N 个 mask
        """
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM] 使用设备: {device}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)

        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=64,          # 采样点密度，越大越细但越慢
            pred_iou_thresh=0.80,         # 置信度阈值
            stability_score_thresh=0.90,
            min_mask_region_area=min_area,
        )
        self.min_area = min_area
        self.max_masks = max_masks

    def segment(self, image_input: str) -> SegmentationResult:
        """
        Args:
            image_input: 本地图片路径

        Returns:
            SegmentationResult，包含所有分割区域
        """
        image_bgr = cv2.imread(image_input)
        if image_bgr is None:
            raise FileNotFoundError(f"找不到图片: {image_input}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        print(f"[SAM] 开始分割，图片尺寸: {image_rgb.shape[:2]}")
        masks = self.generator.generate(image_rgb)
        print(f"[SAM] 原始 mask 数量: {len(masks)}")

        regions = self._process_masks(masks, image_rgb)
        print(f"[SAM] 过滤后 mask 数量: {len(regions)}")

        return SegmentationResult(regions=regions, image=image_rgb)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _process_masks(self, masks: list[dict], image: np.ndarray) -> list[SegmentedRegion]:
        # 按面积降序排列，取前 max_masks 个
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        masks = masks[: self.max_masks]

        regions = []
        for m in masks:
            if m["area"] < self.min_area:
                continue

            mask = m["segmentation"]  # bool array (H, W)
            x, y, w, h = m["bbox"]    # SAM 输出的 bbox 是 [x, y, w, h]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 裁剪出该区域的小图
            crop = image[y:y+h, x:x+w].copy()
            # 将 mask 之外的像素置为黑色，突出该区域
            mask_crop = mask[y:y+h, x:x+w]
            crop[~mask_crop] = 0

            regions.append(SegmentedRegion(
                mask=mask,
                bbox=(x, y, w, h),
                area=int(m["area"]),
                crop=crop,
            ))

        return regions


# ---------------------------------------------------------------------------
# 可视化工具
# ---------------------------------------------------------------------------

def visualize_segmentation(result: SegmentationResult, save_path: str = None):
    """
    可视化分割结果：
    - 左图：原图叠加彩色 mask
    - 右图：每个 mask 的裁剪小图（最多显示前20个）
    """
    image = result.image
    regions = result.regions

    print(f"[可视化] 共 {len(regions)} 个 mask")

    fig = plt.figure(figsize=(16, 8))

    # --- 左图：叠加所有 mask ---
    ax_main = fig.add_subplot(1, 2, 1)
    ax_main.imshow(image)
    ax_main.set_title(f"Segmentation Result ({len(regions)} regions)", fontsize=13)
    ax_main.axis("off")

    colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        overlay = np.zeros((*image.shape[:2], 4), dtype=float)
        overlay[region.mask] = [*color[:3], 0.45]
        ax_main.imshow(overlay)

        # 在 bbox 中心标注编号
        x, y, w, h = region.bbox
        ax_main.text(
            x + w / 2, y + h / 2, str(i + 1),
            color="white", fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc=color[:3], alpha=0.7),
        )

    # --- 右图：裁剪小图网格 ---
    show_n = min(20, len(regions))
    cols = 5
    rows = (show_n + cols - 1) // cols

    ax_grid = fig.add_subplot(1, 2, 2)
    ax_grid.axis("off")
    ax_grid.set_title(f"Cropped Regions (top {show_n})", fontsize=13)

    for i in range(show_n):
        region = regions[i]
        sub_ax = fig.add_axes([
            0.52 + (i % cols) * 0.096,
            0.08 + (rows - 1 - i // cols) * (0.85 / rows),
            0.088,
            0.75 / rows,
        ])
        sub_ax.imshow(region.crop)
        sub_ax.set_title(f"#{i+1}\n{region.area}px", fontsize=6)
        sub_ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[可视化] 已保存到 {save_path}")
    else:
        plt.show()


def save_crops(result: SegmentationResult, output_dir: str):
    """将所有 mask 裁剪图保存到指定文件夹，文件名含编号和面积，方便调试。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, region in enumerate(result.regions):
        crop_rgb = region.crop
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        filename = out / f"{i+1:03d}_area{region.area}.jpg"
        cv2.imwrite(str(filename), crop_bgr)

    print(f"[保存] {len(result.regions)} 个 mask 已保存到 {output_dir}/")


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CHECKPOINT = "sam_vit_b_01ec64.pth"  # 下载后放到项目目录
    IMAGE = "Data/release_data/test/000030.jpg"

    segmentor = SAMSegmentor(checkpoint=CHECKPOINT, model_type="vit_b")
    result = segmentor.segment(IMAGE)

    print(f"\n共分割出 {len(result)} 个区域")
    for i, r in enumerate(result.regions[:5]):
        print(f"  #{i+1}: 面积={r.area}px  bbox={r.bbox}")

    save_crops(result, "crops_debug")
    visualize_segmentation(result, save_path="segmentation_result.png")

    visualize_segmentation(result, save_path="segmentation_result.png")
