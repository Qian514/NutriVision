"""
交互式 SAM 分割工具

用法：
  - 按 m：切换模式（点击 / 框选）
  - 【点击模式】左键点击食物区域 → 生成 mask
  - 【框选模式】左键拖拽框选区域 → 松开后生成 mask
  - 右键：撤销上一个 mask
  - 按 s：保存所有 mask 到 crops_debug/
  - 按 q / 关闭窗口：退出
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


class InteractiveSegmentor:
    def __init__(self, checkpoint: str, model_type: str = "vit_b"):
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM] 使用设备: {device}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        self.regions = []
        self.image = None
        self.mode = "box"          # "point" 或 "box"

        # 框选状态
        self._press_xy = None      # 鼠标按下的坐标
        self._rect_patch = None    # 正在绘制的矩形

    def run(self, image_path: str):
        image_bgr = cv2.imread(image_path) # 读图片，OpenCV 默认是 BGR 格式
        if image_bgr is None:
            raise FileNotFoundError(f"找不到图片: {image_path}")
        self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 

        print("[SAM] 正在编码图片...")
        self.predictor.set_image(self.image) # SAM 需要先编码图片，后续交互才会流畅
        print("[SAM] 就绪，左键拖拽框选食物区域")

        # 设置 Matplotlib 交互式界面
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        self._redraw()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def _on_press(self, event): # 点击事件（点击模式直接分割，框选模式记录起点）
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 1:
            if self.mode == "point":  # 点击模式：直接生成 mask
                x, y = int(event.xdata), int(event.ydata)
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best = masks[np.argmax(scores)]
                self.regions.append({"mask": best, "point": (x, y)})
                self._redraw()
                print(f"  [+] 点击 ({x},{y})，面积 {best.sum()} px，共 {len(self.regions)} 个 mask")
            else:  # 框选模式：记录起点
                self._press_xy = (event.xdata, event.ydata)

        elif event.button == 3:  # 右键：撤销
            if self.regions:
                self.regions.pop()
                self._redraw()
                print(f"  [-] 已撤销，当前共 {len(self.regions)} 个 mask")

    def _on_drag(self, event):
        if self.mode == "point" or self._press_xy is None or event.inaxes != self.ax or event.xdata is None:
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        # 实时更新矩形预览
        if self._rect_patch:
            self._rect_patch.remove()
        self._rect_patch = patches.Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0), abs(y1 - y0),
            linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--",
        )
        self.ax.add_patch(self._rect_patch)
        self.fig.canvas.draw_idle()

    def _on_release(self, event): # 松开鼠标，完成框选，作为 SAM 输入生成 mask
        if self._press_xy is None or event.button != 1:
            return
        if event.xdata is None:
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None

        # 清除预览矩形
        if self._rect_patch:
            self._rect_patch.remove()
            self._rect_patch = None

        # 框太小则忽略
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.fig.canvas.draw_idle()
            return

        box = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        self.regions.append({"mask": best, "box": box})
        self._redraw()
        print(f"  [+] 框选 {box.astype(int).tolist()}，面积 {best.sum()} px，当前共 {len(self.regions)} 个 mask")

    def _on_key(self, event):
        if event.key == "m":
            self.mode = "point" if self.mode == "box" else "box"
            self._redraw()
            print(f"  [模式] 已切换为 {'点击' if self.mode == 'point' else '框选'} 模式")
        elif event.key == "s":
            self._save_crops()
        elif event.key == "q":
            plt.close()

    def _redraw(self):
        self.ax.cla()
        self.ax.imshow(self.image)
        mode_hint = "Click to segment" if self.mode == "point" else "Drag to segment"
        self.ax.set_title(
            f"[M] Mode: {'POINT' if self.mode == 'point' else 'BOX'} | "
            f"{mode_hint} | Right click: undo | S: save | Q: quit  "
            f"[{len(self.regions)} masks]"
        )
        self.ax.axis("off")

        for i, region in enumerate(self.regions):
            color = self.colors[i % len(self.colors)]
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=float)
            overlay[region["mask"]] = [*color[:3], 0.45]
            self.ax.imshow(overlay)

            if "box" in region:
                x1, y1, x2, y2 = region["box"]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=color[:3], facecolor="none",
                )
                self.ax.add_patch(rect)
                self.ax.text(x1 + 3, y1 + 12, str(i + 1), color="white", fontsize=8,
                             bbox=dict(fc=color[:3], pad=1, alpha=0.7))
            else:
                px, py = region["point"]
                self.ax.plot(px, py, "o", color=color[:3], markersize=7)
                self.ax.text(px + 6, py, str(i + 1), color="white", fontsize=8,
                             bbox=dict(fc=color[:3], pad=1, alpha=0.7))

        self.fig.canvas.draw()

    def _save_crops(self):
        if not self.regions:
            print("[保存] 没有 mask，请先框选")
            return

        out = Path("crops_debug")
        out.mkdir(exist_ok=True)

        for i, region in enumerate(self.regions):
            mask = region["mask"]
            ys, xs = np.where(mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()

            crop = self.image[y1:y2, x1:x2].copy()
            mask_crop = mask[y1:y2, x1:x2]
            crop[~mask_crop] = 0

            area = int(mask.sum())
            path = out / f"{i+1:03d}_area{area}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        print(f"[保存] {len(self.regions)} 个 mask 已保存到 crops_debug/")


if __name__ == "__main__":
    CHECKPOINT = "sam_vit_b_01ec64.pth"
    IMAGE = "test_gong.png"

    seg = InteractiveSegmentor(checkpoint=CHECKPOINT, model_type="vit_b")
    seg.run(IMAGE)
