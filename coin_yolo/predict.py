"""
硬币检测推理模块。

用法：
  # 单张图片，显示结果
  python predict.py --image path/to/img.jpg --weights runs/coin_v1/weights/best.pt

  # 作为模块导入
  from coin_yolo.predict import CoinPredictor
  predictor = CoinPredictor("runs/coin_v1/weights/best.pt")
  result = predictor.detect("image.jpg")
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config


@dataclass
class CoinPrediction:
    found: bool
    bbox_xyxy: tuple = None    # (x1, y1, x2, y2) 像素坐标
    confidence: float = 0.0
    px_per_mm: float = 0.0     # 如果检测到硬币，计算比例尺
    coin_diameter_mm: float = 25.0


class CoinPredictor:
    """
    使用训练好的 YOLO 权重检测图片中的硬币，并计算 px/mm 比例尺。
    """

    def __init__(self, weights: str, conf_thresh: float = 0.25):
        """
        Args:
            weights:     训练好的 .pt 权重路径
            conf_thresh: 置信度阈值，低于此值的检测结果忽略
        """
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_thresh = conf_thresh

    def detect(self, image_input, coin_diameter_mm: float = 25.0) -> CoinPrediction:
        """
        Args:
            image_input:       图片路径（str/Path）或 BGR numpy 数组
            coin_diameter_mm:  真实硬币直径（mm），1元 = 25mm

        Returns:
            CoinPrediction
        """
        results = self.model(image_input, conf=self.conf_thresh, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return CoinPrediction(found=False)

        # 取置信度最高的检测框
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])

        # 用检测框的短边近似硬币直径（避免倾斜影响）
        diameter_px = min(x2 - x1, y2 - y1)
        px_per_mm = diameter_px / coin_diameter_mm

        return CoinPrediction(
            found=True,
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=conf,
            px_per_mm=px_per_mm,
            coin_diameter_mm=coin_diameter_mm,
        )

    def visualize(self, image_input, prediction: CoinPrediction, save_path: str = None):
        """在图片上绘制检测结果并显示/保存。"""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        else:
            img = image_input.copy()

        if prediction.found:
            x1, y1, x2, y2 = [int(v) for v in prediction.bbox_xyxy]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"coin {prediction.confidence:.2f} | {prediction.px_per_mm:.2f}px/mm"
            cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No coin detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"[保存] {save_path}")
        else:
            cv2.imshow("Coin Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image",   required=True, help="输入图片路径")
    p.add_argument("--weights", required=True, help="YOLO 权重路径 (.pt)")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--save",    type=str, default=None, help="保存结果图片路径")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictor = CoinPredictor(args.weights, conf_thresh=args.conf)
    result = predictor.detect(args.image)

    if result.found:
        print(f"检测到硬币")
        print(f"  bbox    : {[round(v) for v in result.bbox_xyxy]}")
        print(f"  置信度  : {result.confidence:.3f}")
        print(f"  比例尺  : {result.px_per_mm:.3f} px/mm")
        print(f"  1cm = {result.px_per_mm * 10:.1f} px")
    else:
        print("未检测到硬币")

    predictor.visualize(args.image, result, save_path=args.save)
