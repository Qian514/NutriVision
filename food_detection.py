"""
食物识别模块

架构：
  FoodDetector (抽象基类)
    ├── LLMFoodDetector   — 当前实现，使用 Qwen-VL 等多模态 LLM
    └── DLFoodDetector    — 预留，后续使用深度学习模型（EfficientNet 等）

统一输出 FoodDetectionResult dataclass。
"""

import os
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FoodDetectionResult:
    dish: str                                # 菜品名称（一道菜）
    ingredients: list[str]                   # 主要食材列表（无占比）(估计)
    cooking_method: str = ""                 # 烹饪方式
    notes: str = ""                          # 备注
    raw_response: str = ""                   # 原始模型输出（调试用）
    detector: str = ""                       # 使用的检测器名称


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class FoodDetector(ABC):
    """所有食物识别器的基类，子类只需实现 detect()。"""

    @abstractmethod
    def detect(self, image_input: str) -> FoodDetectionResult:
        """
        识别图片中的食物。

        Args:
            image_input: 本地图片路径 或 图片 URL

        Returns:
            FoodDetectionResult
        """

    def _is_url(self, path: str) -> bool:
        return path.startswith("http://") or path.startswith("https://")

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """返回 (base64字符串, 图片格式)"""
        ext = Path(image_path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, ext


# ---------------------------------------------------------------------------
# LLM 实现（当前使用）
# ---------------------------------------------------------------------------

class LLMFoodDetector(FoodDetector):
    """
    使用多模态 LLM 进行食物识别。
    默认接入 Qwen-VL（通过 DashScope OpenAI 兼容接口）。
    通过修改 base_url 和 model 也可切换到其他兼容 OpenAI 格式的视觉模型。
    """

    DEFAULT_PROMPT = """请分析这张餐食图片，严格按照以下 JSON 格式回答，不要输出任何其他内容：

{
  "dish": "菜品名称，若无法确定则填空字符串",
  "ingredients": ["可见食材1", "可见食材2"],
  "cooking_method": "烹饪方式，若无法判断则填空字符串",
  "notes": "见下方说明"
}

【识别规则】
- 只识别图中的一道菜。
- ingredients 填写图中可见的主要食材，无法判断时填空数组。
- cooking_method 根据食物外观推断（如：炒、炸、蒸、煮、红烧、凉拌等）。

【notes 填写规则】
- 如果能确认菜品名称：简述该菜的典型特点，例如主要食材比例、常见调料、口味特征。
- 如果无法确认菜品名称（dish 为空）：详细描述食物的外观特征，包括：
    1. 主要颜色与质地（如：深红色、油亮、有汤汁）
    2. 可辨认的食材形状（如：块状肉、叶状蔬菜、颗粒状豆类）
    3. 推测的烹饪痕迹（如：表面焦化、裹有芡汁、撒有芝麻）
  这些描述将用于后续营养推理，请尽量具体。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-vl-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        """
        Args:
            api_key:  API Key，默认读取环境变量 DASHSCOPE_API_KEY
            model:    模型名，可换为 qwen-vl-plus 节省费用
            base_url: 接口地址，切换其他厂商时修改此项
        """
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ["DASHSCOPE_API_KEY"],
            base_url=base_url,
        )

    def detect(self, image_input: str) -> FoodDetectionResult:
        image_content = self._build_image_content(image_input)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": self.DEFAULT_PROMPT},
                    ],
                }
            ],
        )

        raw = response.choices[0].message.content
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_image_content(self, image_input: str) -> dict:
        if self._is_url(image_input):
            return {"type": "image_url", "image_url": {"url": image_input}}
        b64, ext = self._encode_image(image_input)
        return {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}

    def _parse_response(self, raw: str) -> FoodDetectionResult:
        import json
        import re

        # 提取 JSON 块（防止模型在 JSON 前后多输出文字）
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return FoodDetectionResult(
                dish="", ingredients=[], raw_response=raw, detector=self.model
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return FoodDetectionResult(
                dish="", ingredients=[], raw_response=raw, detector=self.model
            )

        return FoodDetectionResult(
            dish=data.get("dish", ""),
            ingredients=data.get("ingredients", []),
            cooking_method=data.get("cooking_method", ""),
            notes=data.get("notes", ""),
            raw_response=raw,
            detector=self.model,
        )


# ---------------------------------------------------------------------------
# 深度学习实现（预留）
# ---------------------------------------------------------------------------

class DLFoodDetector(FoodDetector):
    """
    使用本地深度学习模型进行食物识别（待实现）。
    预期：加载微调后的 EfficientNet / ResNet 权重，在本地推理。
    """

    def __init__(self, model_path: str, class_names: list[str]):
        """
        Args:
            model_path:  本地权重文件路径（.pth）
            class_names: 类别名称列表，顺序与训练时一致
        """
        raise NotImplementedError("DLFoodDetector 尚未实现，敬请期待。")

    def detect(self, image_input: str) -> FoodDetectionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 快捷工厂函数
# ---------------------------------------------------------------------------

def create_detector(backend: str = "llm", **kwargs) -> FoodDetector:
    """
    Args:
        backend: "llm" 或 "dl"
        **kwargs: 传递给对应检测器的构造参数
    """
    if backend == "llm":
        return LLMFoodDetector(**kwargs)
    if backend == "dl":
        return DLFoodDetector(**kwargs)
    raise ValueError(f"未知 backend: {backend}，可选 'llm' 或 'dl'")


# ---------------------------------------------------------------------------
# 简单测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = create_detector("llm")
    result = detector.detect("test_gong.png")

    print(f"菜品：{result.dish}")
    print(f"食材：{result.ingredients}")
    print(f"烹饪方式：{result.cooking_method}")
    print(f"备注：{result.notes}")
