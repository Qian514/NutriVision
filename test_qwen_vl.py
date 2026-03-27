import os
import base64
from pathlib import Path
from openai import OpenAI  
from dotenv import load_dotenv

load_dotenv()


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_food(image_path: str, api_key: str) -> dict:
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 判断输入是本地路径还是URL
    if image_path.startswith("http"):
        image_content = {"type": "image_url", "image_url": {"url": image_path}}
    else:
        ext = Path(image_path).suffix.lstrip(".")
        b64 = encode_image(image_path)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
        }

    prompt = """请分析这张餐食图片，按以下格式回答：

1. 菜品名称：（识别出的菜品，可以多个）
2. 主要食材：（列出主要食材及估计占比）
3. 烹饪方式：（炒、蒸、炸等）
4. 大致份量：（少/中/多）
5. 备注：（其他观察，如颜色、摆盘等）

请尽量准确，如果无法确定请注明。"""

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [image_content, {"type": "text", "text": prompt}],
            }
        ],
    )

    result_text = response.choices[0].message.content
    return {
        "raw": result_text,
        "model": response.model,
        "tokens": response.usage.total_tokens,
    }


if __name__ == "__main__":
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your_api_key_here")

    TEST_IMAGE = "test_gong.png"  

    print("正在分析餐食图片...\n")
    result = analyze_food(TEST_IMAGE, API_KEY)

    print("=" * 50)
    print(result["raw"])
    print("=" * 50)
    print(f"模型: {result['model']}  |  消耗tokens: {result['tokens']}")
