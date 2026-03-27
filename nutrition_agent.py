"""
营养分析智能体

接收 PipelineResult，输出餐食营养估算报告。

流程：
  1. 面积 → 重量（本地密度表）
  2. 查询食材营养（本地 nutrition_db → Tavily 联网）
  3. 查询菜谱比例（本地 recipe_db  → Tavily 联网 → 自主推理）
  4. 汇总计算营养
"""

import json
import os
from pathlib import Path
import requests

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

BOCHA_KEY = os.environ.get("BOCHA_KEY", "")

# ---------------------------------------------------------------------------
# 本地知识库路径
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).parent / "data"
_NUTRITION_DB: dict = json.loads((_DB_DIR / "nutrition_db.json").read_text(encoding="utf-8"))
_RECIPE_DB: dict    = json.loads((_DB_DIR / "recipe_db.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 博查搜索
# ---------------------------------------------------------------------------
def bocha_search(query: str) -> str:
    """使用博查搜索获取查询结果的纯文本摘要（失败时返回空字符串）。"""
    try:
        BOCHA_URL = "https://api.bocha.cn/v1/web-search"
        headers = {
            "Authorization": f"Bearer {BOCHA_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "summary": True,
            "count": 3,
        }
        results = requests.post(BOCHA_URL, headers=headers, json=payload).json()
        return results
    except Exception as e:
        print(f"  [博查搜索] 搜索 {query} 时出错：{e}")
        return ""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_recipe_db(dish: str) -> str:
    """
    查询菜品的食材构成比例、估算厚度和密度信息。
    优先查本地数据库，未找到时联网搜索，仍未找到则返回空让智能体自主推理。
    返回 JSON 字符串。
    """
    # 本地查询
    for key, val in _RECIPE_DB.items():
        if key in dish or dish in key:
            print(f"  [食谱库] 📚 本地找到 {key}<->{dish}，返回食谱数据")
            return json.dumps({"source": "local", "dish": key, **val}, ensure_ascii=False)

    # 联网查询
    try:
        # from langchain_community.tools.tavily_search import TavilySearchResults
        # tavily = TavilySearchResults(max_results=3)
        # results = tavily.invoke(f"{dish} 食谱 食材比例 做法")
        results = bocha_search(f"{dish} 食谱 食材比例 做法")
        print(f"  [食谱库] 🌐 联网搜索 {dish} 食谱数据，返回原始结果")
        return json.dumps({"source": "web", "dish": dish, "raw": results}, ensure_ascii=False)
    except Exception as e:
        print(f"  [食谱库] 😭 联网搜索 {dish} 食谱数据时出错：{e}")
        return json.dumps({"source": "not_found", "dish": dish, "error": str(e)}, ensure_ascii=False)


@tool
def calculate_nutrition(total_area_mm2: float, ingredients_ratios: str, thickness_mm: float) -> str:
    """
    根据食物总投影面积、食材比例和估算厚度，计算各食材重量及营养汇总。

    参数：
      total_area_mm2:     食物总投影面积（mm²），来自 pipeline
      ingredients_ratios: JSON 字符串，格式 {"食材名": 占比(0~1), ...}，比例之和应为 1
      thickness_mm:       食物估算厚度（mm），从 search_recipe_db 获取，未知时传 15

    计算流程（逐食材）：
      1. 食材投影面积 = total_area_mm2 × 比例  (mm²)
      2. 食材体积     = 食材面积(cm²) × 厚度(cm)  (cm³)
      3. 食材重量     = 体积 × 密度(g/cm³)  (g)   ← 密度来自本地 nutrition_db
      4. 营养值       = 重量 / 100 × 每100g营养

    返回各食材明细及总营养 JSON。
    """
    try:
        ratios: dict = json.loads(ingredients_ratios)
    except json.JSONDecodeError:
        return json.dumps({"error": "ingredients_ratios 格式错误，请传入 JSON 字符串"}, ensure_ascii=False)

    area_cm2_total = total_area_mm2 / 100.0
    thickness_cm = thickness_mm / 10.0
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    details = []

    for ingredient, ratio in ratios.items():
        # Step 1-2：面积 → 体积
        ingredient_area_cm2 = area_cm2_total * ratio
        ingredient_volume_cm3 = ingredient_area_cm2 * thickness_cm

        # Step 3：查密度，计算重量
        nutrition = None
        for key, val in _NUTRITION_DB.items():
            if key in ingredient or ingredient in key:
                nutrition = val
                print(f"  [营养库] 📚 本地找到 {key}<->{ingredient}，密度 {nutrition.get('density', '未知')} g/cm³") # 这里是假设 nutrition_db 中有密度字段，实际可能没有，需要根据食材类型估算或联网查询
                break

        if nutrition is None:
            # 本地未找到 → 联网补充
            web_raw = bocha_search(f"{ingredient} 每100克 营养成分 热量 蛋白质 脂肪 碳水化合物 密度")
            print(f"  [营养库] 🌐 联网搜索 {ingredient} 营养数据，返回原始结果")
            details.append({
                "ingredient": ingredient,
                "ratio": ratio,
                "area_cm2": round(ingredient_area_cm2, 2),
                "note": "本地无数据，需从联网结果提取",
                "web_raw": web_raw,
            })
            continue

        density = nutrition.get("density", 1.0)
        weight_g = ingredient_volume_cm3 * density

        # Step 4：计算营养
        factor = weight_g / 100.0
        row = {
            "ingredient": ingredient,
            "ratio": ratio,
            "area_cm2": round(ingredient_area_cm2, 2),
            "volume_cm3": round(ingredient_volume_cm3, 2),
            "density": density,
            "weight_g": round(weight_g, 1),
            "calories": round(nutrition["calories"] * factor, 1),
            "protein":  round(nutrition["protein"]  * factor, 1),
            "fat":      round(nutrition["fat"]       * factor, 1),
            "carbs":    round(nutrition["carbs"]     * factor, 1),
        }
        for k in ["calories", "protein", "fat", "carbs"]:
            total[k] += row[k]
        details.append(row)

    print(f"  [营养计算] 完成，总热量 {total['calories']:.1f} kcal")
    return json.dumps({
        "details": details,
        "total": {k: round(v, 1) for k, v in total.items()},
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 输出结构定义
# ---------------------------------------------------------------------------

class IngredientNutrition(BaseModel):
    name:     str   = Field(description="食材名称")
    weight_g: float = Field(description="估算重量（克）")
    calories: float = Field(description="热量（kcal）")
    protein:  float = Field(description="蛋白质（克）")
    fat:      float = Field(description="脂肪（克）")
    carbs:    float = Field(description="碳水化合物（克）")


class TotalNutrition(BaseModel):
    calories: float = Field(description="总热量（kcal）")
    protein:  float = Field(description="总蛋白质（克）")
    fat:      float = Field(description="总脂肪（克）")
    carbs:    float = Field(description="总碳水化合物（克）")


class NutritionReport(BaseModel):
    dish:            str                      = Field(description="菜品名称，无法识别时填推理结果")
    total_weight_g:  float                    = Field(description="估算总重量（克）")
    ingredients:     list[IngredientNutrition]= Field(description="各食材明细")
    total_nutrition: TotalNutrition           = Field(description="营养汇总")
    advice:          str                      = Field(description="简短饮食建议（1-2句）")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名专业的餐食营养分析师。
用户会提供菜品信息和食物投影面积，你需要分析这道菜的营养成分。

分析步骤：
1. 调用 search_recipe_db 获取食材构成比例和厚度
   - 本地找到：直接使用 ingredients_ratio 和 thickness_mm
   - 未找到（返回联网结果）：从中提取食材比例，thickness_mm 使用默认值 15
   - 完全找不到：根据菜品类型和可见食材自主推理比例，thickness_mm 使用默认值 15
2. 调用 calculate_nutrition，传入：
   - total_area_mm2：用户提供的食物总投影面积
   - ingredients_ratios：上一步得到的食材比例（JSON，忽略盐糖等微量调料，比例重新归一化至和为1）
   - thickness_mm：上一步得到的厚度
3. 若 calculate_nutrition 返回某食材"本地无数据"，从 web_raw 中提取该食材的密度和营养数据，
   结合其 area_cm2 和 thickness_mm 手动补算该食材的重量和营养，并纳入总计
4. 按照要求的结构输出最终结果

注意：如果菜品名称未识别（为空），请先根据外观描述推理菜品类型填入 dish 字段，再按上述步骤分析。"""


def build_agent():
    """构建营养分析智能体（LangChain 1.x）。"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )

    tools = [search_recipe_db, calculate_nutrition]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=NutritionReport,
    )


def analyze(pipeline_result) -> NutritionReport:
    """
    接收 PipelineResult，返回结构化营养分析报告。

    Args:
        pipeline_result: pipeline.PipelineResult 实例

    Returns:
        NutritionReport Pydantic 对象
    """
    dish = pipeline_result.dish or ""
    notes = getattr(pipeline_result, "notes", "")

    if dish:
        dish_desc = f"菜品名称：{dish}"
    else:
        dish_desc = f"菜品名称：未识别\n外观描述：{notes}"

    input_text = f"""{dish_desc}
食材列表：{', '.join(pipeline_result.ingredients) or '未知'}
烹饪方式：{pipeline_result.cooking_method or '未知'}
食物总投影面积：{pipeline_result.total_food_area_mm2:.1f} mm²（{pipeline_result.total_food_area_mm2/100:.1f} cm²）"""

    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": input_text}]})
    return result["structured_response"]


# ---------------------------------------------------------------------------
# 独立测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 模拟一个 PipelineResult 进行测试
    from dataclasses import dataclass, field

    @dataclass
    class MockPipelineResult:
        dish: str = "红烧肉"
        ingredients: list = field(default_factory=lambda: ["猪五花", "酱油", "糖","玉米"])
        cooking_method: str = "红烧"
        notes: str = ""
        total_food_area_mm2: float = 8000.0

    report = analyze(MockPipelineResult())
    print("\n" + "="*50)
    print("营养分析报告（结构化）")
    print("="*50)
    print(report.model_dump_json(indent=2, ensure_ascii=False))

