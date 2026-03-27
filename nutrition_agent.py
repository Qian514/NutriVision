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

# 常见别名表（双向）：键和值互为别名，查询时自动扩展
_ALIASES: dict[str, list[str]] = {
    "番茄":   ["西红柿"],
    "西红柿": ["番茄"],
    "土豆":   ["马铃薯"],
    "马铃薯": ["土豆"],
    "香菜":   ["芫荽"],
    "芫荽":   ["香菜"],
    "玉米":   ["苞米", "玉蜀黍"],
    "花生":   ["落花生"],
    "落花生": ["花生"],
    "黑木耳": ["木耳"],
    "木耳":   ["黑木耳"],
    "大白菜": ["白菜"],
    "白菜":   ["大白菜"],
    "青椒":   ["甜椒", "灯笼椒"],
    "甜椒":   ["青椒", "灯笼椒"],
    "洋葱":   ["葱头"],
    "葱头":   ["洋葱"],
    "豆腐脑": ["老豆腐"],
    "老豆腐": ["豆腐脑"],
    "黄豆":   ["大豆"],
    "大豆":   ["黄豆"],
    "绿豆芽": ["豆芽"],
    "豆芽":   ["绿豆芽"],
    "猪里脊": ["里脊肉"],
    "里脊肉": ["猪里脊"],
    "五花肉": ["猪肉"],
    "猪五花": ["猪肉"],
}

# 查询食材营养信息
def _lookup_nutrition(ingredient: str) -> tuple[dict | None, str]:
    """
    按优先级查询食材营养数据，返回 (nutrition_dict, matched_key)。
    优先级：精确匹配 → 模糊匹配（含子串）→ 别名扩展后重试 → 返回 None。
    """
    # 1. 精确匹配
    if ingredient in _NUTRITION_DB:
        return _NUTRITION_DB[ingredient], ingredient

    # 2. 模糊匹配：ingredient 是 key 的子串，或 key 是 ingredient 的子串
    for key, val in _NUTRITION_DB.items():
        if ingredient in key or key in ingredient:
            return val, key

    # 3. 别名扩展后重试（精确 → 模糊）
    for alias in _ALIASES.get(ingredient, []):
        if alias in _NUTRITION_DB:
            return _NUTRITION_DB[alias], alias
        for key, val in _NUTRITION_DB.items():
            if alias in key or key in alias:
                return val, key

    return None, ""


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

    # 所有需要累加的营养字段
    NUTRIENT_FIELDS = [
        "calories", "protein", "fat", "carbs", "dietary_fiber", "cholesterol",
        "vitamin_a", "vitamin_c", "vitamin_e", "thiamin", "riboflavin", "niacin",
        "calcium", "phosphorus", "potassium", "sodium", "magnesium",
        "iron", "zinc", "selenium", "copper", "manganese",
    ]

    area_cm2_total = total_area_mm2 / 100.0
    thickness_cm = thickness_mm / 10.0
    total = {f: 0.0 for f in NUTRIENT_FIELDS}
    details = []

    for ingredient, ratio in ratios.items():
        # Step 1-2：面积 → 体积
        ingredient_area_cm2 = area_cm2_total * ratio
        ingredient_volume_cm3 = ingredient_area_cm2 * thickness_cm

        # Step 3：查密度，计算重量
        nutrition, matched_key = _lookup_nutrition(ingredient)
        if nutrition is not None:
            print(f"  [营养库] 📚 {ingredient} → 匹配到：{matched_key}")

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
        factor = weight_g / 100.0

        row = {
            "ingredient":   ingredient,
            "ratio":        ratio,
            "area_cm2":     round(ingredient_area_cm2, 2),
            "volume_cm3":   round(ingredient_volume_cm3, 2),
            "density":      density,
            "weight_g":     round(weight_g, 1),
        }
        for f in NUTRIENT_FIELDS:
            row[f] = round(nutrition[f] * factor, 3)
            total[f] += row[f]
        details.append(row)

    print(f"  [营养计算] 完成，总热量 {total['calories']:.1f} kcal")
    return json.dumps({
        "details": details,
        "total": {k: round(v, 3) for k, v in total.items()},
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 输出结构定义
# ---------------------------------------------------------------------------

class IngredientNutrition(BaseModel):
    name:          str   = Field(description="食材名称")
    weight_g:      float = Field(description="估算重量（克）")
    # 宏量营养素
    calories:      float = Field(description="热量（kcal）")
    protein:       float = Field(description="蛋白质（克）")
    fat:           float = Field(description="脂肪（克）")
    carbs:         float = Field(description="碳水化合物（克）")
    dietary_fiber: float = Field(description="膳食纤维（克）")
    cholesterol:   float = Field(description="胆固醇（毫克）")
    # 维生素
    vitamin_a:     float = Field(description="维生素A（μg RAE）")
    vitamin_c:     float = Field(description="维生素C（毫克）")
    vitamin_e:     float = Field(description="维生素E（毫克）")
    thiamin:       float = Field(description="硫胺素B1（毫克）")
    riboflavin:    float = Field(description="核黄素B2（毫克）")
    niacin:        float = Field(description="烟酸B3（毫克）")
    # 矿物质
    calcium:       float = Field(description="钙（毫克）")
    phosphorus:    float = Field(description="磷（毫克）")
    potassium:     float = Field(description="钾（毫克）")
    sodium:        float = Field(description="钠（毫克）")
    magnesium:     float = Field(description="镁（毫克）")
    iron:          float = Field(description="铁（毫克）")
    zinc:          float = Field(description="锌（毫克）")
    selenium:      float = Field(description="硒（微克）")
    copper:        float = Field(description="铜（毫克）")
    manganese:     float = Field(description="锰（毫克）")


class TotalNutrition(BaseModel):
    calories:      float = Field(description="总热量（kcal）")
    protein:       float = Field(description="总蛋白质（克）")
    fat:           float = Field(description="总脂肪（克）")
    carbs:         float = Field(description="总碳水化合物（克）")
    dietary_fiber: float = Field(description="总膳食纤维（克）")
    cholesterol:   float = Field(description="总胆固醇（毫克）")
    vitamin_a:     float = Field(description="总维生素A（μg RAE）")
    vitamin_c:     float = Field(description="总维生素C（毫克）")
    vitamin_e:     float = Field(description="总维生素E（毫克）")
    thiamin:       float = Field(description="总硫胺素B1（毫克）")
    riboflavin:    float = Field(description="总核黄素B2（毫克）")
    niacin:        float = Field(description="总烟酸B3（毫克）")
    calcium:       float = Field(description="总钙（毫克）")
    phosphorus:    float = Field(description="总磷（毫克）")
    potassium:     float = Field(description="总钾（毫克）")
    sodium:        float = Field(description="总钠（毫克）")
    magnesium:     float = Field(description="总镁（毫克）")
    iron:          float = Field(description="总铁（毫克）")
    zinc:          float = Field(description="总锌（毫克）")
    selenium:      float = Field(description="总硒（微克）")
    copper:        float = Field(description="总铜（毫克）")
    manganese:     float = Field(description="总锰（毫克）")


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
用户会提供视觉识别的菜品信息（菜名、食材列表、烹饪方式）和食物投影面积，你需要分析这道菜的营养成分。

分析步骤：
1. 确定参与计算的食材及其比例
   - 以用户提供的【食材列表】为准（这是视觉识别结果），忽略盐、糖、酱油等调味料
   - 调用 search_recipe_db 获取该菜品的估算厚度（thickness_mm）和各食材的参考比例
   - 若 recipe_db 中有该菜品：以视觉识别的食材为准，用 recipe_db 的比例作参考；
     若识别食材与 recipe_db 不符，优先相信视觉识别结果，自主调整比例使之归一化至和为 1
   - 若 recipe_db 中无该菜品：thickness_mm 使用默认值 15，根据菜品烹饪方式和食材自主推理各食材比例
2. 调用 calculate_nutrition，传入：
   - total_area_mm2：用户提供的食物总投影面积
   - ingredients_ratios：上一步确定的食材比例（JSON 字符串，比例之和为 1）
   - thickness_mm：上一步得到的厚度
3. 若 calculate_nutrition 返回某食材"本地无数据"，从 web_raw 中提取该食材的密度和基础营养数据（热量、蛋白质、脂肪、碳水），
   结合其 area_cm2 和 thickness_mm 手动补算该食材的重量和营养，微量元素填 0，并纳入总计
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
        ingredients: list = field(default_factory=lambda: ["猪肉", "酱油", "糖","玉米"])
        cooking_method: str = "红烧"
        notes: str = ""
        total_food_area_mm2: float = 8000.0

    report = analyze(MockPipelineResult())
    print("\n" + "="*50)
    print("营养分析报告（结构化）")
    print("="*50)
    print(report.model_dump_json(indent=2, ensure_ascii=False))

