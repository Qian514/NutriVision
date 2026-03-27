"""
餐食营养分析系统 — 主入口

完整流程：
  1. 食物识别   — Qwen-VL 识别菜名 + 食材 + 烹饪方式
  2. 硬币分割   — YOLO + SAM → px²/mm² 比例尺
  3. 食物分割   — 交互式 SAM，用户手动框选食物区域
  4. 面积换算   — 像素面积 → 真实投影面积（mm²）
  5. 营养分析   — 智能体推理：面积 → 体积 → 重量 → 营养

运行：
  python main.py \
    --image Data/test_coin.jpg \
    --sam weights/sam_vit_b_01ec64.pth \
    --yolo weights/best.pt \
    [--no-llm]  # 跳过 Qwen-VL，省 API 费用
"""

import argparse
from pathlib import Path

from pipeline import NutritionPipeline, visualize_pipeline, save_result_json
from nutrition_agent import analyze


def main():
    p = argparse.ArgumentParser(description="餐食营养分析系统")
    p.add_argument("--image",    required=True,  help="输入图片路径")
    p.add_argument("--sam",      required=True,  help="SAM 权重路径 (.pth)")
    p.add_argument("--yolo",     required=True,  help="硬币 YOLO 权重路径 (.pt)")
    p.add_argument("--no-llm",   action="store_true", help="跳过 Qwen-VL 食物识别")
    p.add_argument("--out-dir",  default="out",  help="输出目录（默认 out/）")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    image_stem = Path(args.image).stem

    # ── Step 1-4: Pipeline ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Step 1-4: Pipeline（识别 + 分割 + 面积换算）")
    print("="*60)

    pipeline = NutritionPipeline(
        sam_checkpoint=args.sam,
        yolo_weights=args.yolo,
        use_llm=not args.no_llm,
    )
    pipeline_result = pipeline.run(args.image)

    # 保存 pipeline 结果
    img_path  = str(out_dir / f"{image_stem}_pipeline.jpg")
    json_path = str(out_dir / f"{image_stem}_pipeline.json")
    visualize_pipeline(pipeline_result, save_path=img_path)
    save_result_json(pipeline_result, save_path=json_path)

    # ── Step 5: 营养分析智能体 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Step 5: 营养分析智能体")
    print("="*60)

    if pipeline_result.total_food_area_mm2 == 0:
        print("[警告] 未获得有效食物面积，跳过营养分析")
        return

    report = analyze(pipeline_result)

    # 打印报告
    print("\n" + "="*60)
    print("  营养分析报告")
    print("="*60)
    print(f"菜品      : {report.dish}")
    print(f"估算总重量: {report.total_weight_g:.1f} g")
    print("\n食材明细:")
    for ing in report.ingredients:
        print(f"  {ing.name:8s}  {ing.weight_g:6.1f}g  "
              f"热量{ing.calories:.0f}kcal  蛋白{ing.protein:.1f}g  "
              f"脂肪{ing.fat:.1f}g  碳水{ing.carbs:.1f}g")
    t = report.total_nutrition
    print(f"\n营养汇总  : 热量{t.calories:.0f}kcal | 蛋白{t.protein:.1f}g | 脂肪{t.fat:.1f}g | 碳水{t.carbs:.1f}g")
    print(f"饮食建议  : {report.advice}")

    # 保存 JSON
    report_path = out_dir / f"{image_stem}_nutrition.json"
    report_path.write_text(report.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[报告] 已保存: {report_path}")


if __name__ == "__main__":
    main()
