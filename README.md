# NutriVision

**基于视觉感知与推理的餐食营养分析系统**

通过一张包含参照硬币的餐食图片，自动识别菜品、分割食物区域、估算重量，并由智能体推理输出结构化营养报告。无需专业设备，仅凭图片即可完成分析。

---

## 系统流程

```
输入图片（含1元硬币作为参照）
        │
        ▼
  Step 1  食物识别        Qwen-VL → 菜品名、食材列表、烹饪方式
        │
        ▼
  Step 2  硬币分割        YOLOv8 + SAM → 检测硬币，标定 px²/mm² 比例尺
        │
        ▼
  Step 3  食物分割        交互式 SAM → 用户手动框选食物区域（排除盘子）
        │
        ▼
  Step 4  面积换算        像素面积 × 比例尺 → 真实投影面积（mm²）
        │
        ▼
  Step 5  营养分析        DeepSeek 智能体 → 面积×食材比例 → 体积 → 重量 → 营养汇总
        │
        ▼
  输出：可视化图 + Pipeline JSON + 营养报告 JSON
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 食物识别 | Qwen-VL-Max（通义千问视觉模型，DashScope API） |
| 硬币检测 | YOLOv8（自训练硬币检测模型） |
| 图像分割 | Segment Anything Model（SAM ViT-B） |
| 智能体框架 | LangChain 1.x + DeepSeek |
| 联网搜索 | 博查 Web Search API |
| 图像处理 | OpenCV、NumPy、Matplotlib |

---

## 项目结构

```
NutriVision/
├── main.py                  # 主入口，串联完整流程
├── pipeline.py              # Step 1-4：识别、分割、面积换算
├── nutrition_agent.py       # Step 5：营养分析智能体
├── food_detection.py        # Qwen-VL 食物识别模块
├── interactive_segment.py   # 交互式 SAM 分割工具
├── coin_segmentation.py     # YOLO + SAM 硬币分割模块
├── coin_yolo/               # 硬币检测 YOLO 训练相关
├── data/
│   ├── nutrition_db.json    # 本地食材营养知识库
│   └── recipe_db.json       # 本地菜谱比例知识库
└── weights/
    ├── sam_vit_b_01ec64.pth # SAM 模型权重
    └── best.pt              # YOLO 硬币检测权重
```

---

## 快速开始

**1. 安装依赖**
```bash
conda activate nutrition
pip install -r requirements.txt
```

**2. 配置 API Key**

在 `.env` 文件中填写：
```
DASHSCOPE_API_KEY=   # Qwen-VL（食物识别）
DEEPSEEK_API_KEY=    # DeepSeek（智能体推理）
BOCHA_KEY=           # 博查搜索（联网查询营养/菜谱）
```

**3. 运行**
```bash
python main.py \
  --image Data/your_food.jpg \
  --sam weights/sam_vit_b_01ec64.pth \
  --yolo weights/best.pt
```

运行到 Step 3 时会弹出交互窗口，拖拽框选食物区域后按 **Q** 继续。

**跳过 Qwen-VL（调试用）：**
```bash
python main.py --image Data/test.jpg --sam weights/sam_vit_b_01ec64.pth --yolo weights/best.pt --no-llm
```

**输出文件**（保存在 `out/` 目录）：

| 文件 | 内容 |
|------|------|
| `*_pipeline.jpg` | 分割结果可视化 |
| `*_pipeline.json` | 硬币比例尺、食物区域面积等结构化数据 |
| `*_nutrition.json` | 各食材重量、营养汇总、饮食建议 |

---

## 单独测试各模块

```bash
# 仅测试食物识别
python food_detection.py

# 仅测试交互式分割
python interactive_segment.py

# 仅测试营养智能体（使用模拟数据）
python nutrition_agent.py
```

---

## 开发进度

### 已完成
- [x] 食物识别模块（Qwen-VL，支持菜品未识别时的外观描述 fallback）
- [x] 硬币检测与分割（YOLO 自训练 + SAM 精割，双比例尺校验）
- [x] 交互式食物分割（SAM 框选模式，排除盘子）
- [x] 面积换算（像素面积 → 真实投影面积 mm²）
- [x] 营养智能体框架（LangChain 1.x + DeepSeek，本地知识库 + 联网兜底）
- [x] 结构化输出（Pydantic NutritionReport，便于后续开发）
- [x] 主流程串联（main.py）

### 进行中
- [ ] 扩充本地知识库（`nutrition_db.json` / `recipe_db.json`）
- [ ] 自动化食物分割（现阶段采用 SAM 框选模式）

### 待开发
- [ ] 前端展示界面
- [ ] 多菜品支持（当前仅处理单道菜）
- [ ] 体积估算优化（引入深度信息）
...

---

## 注意事项

- 拍摄时需将 **1元人民币硬币**放在餐盘旁边作为比例参照
- 当前仅支持单道菜的分析
- 面积→重量的换算依赖估算厚度，存在一定误差，属于近似值
