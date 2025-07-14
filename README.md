
# zhzAI - “小脑计划” (Project Cerebellum) 微模型铸造厂

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
</p>

<p align="center">
  <strong>告别大模型的“概率性”烦恼，为您的AI应用，亲手铸造一个100%可靠、极致高效的“微模型大脑”！</strong>
</p>

---

`model_foundry` 是一个独特的“宪法驱动”微模型铸造厂。它提供了一套完整的、从Python训练到任何语言（如Rust）部署的工业级全自动解决方案，并完美解决了`scikit-learn`复杂预处理器跨语言部署时最棘手的一致性难题。

## 为什么需要“小脑计划”？—— 在离线与低配的边缘，寻求智能的最优解

我们的终极目标是打造一个能**在任何设备上、完全离线运行**的个人AI助理。这个严苛的约束，意味着我们必须依赖像Qwen-0.6B这样极致小巧的本地模型。

但这带来了一个残酷的现实：**0.6B模型的能力，不足以独立、可靠地承担精准意图识别的重任。**

直接让它去判断用户的复杂指令，我们得到的是：

-   **灾难性的幻觉：** 在被要求“删除记忆”时，它可能会去执行“搜索”。
-   **高昂的推理成本：** 即便是小模型，每一次不必要的推理都在消耗宝贵的计算资源和用户的等待时间。
-   **不可预测的延迟：** 无法为用户的每一次交互提供稳定、瞬时的反馈。

> 一个真正可用的本地AI，绝不能是这样。

**“小脑计划”正是我们对这一困境给出的、最硬核的工程化回答。**

我们拒绝在“智能”和“性能”之间做妥协。我们选择**混合智能**，构建一个分工明确、协同作战的“大脑”：

1.  **让代码规则成为“反射弧”**  
    对于“删除”、“修改”等高风险指令，我们用100%确定的代码规则进行**零成本、零延迟**的瞬时捕获。这是系统不可动摇的“安全保险丝”。

2.  **让微模型成为“小脑”**  
    对于“这是个问题还是陈述？”这类海量的、模式化的判断，我们通过本仓库的流水线，**铸造**出专用的、仅有KB大小的微模型。它比0.6B的LLM**快数百倍、准数个数量级**，且行为高度可控。它以极低的资源开销，完美地弥补了小尺寸LLM在精准分类能力上的短板。

3.  **让LLM成为“大脑皮层”**  
    只有在真正需要深度理解、推理和生成时（例如，从长对话中提取核心事实），我们才调用宝贵的LLM资源，让它去完成自己最擅长、最高价值的工作。

**`model_foundry`就是您的“小脑”铸造工厂。** 它将赋予您为AI应用量产各种高精度、低成本“辅助处理器”的能力，让您的小尺寸本地模型，也能构建出一个真正稳定、高效、智能的混合AI系统。

---

## ✨ 核心特性

-   **蓝图驱动 (Blueprint-Driven):** 在`blueprints/`下定义新的模型“蓝图”（宪法+配置），即可轻松扩展，铸造全新架构的模型。
-   **一键式自动化引擎:** 由`main.py`统一调度，告别繁琐的手动分步执行。
-   **工业级数据提纯:** 包含精确去重与基于MinHash的近似去重，以及类别平衡策略。
-   **生产级模型导出:** 最终产出物是跨平台、高性能的`.onnx`格式模型，以及通过**Protocol Buffers (Protobuf)** 序列化的、类型安全且保证顺序的`.bin`预处理器数据，完美适配Rust、C++等高性能环境。
-   **闭环验证体验:** 提供交互式的“模型唤醒实验室”，让您在Python环境中就能立刻与自己铸造的模型进行对话，所见即所得。

---

## ⚙️ 快速上手：复现并测试现有模型

这是一个开箱即用的实践教程，让您在5分钟内体验铸造厂的威力。

### 1. 环境设置

-   克隆本仓库。
-   **安装Python依赖**:
    ```bash
    # (推荐) 创建并激活一个新的虚拟环境
    # python -m venv .venv
    # source .venv/bin/activate
    
    # 安装所有必需的包 (我们使用uv，你也可以用pip)
    uv pip install -r requirements.txt
    ```
-   在项目根目录创建`.env`文件，并填入您的Gemini API密钥。您可以使用多个以逗号分隔的密钥来应对速率限制：
    ```
    GEMINI_API_KEYS=your_api_key_1,your_api_key_2
    ```

### 2. 执行一键式自动化生产线

告别繁琐的手动操作！现在，只需一条命令即可启动整个流水线。在`model_foundry/`根目录下执行：

```bash
# 这将为 "text_classification" 蓝图下的所有任务，完整地执行从数据生成到模型导出的所有步骤。
python main.py --blueprint text_classification
```
> **提示:** 如果您已经有`datasets/raw`下的原始数据，流水线会自动跳过数据生成阶段，为您节省时间和API成本。

恭喜！您已经成功铸造出了两个高精度的微模型（`is_question` 和 `confirmation`），它们的所有必需文件都已位于`models/`目录下。

### 3. 唤醒并与你的模型对话！

现在，是时候与您亲手创造的AI进行第一次对话了。我们为您准备了一个独立的“唤醒实验室”。

-   运行交互式测试脚本：
    
    ```bash
    python scripts/interactive_tester.py
    ```

-   **实践测试示例：**
    
    程序启动后，会首先进入“是否为问题”分类器的测试环节。
    
    ```
    --- 任务1: “是否为问题”分类器测试 ---
    请输入任意一句话，看看模型如何判断 (输入'q'退出):
    > 今天天气怎么样？
      模型预测 ->: **Question**

    > 帮我记一下明天的会议
      模型预测 ->: **Statement**
    ```
---

## 开发者指南：如何铸造你自己的微模型

这才是铸造厂的真正威力所在！我们将以创建一个全新的“**情感分析分类器**”为例，一步步教您如何扩展。

### 第1步: 创建你的蓝图

在`blueprints/`目录下，为您要创造的模型家族新建一个文件夹。

```bash
mkdir blueprints/sentiment_analysis
```

### 第2步: 撰写你的“宪法”

在`blueprints/sentiment_analysis/`目录下，创建一个`constitution.py`文件。这是您模型灵魂的所在，您将在这里教会大模型如何为您生成数据。

**`blueprints/sentiment_analysis/constitution.py` 文件内容示例:**
```python
# blueprints/sentiment_analysis/constitution.py
def SENTIMENT_ANALYSIS_CONSTITUTION(samples_per_request: int) -> str:
    """为情感分析分类器生成数据的“宪法”。"""
    return f\"\"\"
你是一位为机器学习分类器生成训练数据的专家。
你的任务是生成一系列包含明确情感倾向的中文评论文本。
对于每一条文本，你必须为其分配一个分类标签：“Positive” (正面), “Negative” (负面), 或 “Neutral” (中性)。

**标签定义:**
- `Positive`: 包含明确的赞美、喜爱、满意等积极情绪。 (例如: "这款产品太棒了，强烈推荐！")
- `Negative`: 包含明确的批评、失望、不满等消极情绪。 (例如: "体验非常糟糕，不会再来了。")
- `Neutral`: 只是在陈述一个客观事实，没有明显的情感色彩。 (例如: "这个包裹是昨天下午送达的。")

**核心指令:**
1.  **精确生成 {samples_per_request} 条**独一无二的示例。
2.  输出**必须是**一个严格合法的JSON数组，每个对象包含 "text" 和 "label" 两个键。
3.  除了JSON数组，不要包含任何额外的文字。
4.  确保三种标签的数量大致均衡。
\"\"\"
```

### 第3步: 配置你的蓝图

在`blueprints/sentiment_analysis/`目录下，创建一个`config.py`文件。它就像是您生产线的控制面板。

**`blueprints/sentiment_analysis/config.py` 文件内容示例:**
```python
# blueprints/sentiment_analysis/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# 关键：定义这个蓝图包含的任务
TASKS = {
    "sentiment": { # 这是你给模型起的名字
        "constitution_func_name": "SENTIMENT_ANALYSIS_CONSTITUTION",
        "valid_labels": {"Positive", "Negative", "Neutral"},
    }
}

# 复用或自定义其他配置
# 这里我们直接复用 text_classification 的配置
from blueprints.text_classification.config import GENERATOR_CONFIG, REFINER_CONFIG, TRAINER_CONFIG
```

### 第4步: 链接你的训练器

打开根目录下的 `main.py` 文件，找到训练阶段的逻辑，告诉主控程序：当遇到`sentiment_analysis`蓝图时，应该使用哪个训练器。

由于情感分析也是一种文本分类，我们可以**复用**现有的`tfidf_classifier_trainer`。

**修改 `main.py`:**
```python
# ... 在 main.py 中找到这部分 ...
    if "train" in steps_to_run:
        print("\\n--- [阶段 3/4] 执行训练引擎 ---")
        # 在这里添加一个新的 elif 条件
        if args.blueprint == "text_classification" or args.blueprint == "sentiment_analysis":
            from foundry_engine.trainers import tfidf_classifier_trainer
            tfidf_classifier_trainer.run(config_module)
        elif args.blueprint == "sequence_tagging_ner":
            print("[提示] BiLSTM-CRF 训练器尚未实现。跳过此步骤。")
        # ...
```

### 第5步: 铸造！

一切就绪！现在，用一条命令铸造您的全新情感分析模型：


```bash
python main.py --blueprint sentiment_analysis
```

流水线会自动执行所有步骤，最终在`models/`目录下生成`sentiment_classifier.onnx`和`sentiment_preprocessor.bin`！

### 第6步: 唤醒你的新模型

最后，修改`scripts/interactive_tester.py`，让它加载并测试您的新模型，亲眼见证您的创造！

---
## 架构展望：添加新训练器

当您需要引入像`BiLSTM-CRF`这样的全新模型架构时，只需：
1.  在`foundry_engine/trainers/`下创建一个新的训练器文件，如`bilstm_crf_trainer.py`。
2.  在`main.py`中添加对应的`elif`逻辑来调用它。

本铸造厂的模块化设计，旨在让这类扩展变得轻松而愉快。
