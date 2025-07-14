# zhzAI - "小脑计划" (Project Cerebellum) 微模型铸造厂

<p align="center">
  <a href="https://github.com/zhzAI/model_foundry/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-PyTorch_&_Scikit--learn-orange.svg" alt="Framework"></a>
  <a href="#"><img src="https://img.shields.io/badge/Deployment-ONNX_&_Protobuf-green.svg" alt="Deployment"></a>
</p>

<p align="center">
  <strong>告别大模型的“概率性”烦恼，为您的AI应用，亲手铸造100%可靠、极致高效的“微模型大脑”！</strong>
</p>

---

`model_foundry` 是一个独特的“蓝图驱动”微模型铸造厂。它提供了一套完整的、从Python训练到任何语言（如Rust、C++）部署的工业级全自动解决方案，并完美解决了`scikit-learn`复杂预处理器跨语言部署时最棘手的一致性难题。

## 核心理念：为什么不直接用LLM？

这是一个非常好的问题。在能够调用强大LLM（无论是通过API还是本地部署）的时代，为什么我们还要“多此一举”地铸造微模型？

答案是：**因为一个真正可靠、高效、可信赖的AI产品，需要的远不止是强大的语言能力。**

直接将所有任务都交给一个单一的、庞大的LLM，就像让一位大学教授去同时担任公司的CEO、前台、保安和速记员。他或许都能做，但代价是高昂的、低效的，而且在关键时刻是不可靠的。

我们发现，单一LLM驱动的系统在现实世界中会面临三大“天花板”：

1.  **经济学天花板 (Economics)**
    *   **API成本:** 每一次无关紧要的意图判断（例如，用户输入的是问题还是陈述？），都在燃烧真金白银。对于高流量应用，这笔开销会迅速失控。
    *   **硬件成本:** 在本地运行LLM需要昂贵的GPU和巨大的内存。我们的目标是让AI运行在普通笔记本电脑甚至嵌入式设备上，这是一个硬性约束。

2.  **物理学天花板 (Physics)**
    *   **延迟:** 即便是最快的LLM，其“思考”时间也是按秒计算的。用户无法忍受每一次简单的交互（如确认“是/否”）都需要等待屏幕上的光标闪烁数秒。**毫秒级的响应速度**是构建流畅用户体验的物理基石。
    *   **离线能力:** 依赖云端API的模型，在断网或信号不佳的环境（如飞机、地下室、无人驾驶汽车经过的隧道）中会完全“失能”。

3.  **哲学天花板 (Philosophy)**
    *   **可靠性与确定性:** LLM本质上是概率性的。你无法100%保证它在处理“删除我的项目”这条指令时，不会因为一次“随机的奇思妙想”而错误地理解成“搜索我的项目”。对于需要高确定性的任务（如操作数据库、控制硬件），这种“不可预测性”是致命的。
    *   **数据隐私:** 对于处理个人或企业敏感数据的应用，将每一条输入都发送给第三方API，是不可接受的。

**“小脑计划”正是我们为了突破这三大天花板，提出的工程化解决方案。**

我们采用**混合智能 (Hybrid Intelligence)** 架构，构建一个分工明确、协同作战的“数字大脑”：

1.  **代码规则 (反射弧 / Brainstem):**
    *   **职责**: 处理高风险、确定性指令（如`删除`, `修改`）。
    *   **优势**: **零成本、零延迟、100%可靠**。这是系统不可动摇的“安全保险丝”。

2.  **微模型 (小脑 / Cerebellum):**
    *   **职责**: 使用本铸造厂产出的、KB级的专用模型，处理海量的、高频的、模式化的识别任务（如意图分类、实体识别）。
    *   **优势**:
        *   **快如闪电**: 响应速度是**毫秒级**的，比LLM快数百倍。
        *   **极致高效**: 模型大小仅**几十到几百KB**，几乎不占用任何内存和计算资源。
        *   **高度可靠**: 在其专精的、有边界的任务上，准确率可轻松达到**99%以上**，远超通用LLM在该类任务上的表现。
        *   **完全离线**: 100%在本地运行，无网络依赖，保障数据隐私。

3.  **LLM (大脑皮层 / Cerebral Cortex):**
    *   **职责**: 只在真正需要**深度理解、推理、内容总结和生成**时调用。
    *   **优势**: 将宝贵的LLM资源用在“刀刃”上，去完成它最擅长、最高价值的工作，而不是消耗在简单的分类判断上。

**`model_foundry` 就是您的“小脑”铸造工厂。** 它让您的AI应用在保持轻量化和低成本的同时，获得企业级的**速度、可靠性与智能**。

## ✨ 特性

-   **蓝图驱动 (Blueprint-Driven):** 在`blueprints/`下定义新的模型“蓝图”（宪法+配置），即可轻松扩展，铸造全新架构的模型。
-   **一键式自动化引擎:** 由`main.py`和`test.py`统一调度，告别繁琐的手动分步执行。
-   **生产级模型导出:** 最终产出物是跨平台、高性能的`.onnx`格式模型，以及通过**Protocol Buffers (Protobuf)** 序列化的、类型安全且保证顺序的`.bin`预处理器数据。
-   **闭环验证体验:** 每个蓝图都包含独立的交互式测试器，让您能立刻与自己铸造的模型对话。

---

## 🚀 快速上手

### 1. 环境设置

```bash
# 克隆本仓库
git clone https://github.com/your-username/model_foundry.git
cd model_foundry

# (推荐) 创建并激活一个新的虚拟环境
# python -m venv .venv && source .venv/bin/activate

# 安装所有必需的包 (我们使用uv，你也可以用pip)
uv pip install -r requirements.txt
```

在项目根目录创建`.env`文件，并填入您的Gemini API密钥。您可以使用多个以逗号分隔的密钥来应对速率限制：
```env
GEMINI_API_KEYS=your_api_key_1,your_api_key_2
```

### 2. 铸造所有预置模型

只需两条命令，即可从零开始构建并验证仓库内预置的所有微模型。

```bash
# 1. 铸造文本分类模型 (is_question, confirmation)
python main.py --blueprint text_classification

# 2. 铸造命名实体识别模型 (ner_core_entity)
python main.py --blueprint sequence_tagging_ner
```
> **提示:** 如果`datasets/raw`下已存在原始数据，流水线会自动跳过数据生成阶段，为您节省时间和API成本。

### 3. 唤醒并与你的模型对话

我们为每个蓝图都提供了独立的“唤醒实验室”。

```bash
# 测试文本分类模型
python test.py --blueprint text_classification

# 测试命名实体识别模型
python test.py --blueprint sequence_tagging_ner
```

---

## 🛠️ 开发者指南：如何铸造一个全新的模型

我们将以创建一个“**情感分析分类器**”为例，展示铸造厂的扩展能力。

### 第1步: 定义蓝图

在`blueprints/`下创建新目录`sentiment_analysis`，并在其中添加两个文件：

**`blueprints/sentiment_analysis/constitution.py`** (定义数据生成规则)
```python
def SENTIMENT_ANALYSIS_CONSTITUTION(samples_per_request: int) -> str:
    # ... (此处省略，内容为定义 "Positive", "Negative", "Neutral" 的Prompt)
```

**`blueprints/sentiment_analysis/config.py`** (定义生产线配置)
```python
from pathlib import Path
# 复用现有配置，保持简洁
from blueprints.text_classification.config import GENERATOR_CONFIG, REFINER_CONFIG, TRAINER_CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

TASKS = {
    "sentiment": {
        "constitution_func_name": "SENTIMENT_ANALYSIS_CONSTITUTION",
        "valid_labels": {"Positive", "Negative", "Neutral"},
        "expected_keys": {"text", "label"},
    }
}
```

### 第2步: 链接训练器

打开根目录下的 `main.py`，让主控程序知道这个新蓝图应该使用哪个训练器。由于情感分析也是文本分类，我们复用现有的`tfidf_classifier_trainer`。

**修改 `main.py`:**
```python
# ... 在 main.py 的训练阶段逻辑中 ...
        # 添加对新蓝图的判断
        if args.blueprint in ["text_classification", "sentiment_analysis"]:
            tfidf_classifier_trainer.run(config_module)
        elif args.blueprint == "sequence_tagging_ner":
            bilstm_crf_trainer.run(config_module)
```

### 第3步: 铸造与测试

```bash
# 一键铸造你的新模型
python main.py --blueprint sentiment_analysis

# 在测试前，为新蓝图创建专属的 tester.py
# (参考 blueprints/text_classification/tester.py 进行创建)

# 唤醒你的新模型
python test.py --blueprint sentiment_analysis
```

本铸造厂的模块化设计，旨在让这类扩展变得轻松而愉快。