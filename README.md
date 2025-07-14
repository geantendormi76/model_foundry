<h1 align="center">zhzAI - "小脑计划" (Project Cerebellum)</h1>

<p align="center">
  <strong>告别大模型的“概率性”烦恼，为您的AI应用，亲手铸造100%可靠、极致高效的“微模型大脑”！</strong>
</p>

<p align="center">
  <a href="https://github.com/zhzAI/model_foundry/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-PyTorch_&_Scikit--learn-orange.svg" alt="Framework"></a>
  <a href="#"><img src="https://img.shields.io/badge/Deployment-ONNX_&_Protobuf-green.svg" alt="Deployment"></a>
</p>

---

`model_foundry` 是一个独特的“蓝图驱动”微模型铸造厂。它提供了一套完整的、从Python训练到任何语言（如Rust、C++）部署的工业级全自动解决方案，并完美解决了`scikit-learn`复杂预处理器跨语言部署时最棘手的一致性难题。

## 核心理念：为什么不直接用LLM？

这是一个非常好的问题。在能够调用强大LLM的时代，为什么我们还要“多此一举”地铸造微模型？

答案是：**因为一个真正可靠、高效、可信赖的AI产品，需要的远不止是强大的语言能力。**

直接将所有任务都交给一个单一的、庞大的LLM，就像让一位大学教授去同时担任公司的CEO、前台、保安和速记员。他或许都能做，但代价是高昂的、低效的，而且在关键时刻是不可靠的。我们发现，单一LLM驱动的系统在现实世界中会面临三大“天花板”：

| 天花板 (Ceiling) | 核心制约 (Constraint) | “小脑计划”解决方案 |
| :--- | :--- | :--- |
| 💸 **经济学 (Economics)** | **API与硬件成本高昂。** 每次调用都在燃烧经费，本地部署需要昂贵GPU。 | **成本几乎为零。** 微模型大小仅为KB级，在任何CPU上瞬时完成，无需昂贵硬件。|
| ⚡️ **物理学 (Physics)** | **高延迟与在线依赖。** LLM响应以秒计，且断网即“失能”。 | **毫秒级响应与完全离线。** 提供流畅的本地体验，不受网络环境限制。 |
| 🛡️ **哲学 (Philosophy)** | **概率性与不可靠。** LLM无法保证100%的确定性，关键任务存在风险。 | **高度可靠与可预测。** 在其专精领域，微模型准确率可达99%以上，行为稳定。|

**“小脑计划”** 正是为突破这三大天花板而生。我们采用**混合智能 (Hybrid Intelligence)** 架构，构建分工明确的“数字大脑”：

-   **代码规则 (反射弧 / Brainstem):** 处理高风险、确定性指令。**零成本、零延迟、100%可靠**。
-   **微模型 (小脑 / Cerebellum):** 处理海量的、高频的、模式化的识别任务。**快如闪电、极致高效、高度可靠**。
-   **LLM (大脑皮层 / Cerebral Cortex):** 只在需要**深度理解、推理和生成**时调用，发挥其最大价值。

**`model_foundry` 就是您的“小脑”铸造工厂。**

---

## ✨ 特性

-   **蓝图驱动 (Blueprint-Driven):** 在`blueprints/`下定义新的模型“蓝图”，即可轻松扩展，铸造全新架构的模型。
-   **一键式自动化引擎:** 由`main.py`和`test.py`统一调度，告别繁琐的手动分步执行。
-   **生产级模型导出:** 产出物为高性能的`.onnx`模型和`.bin`(Protobuf)预处理器数据，完美适配Rust、C++等高性能环境。
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

在项目根目录创建`.env`文件，并填入您的Gemini API密钥（可使用多个，以逗号分隔）：
```env
GEMINI_API_KEYS=your_api_key_1,your_api_key_2
```

### 2. 一键铸造与测试

只需两条命令，即可构建并测试仓库内预置的**文本分类模型**。

```bash
# 1. 铸造 (生成数据、训练、导出)
python main.py --blueprint text_classification

# 2. 测试 (交互式唤醒)
python test.py --blueprint text_classification
```
> 要铸造和测试**命名实体识别模型**，只需将`text_classification`替换为`sequence_tagging_ner`即可。

---

## 🛠️ 开发者指南：铸造一个全新的模型

以创建一个“**情感分析分类器**”为例：

1.  **定义蓝图**: 在`blueprints/`下创建`sentiment_analysis`目录，并在其中添加`constitution.py` (定义数据生成规则) 和 `config.py` (定义生产线配置)。

2.  **链接训练器**: 打开根目录下的 `main.py`，在训练阶段的逻辑中，添加对新蓝图的判断，并为其指定一个训练器（可复用现有训练器）。
    ```python
    # 在 main.py 中...
    if args.blueprint in ["text_classification", "sentiment_analysis"]:
        tfidf_classifier_trainer.run(config_module)
    ```

3.  **铸造与测试**:
    ```bash
    # 一键铸造
    python main.py --blueprint sentiment_analysis
    # 创建专属测试器 (参考其他蓝图的 tester.py)
    # ...
    # 唤醒新模型
    python test.py --blueprint sentiment_analysis
    ```
本铸造厂的模块化设计，旨在让这类扩展变得轻松而愉快。