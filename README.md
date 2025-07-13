
# zhzAI - “小脑计划” (Project Cerebellum) 微模型铸造厂

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

欢迎来到zhzAI的“小脑计划”微模型铸造厂。这里的核心使命是：**为zhzAI个人助理，设计、培育并交付一系列极致小巧、高性能、可100%离线运行的意图分类微模型。**

这个项目不仅仅是代码，它是一套完整的、经过实战检验的**AI工程化思想和自动化流水线**。它将抽象的“数据宪法”思想，通过一系列严谨的脚本，最终“铸造”成可直接部署到生产环境的`.onnx`模型和`.bin`预处理器数据文件。

## ✨ 项目哲学

我们坚信，一个真正可靠的AI助理，其核心调度逻辑不应完全依赖于概率性的、不可预测的大语言模型（LLM）。因此，我们采用“代码优先，模型辅助”的混合智能架构。

“小脑计划”正是这一哲学的产物。我们在这里培育的微模型（小脑），负责对用户的输入进行快速、确定性、高精度的第一层意图识别，为上层由Rust编写的核心逻辑（大脑）提供决策依据，而LLM则作为能力更强的“专家顾问”在需要时被调用。

## 🚀 核心特性

-   **宪法驱动的数据工程：** 首创`prompt_constitutions.py`，将数据生成的逻辑与代码分离，确保了源头数据的质量与一致性。
-   **全自动流水线：** 从API数据生成、深度提纯、模型训练、跨语言导出到最终验证，五个脚本即可完成端到端全流程。
-   **工业级数据提纯：** 包含精确去重与基于MinHash的近似去重，以及类别平衡策略，确保模型训练在高质量的数据集上进行。
-   **生产级模型导出：** 最终产出物是跨平台、高性能的`.onnx`格式模型，以及配套的、通过**Protocol Buffers (Protobuf)** 序列化的、类型安全且保证顺序的`.bin`预处理器数据文件。
-   **严格的质量保证：** 独立的`4_verify_models.py`脚本，作为模型出厂前的最后一道“黄金测试集”验证，确保其在Python环境下的行为100%符合预期。

## 🛠️ 流水线详解 (The Pipeline)

本铸造厂由`scripts/`目录下的六个核心模块驱动，它们如同一条精密的生产线，环环相扣。

| 脚本                               | 角色         | 职责                                                                                               |
| ---------------------------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| `prompt_constitutions.py`          | **大宪章**   | **项目的灵魂。** 以“宪法”形式定义每个分类任务的数据逻辑、边界和示例，确保数据生成的质量与一致性。 |
| `1_generate_datasets.py`           | **矿工**     | 读取“宪法”，调用大模型API（如Gemini），高效、并发地挖掘海量原始数据。内置API Key池和智能熔断机制。 |
| `2_refine_and_analyze.py`          | **精炼厂**   | 对原始数据进行深度清洗、去重、平衡和规范化，并产出详细的数据质量分析报告。                           |
| `3_train_and_evaluate_final.py`    | **铸造炉**   | 使用提纯后的数据，训练高性能的Scikit-learn模型，并将其“铸造”为可部署的`.onnx`和`.joblib`文件。      |
| `4_verify_models.py`               | **质检实验室** | 对出厂的`.onnx`和`.joblib`文件进行端到端的“黄金测试集”验证，确保其在Python环境下的行为100%符合预期。       |
| `5_export_preprocessor_data.py`| **灵魂提取器** | **跨语言部署的关键桥梁。** 从Python私有的`.joblib`文件中，提取出核心的词汇表和IDF权重，并将其**序列化为Protobuf二进制格式 (`.bin`)**，供Rust后端使用。|

## 📦 最终产出物 (Artifacts)

流水线成功运行后，您将在以下目录中找到最终的产出物：

-   `models/`: **最终交付的产品。**
    -   `is_question_classifier.onnx`: “是否为问题”分类器。
    -   `is_question_preprocessor.bin`: 配套的、可被Rust读取的Protobuf预处理器数据。
    -   `confirmation_classifier.onnx`: “肯定/否定”分类器。
    -   `confirmation_preprocessor.bin`: 配套的、可被Rust读取的Protobuf预处理器数据。
-   `datasets/processed/`: 提纯后，用于训练和测试的最终数据集 (`.jsonl`格式)。
-   `datasets/reports/`: 每次运行时生成的数据质量和模型性能报告 (`.md`格式)。

## ⚙️ 如何从零开始运行

1.  **环境设置**
    -   克隆本仓库。
    -   **安装Protobuf编译器 (`protoc`)**: 这是将`.proto`契约文件编译成代码的核心工具。请确保其版本**不低于3.19.0**。
        *   在Ubuntu/WSL上: `sudo apt install protobuf-compiler` (如果版本过低，请参考脚本注释从GitHub下载最新版)。
        *   在macOS上: `brew install protobuf`。
        *   在Windows上: 可通过`scoop`或`chocolatey`安装，或从GitHub下载。
    -   **安装Python依赖**: `uv pip install -r requirements.txt` (请确保您已创建该文件，且其中包含`protobuf`)。
    -   在项目根目录创建`.env`文件，并填入您的Gemini API密钥：
        ```
        GEMINI_API_KEYS=your_api_key_1,your_api_key_2
        ```

2.  **为Rust项目提供“数据契约”**
    -   这是一个**一次性**的手动步骤，用于连接Python和Rust两个项目。
    -   将zhzAI主项目`backend/micromodels/src/proto/preprocessor.proto`文件，复制到本项目的`scripts/`目录下。

3.  **清理旧数据 (可选，推荐用于完全重建)**
    -   为了保证从一个纯净的状态开始，建议删除`datasets/`和`models/`目录下的所有旧文件。

4.  **执行流水线**
    -   请严格按照以下顺序，在`scripts/`目录下执行脚本：
    
    ```bash
    # 1. 根据“宪法”生成原始数据
    python 1_generate_datasets.py

    # 2. 提纯数据并生成报告
    python 2_refine_and_analyze.py

    # 3. 训练模型并导出为ONNX
    python 3_train_and_evaluate_final.py

    # 4. 在“黄金测试集”上进行最终验证
    python 4_verify_models.py
    
    # 5. (关键) 为Rust后端导出Protobuf格式的预处理器数据
    python 5_export_preprocessor_data.py
    ```

## 🧠 已铸造模型规格

经过我们严谨的流水线作业，目前已成功铸造出两个达到生产标准的微模型：

1.  **`is_question_classifier`**
    -   **用途:** 判断用户输入是在**“信息征询” (`Question`)** 还是在下达**“指令或陈述” (`Statement`)**。
    -   **最终测试集准确率:** **96.36%**

2.  **`confirmation_classifier`**
    -   **用途:** 在AI请求用户确认的场景下，判断用户的回应是**“肯定” (`Affirm`)** 还是**“否定” (`Deny`)**。
    -   **最终测试集准确率:** **99.65%**

## 展望未来

“小脑计划”的成功收官，为zhzAI的混合智能架构打下了最坚实的基础。下一步，我们将启动**“核心大脑移植”**计划，将这两个高精度、可信赖的微模型集成到主项目的Rust后端中，以实现更智能、更高效、更可靠的用户意图调度。

---

### **更新内容摘要**

*   **核心特性**：明确指出最终产出物是基于**Protobuf**的`.bin`文件，而非JSON。
*   **流水线详解**：更新了`5_export_preprocessor_data.py`的职责描述，强调其现在生成的是Protobuf二进制格式。
*   **最终产出物**：将`.json`后缀全部更新为`.bin`，与实际产出保持一致。
*   **如何从零开始运行**：
    *   增加了**安装`protoc`编译器**的关键前置步骤。
    *   增加了**手动复制`.proto`文件**的一次性设置步骤，解决了跨项目路径问题。
    *   更新了第5步脚本的描述。

这份更新后的`README.md`现在准确地反映了我们最新、最稳健的工作流程。您可以放心地将其推送到您的GitHub仓库，为社区和未来的自己提供最清晰的指引。