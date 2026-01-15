# DSPy 使用示例

这是一个完整的DSPy（Declarative Self-improving Language Models）使用示例集合，展示了如何使用DSPy构建各种AI应用。

## 📁 文件结构

```
├── dspy_requirements.txt    # 依赖包列表
├── .env.example            # 环境变量示例
├── dspy_basic_qa.py        # 基础问答系统
├── dspy_cot_examples.py    # 链式思考(CoT)示例
├── dspy_rag_examples.py    # 检索增强生成(RAG)示例
├── run_dspy_examples.py    # 统一运行脚本
├── test_dspy_setup.py      # 环境测试脚本
└── README_DSPy.md          # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r dspy_requirements.txt
```

### 2. 配置API密钥

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，添加你的API密钥
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. 测试环境

```bash
python test_dspy_setup.py
```

### 4. 运行示例

```bash
# 交互式运行所有示例
python run_dspy_examples.py all --interactive

# 运行特定示例
python run_dspy_examples.py basic    # 基础问答
python run_dspy_examples.py cot      # 链式思考
python run_dspy_examples.py rag      # RAG系统

# 非交互式运行
python run_dspy_examples.py basic
```

## 📚 示例说明

### 1. 基础问答系统 (`dspy_basic_qa.py`)

展示DSPy的基础功能，包括：
- 简单的问答模块
- 增强版问答（问题分解）
- 模型训练和优化
- 评估指标

**特点：**
- 易于理解的代码结构
- 包含训练数据示例
- 展示了DSPy的核心概念

### 2. 链式思考示例 (`dspy_cot_examples.py`)

展示复杂的推理能力：
- 数学问题求解器
- 代码生成器
- 逻辑推理系统

**特点：**
- 多步骤推理过程
- 详细的思考链
- 验证和置信度评估

### 3. 检索增强生成 (`dspy_rag_examples.py`)

展示RAG系统的构建：
- 文档存储和检索
- 基础RAG管道
- 多跳RAG系统
- 向量相似度搜索

**特点：**
- 使用SentenceTransformer进行嵌入
- 支持复杂查询分解
- 模块化的RAG架构

## 🛠️ 核心概念

### DSPy模块

DSPy中的模块是可组合的组件：

```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought("input -> output")

    def forward(self, input):
        return self.chain(input=input)
```

### 签名（Signature）

定义输入输出关系：

```python
"question -> answer"  # 简单签名
"context, question -> answer, confidence"  # 复杂签名
```

### 链式思考（Chain of Thought）

让模型展示推理过程：

```python
cot = dspy.ChainOfThought("problem -> reasoning, answer")
result = cot(problem="数学问题")
```

### 训练和优化

DSPy支持多种优化方法：

```python
optimizer = dspy.BootstrapFewShot(metric=metric)
optimized_model = optimizer.compile(model, trainset=trainset)
```

## 🎯 使用场景

### 1. 问答系统
- 客服机器人
- 知识库查询
- 教育辅导

### 2. 内容生成
- 文档编写
- 代码生成
- 创意写作

### 3. 推理任务
- 数学问题求解
- 逻辑推理
- 决策支持

### 4. 检索增强
- 文档问答
- 知识检索
- 多跳推理

## ⚙️ 配置选项

### 模型配置

```python
# OpenAI
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key="your_key")

# Anthropic
lm = dspy.Anthropic(model="claude-3-sonnet", api_key="your_key")

# Google
lm = dspy.Google(model="gemini-pro", api_key="your_key")
```

### DSPy设置

```python
dspy.settings.configure(
    lm=lm,              # 语言模型
    rm=None,            # 检索模型（可选）
    temperature=0.7,    # 生成温度
    max_tokens=1000     # 最大token数
)
```

## 🔧 自定义扩展

### 创建自定义模块

```python
class CustomModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.ChainOfThought("intermediate -> output")

    def forward(self, input):
        step1_result = self.step1(input=input)
        final_result = self.step2(intermediate=step1_result.intermediate)
        return final_result
```

### 自定义评估指标

```python
def custom_metric(example, prediction, trace=None):
    # 自定义评估逻辑
    return score
```

## 🚨 注意事项

1. **API密钥安全**：不要将API密钥提交到版本控制系统
2. **成本控制**：使用适当的模型和参数控制API成本
3. **错误处理**：在生产环境中添加适当的错误处理
4. **性能优化**：考虑缓存和批处理以提高性能

## 📖 学习资源

- [DSPy官方文档](https://dspy-docs.vercel.app/)
- [DSPy GitHub仓库](https://github.com/stanfordnlp/dspy)
- [链式思考论文](https://arxiv.org/abs/2201.11903)
- [检索增强生成综述](https://arxiv.org/abs/2005.11401)

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License