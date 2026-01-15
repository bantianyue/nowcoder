"""
DSPy 基础问答系统示例
展示如何使用DSPy创建一个简单的问题回答系统
"""

import dspy
from typing import List
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class BasicQA(dspy.Module):
    """基础问答模块"""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        """前向传播，生成答案"""
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

class EnhancedQA(dspy.Module):
    """增强版问答模块，包含多个步骤"""

    def __init__(self):
        super().__init__()
        # 分解问题的步骤
        self.decompose = dspy.ChainOfThought("question -> subquestions")
        # 回答每个子问题
        self.answer_subquestion = dspy.ChainOfThought("question -> answer")
        # 综合答案
        self.synthesize = dspy.ChainOfThought("question, subanswers -> final_answer")

    def forward(self, question):
        """分解问题并生成综合答案"""
        # 1. 分解问题
        subquestions_result = self.decompose(question=question)
        subquestions = subquestions_result.subquestions

        # 2. 回答每个子问题
        subanswers = []
        for subq in subquestions:
            answer_result = self.answer_subquestion(question=subq)
            subanswers.append(answer_result.answer)

        # 3. 综合最终答案
        final_result = self.synthesize(
            question=question,
            subanswers=subanswers
        )

        return dspy.Prediction(
            subquestions=subquestions,
            subanswers=subanswers,
            final_answer=final_result.final_answer
        )

def setup_dspy():
    """配置DSPy环境"""
    # 配置语言模型
    lm = dspy.OpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    dspy.settings.configure(lm=lm)

    print("✅ DSPy环境配置完成")

def create_training_data():
    """创建训练数据"""
    training_examples = [
        dspy.Example(
            question="什么是机器学习？",
            answer="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。"
        ),
        dspy.Example(
            question="Python和Java有什么区别？",
            answer="Python是解释型语言，语法简洁，适合快速开发；Java是编译型语言，性能更好，适合大型企业应用。"
        ),
        dspy.Example(
            question="什么是深度学习？",
            answer="深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。"
        ),
        dspy.Example(
            question="如何优化神经网络？",
            answer="可以通过调整学习率、使用正则化技术、优化网络结构、增加训练数据等方式来优化神经网络。"
        ),
        dspy.Example(
            question="什么是过拟合？",
            answer="过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象，通常因为模型过于复杂。"
        )
    ]
    return training_examples

def basic_qa_demo():
    """基础问答演示"""
    print("\n🚀 基础问答系统演示")
    print("=" * 50)

    # 创建问答模块
    qa = BasicQA()

    # 测试问题
    test_questions = [
        "什么是人工智能？",
        "如何学习编程？",
        "什么是云计算？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        try:
            result = qa(question=question)
            print(f"答案: {result.answer}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 30)

def enhanced_qa_demo():
    """增强版问答演示"""
    print("\n🎯 增强版问答系统演示")
    print("=" * 50)

    # 创建增强版问答模块
    qa = EnhancedQA()

    # 测试复杂问题
    complex_questions = [
        "如何在机器学习项目中防止过拟合？",
        "深度学习和传统机器学习的主要区别是什么？"
    ]

    for i, question in enumerate(complex_questions, 1):
        print(f"\n复杂问题 {i}: {question}")
        try:
            result = qa(question=question)
            print(f"子问题: {result.subquestions}")
            print(f"子答案: {result.subanswers}")
            print(f"最终答案: {result.final_answer}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 30)

def training_demo():
    """训练演示"""
    print("\n🎓 DSPy模型训练演示")
    print("=" * 50)

    # 创建训练数据
    training_data = create_training_data()
    print(f"训练数据数量: {len(training_data)}")

    # 创建验证数据
    validation_data = training_data[:2]  # 使用前两个作为验证数据

    # 创建评估指标
    metric = dspy.evaluate.answer_exact_match

    # 创建基础模型
    model = BasicQA()

    # 配置优化器
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2
    )

    print("开始训练...")
    try:
        # 训练模型
        optimized_model = optimizer.compile(
            model.compile(),
            trainset=training_data[2:],  # 使用后三个作为训练数据
            valset=validation_data
        )

        print("✅ 训练完成！")

        # 测试优化后的模型
        test_question = "什么是深度学习？"
        result = optimized_model(question=test_question)
        print(f"\n测试问题: {test_question}")
        print(f"优化后答案: {result.answer}")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("这可能是因为缺少API密钥或网络连接问题。")

def main():
    """主函数"""
    print("🤖 DSPy 问答系统示例")
    print("=" * 50)

    # 配置DSPy环境
    setup_dspy()

    # 运行各种演示
    basic_qa_demo()
    enhanced_qa_demo()

    # 训练演示（可选，需要API密钥）
    training_choice = input("\n是否运行训练演示？(需要API密钥) [y/N]: ").lower()
    if training_choice == 'y':
        training_demo()

    print("\n🎉 DSPy示例演示完成！")
    print("\n💡 提示:")
    print("1. 确保在.env文件中配置了API密钥")
    print("2. 可以调整模型参数来获得更好的效果")
    print("3. 训练过程需要一些时间，请耐心等待")

if __name__ == "__main__":
    main()