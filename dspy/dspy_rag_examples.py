"""
DSPy 检索增强生成(RAG)示例
展示如何结合外部知识库进行问答
"""

import dspy
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class DocumentStore:
    """简单文档存储类"""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_documents(self, docs: List[Dict[str, str]]):
        """添加文档到存储"""
        for doc in docs:
            self.documents.append(doc)
            # 生成嵌入
            embedding = self.model.encode(doc['content'])
            self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """搜索相关文档"""
        if not self.documents:
            return []

        # 生成查询嵌入
        query_embedding = self.model.encode(query)

        # 计算相似度
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        # 获取top_k最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            results.append(doc)

        return results

class RAGPipeline(dspy.Module):
    """检索增强生成管道"""

    def __init__(self, document_store: DocumentStore):
        super().__init__()
        self.document_store = document_store

        # 检索相关文档
        self.retrieve_docs = dspy.ChainOfThought(
            "query -> search_query, retrieval_intent"
        )

        # 理解检索到的文档
        self.understand_docs = dspy.ChainOfThought(
            "query, retrieved_documents -> key_information, relevant_context"
        )

        # 生成最终答案
        self.generate_answer = dspy.ChainOfThought(
            "query, key_information, relevant_context -> comprehensive_answer, sources"
        )

    def forward(self, query: str):
        """RAG管道的完整流程"""
        # 1. 优化查询用于检索
        retrieval_plan = self.retrieve_docs(query=query)
        search_query = retrieval_plan.search_query if hasattr(retrieval_plan, 'search_query') else query

        # 2. 检索相关文档
        retrieved_docs = self.document_store.search(search_query, top_k=3)
        docs_content = [doc['content'] for doc in retrieved_docs]

        # 3. 理解文档内容
        doc_understanding = self.understand_docs(
            query=query,
            retrieved_documents="\n".join(docs_content)
        )

        # 4. 生成答案
        final_answer = self.generate_answer(
            query=query,
            key_information=doc_understanding.key_information,
            relevant_context=doc_understanding.relevant_context
        )

        return dspy.Prediction(
            query=query,
            search_query=search_query,
            retrieved_documents=retrieved_docs,
            key_information=doc_understanding.key_information,
            comprehensive_answer=final_answer.comprehensive_answer,
            sources=final_answer.sources
        )

class MultiHopRAG(dspy.Module):
    """多跳检索增强生成，用于复杂查询"""

    def __init__(self, document_store: DocumentStore):
        super().__init__()
        self.document_store = document_store

        # 分解查询
        self.decompose_query = dspy.ChainOfThought(
            "complex_query -> subqueries, query_plan"
        )

        # 执行每个子查询
        self.answer_subquery = dspy.ChainOfThought(
            "subquery, context -> subanswer, additional_queries"
        )

        # 综合答案
        self.synthesize_answers = dspy.ChainOfThought(
            "original_query, subanswers, synthesis_context -> final_answer, confidence"
        )

    def forward(self, complex_query: str):
        """多跳RAG流程"""
        # 1. 分解复杂查询
        decomposition = self.decompose_query(complex_query=complex_query)
        subqueries = decomposition.subqueries if hasattr(decomposition, 'subqueries') else [complex_query]

        # 2. 处理每个子查询
        subanswers = []
        accumulated_context = ""

        for i, subquery in enumerate(subqueries):
            # 检索相关文档
            retrieved_docs = self.document_store.search(subquery, top_k=2)
            docs_content = [doc['content'] for doc in retrieved_docs]
            context = accumulated_context + "\n" + "\n".join(docs_content)

            # 回答子查询
            subanswer_result = self.answer_subquery(
                subquery=subquery,
                context=context
            )

            subanswers.append({
                "subquery": subquery,
                "answer": subanswer_result.subanswer,
                "sources": retrieved_docs
            })

            # 更新累积上下文
            accumulated_context += f"\n子查询{i+1}: {subquery}\n答案: {subanswer_result.subanswer}\n"

        # 3. 综合最终答案
        synthesis = self.synthesize_answers(
            original_query=complex_query,
            subanswers=str(subanswers),
            synthesis_context=accumulated_context
        )

        return dspy.Prediction(
            original_query=complex_query,
            subqueries=subqueries,
            subanswers=subanswers,
            final_answer=synthesis.final_answer,
            confidence=synthesis.confidence
        )

def create_sample_documents():
    """创建示例文档集合"""
    documents = [
        {
            "id": "1",
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。主要类型包括监督学习、无监督学习和强化学习。监督学习使用标记的训练数据，无监督学习发现数据中的模式，强化学习通过奖励和惩罚来学习最优行为。"
        },
        {
            "id": "2",
            "title": "深度学习架构",
            "content": "深度学习使用多层神经网络来学习数据的复杂表示。常见的架构包括卷积神经网络(CNN)用于图像处理，循环神经网络(RNN)用于序列数据，Transformer用于自然语言处理。这些架构通过反向传播算法进行训练，使用梯度下降优化网络参数。"
        },
        {
            "id": "3",
            "title": "自然语言处理技术",
            "content": "自然语言处理(NLP)是计算机科学和人工智能的一个分支，专注于计算机与人类语言之间的交互。主要任务包括文本分类、情感分析、机器翻译、问答系统和文本生成。现代NLP广泛使用预训练模型如BERT和GPT。"
        },
        {
            "id": "4",
            "title": "计算机视觉应用",
            "content": "计算机视觉使计算机能够从数字图像或视频中获取高层次的理解。主要应用包括物体检测、图像分割、人脸识别、姿态估计和场景理解。深度学习特别是CNN在计算机视觉任务中取得了显著成功。"
        },
        {
            "id": "5",
            "title": "强化学习原理",
            "content": "强化学习是一种机器学习方法，智能体通过与环境交互来学习最优策略。核心概念包括状态、动作、奖励、策略和价值函数。算法包括Q-learning、SARSA、深度Q网络(DQN)和策略梯度方法。广泛应用于游戏、机器人和控制系统。"
        },
        {
            "id": "6",
            "title": "模型评估与优化",
            "content": "机器学习模型的评估使用各种指标如准确率、精确率、召回率和F1分数。交叉验证是评估模型泛化能力的重要技术。模型优化包括超参数调优、正则化技术、早停方法和集成学习。过拟合和欠拟合是需要避免的常见问题。"
        }
    ]
    return documents

def setup_dspy():
    """配置DSPy环境"""
    lm = dspy.OpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    dspy.settings.configure(lm=lm)
    print("✅ DSPy环境配置完成")

def basic_rag_demo():
    """基础RAG演示"""
    print("\n📚 基础RAG系统演示")
    print("=" * 50)

    # 创建文档存储
    doc_store = DocumentStore()
    documents = create_sample_documents()
    doc_store.add_documents(documents)
    print(f"文档库已加载 {len(documents)} 个文档")

    # 创建RAG管道
    rag = RAGPipeline(doc_store)

    # 测试查询
    test_queries = [
        "什么是机器学习？",
        "深度学习有哪些主要架构？",
        "如何评估机器学习模型？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        try:
            result = rag(query=query)
            print(f"搜索查询: {result.search_query}")
            print(f"检索到的文档数量: {len(result.retrieved_documents)}")
            for j, doc in enumerate(result.retrieved_documents, 1):
                print(f"  文档{j}: {doc['title']} (相似度: {doc['similarity_score']:.3f})")
            print(f"关键信息: {result.key_information}")
            print(f"综合答案: {result.comprehensive_answer}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 40)

def multi_hop_rag_demo():
    """多跳RAG演示"""
    print("\n🔄 多跳RAG系统演示")
    print("=" * 50)

    # 创建文档存储
    doc_store = DocumentStore()
    documents = create_sample_documents()
    doc_store.add_documents(documents)

    # 创建多跳RAG管道
    multi_hop_rag = MultiHopRAG(doc_store)

    # 测试复杂查询
    complex_queries = [
        "如何使用深度学习技术解决自然语言处理问题？",
        "机器学习模型的评估方法有哪些，如何避免过拟合？",
        "强化学习在计算机视觉中有什么应用？"
    ]

    for i, query in enumerate(complex_queries, 1):
        print(f"\n复杂查询 {i}: {query}")
        try:
            result = multi_hop_rag(complex_query=query)
            print(f"分解的子查询: {result.subqueries}")
            print(f"子查询答案:")
            for j, subanswer in enumerate(result.subanswers, 1):
                print(f"  {j}. {subanswer['subquery']}")
                print(f"     答案: {subanswer['answer']}")
            print(f"最终答案: {result.final_answer}")
            print(f"置信度: {result.confidence}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 40)

def document_retrieval_demo():
    """文档检索演示"""
    print("\n🔍 文档检索系统演示")
    print("=" * 50)

    # 创建文档存储
    doc_store = DocumentStore()
    documents = create_sample_documents()
    doc_store.add_documents(documents)

    # 测试检索
    test_queries = [
        "CNN的应用",
        "强化学习算法",
        "NLP任务"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = doc_store.search(query, top_k=3)
        print(f"检索结果:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc['title']}")
            print(f"     相似度: {doc['similarity_score']:.3f}")
            print(f"     内容片段: {doc['content'][:100]}...")
        print("-" * 30)

def main():
    """主函数"""
    print("📖 DSPy 检索增强生成(RAG)示例")
    print("=" * 50)

    # 配置DSPy环境
    setup_dspy()

    # 运行各种演示
    document_retrieval_demo()
    basic_rag_demo()
    multi_hop_rag_demo()

    print("\n🎉 RAG示例演示完成！")
    print("\n💡 提示:")
    print("1. RAG结合了检索和生成的优势")
    print("2. 文档质量直接影响答案质量")
    print("3. 多跳RAG适合处理复杂查询")
    print("4. 可以根据需要调整检索的文档数量")

if __name__ == "__main__":
    main()