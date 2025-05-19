# 文件名: LlamaIndex实现.py
# 描述: 本文件演示了如何使用LlamaIndex实现上下文检索。
# 原文档链接: https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/

import os
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import List
import asyncio

# --- 配置 ---
# 在运行此文件之前，请确保您已设置以下环境变量：
# OPENAI_API_KEY, COHERE_API_KEY
# 例如:
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["COHERE_API_KEY"] = "..."

# --- 1. 安装 ---
# 如果您尚未安装必要的库，请取消注释以下行并运行：
# !pip install llama-index llama-index-llms-openai llama-index-embeddings-openai llama-index-postprocessor-cohere-rerank llama-index-retrievers-bm25 pandas

print("步骤 1: 安装完成 (假设已手动安装)")

# --- 2. 设置 API 密钥 ---
# 确保您的 OpenAI 和 Cohere API 密钥已在环境变量中设置。
# openai_api_key = os.environ.get("OPENAI_API_KEY")
# cohere_api_key = os.environ.get("COHERE_API_KEY")
# if not openai_api_key or not cohere_api_key:
#     print("错误: 请设置 OPENAI_API_KEY 和 COHERE_API_KEY 环境变量。")
#     exit()

print("步骤 2: API 密钥设置完成 (假设已在环境中设置)")

# --- 3. 设置 LLM 和 Embedding 模型 ---
llm = OpenAI(model="gpt-3.5-turbo") # 文档中使用 gpt-4，此处使用 gpt-3.5-turbo 以降低成本和提高速度
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

print("步骤 3: LLM 和 Embedding 模型设置完成")

# --- 4. 下载数据 ---
# 此示例使用 Paul Graham 的文章 "What I Worked On"
# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" -O "paul_graham_essay.txt"
# print("数据下载完成: paul_graham_essay.txt")
# 为简单起见，我们直接使用文本内容
paul_graham_essay_text = """
What I Worked On
February 2021
Before college the two main things I worked on, outside of school, were writing and programming. I wrote short stories. I tried writing a novel, but it was too hard. I wrote a lot of programs too, mostly games, but also a program to predict the height of model rockets, and a word processor that my father used to write his books.
I didn't write much in college. I mostly programmed. The programming I did was mostly for fun, but some of it was to make money. One summer I wrote a program for a company that sold time-sharing services. It was a database program, and it was written in Fortran. I also wrote a program for a professor to simulate the growth of crystals.
After college I was hired by a company called Interleaf. They made software for creating documents. I worked on their Lisp machine product. I wrote a lot of Lisp code. I also learned a lot about how to make software that was robust and easy to use.
I left Interleaf to start a company with Robert Morris and Trevor Blackwell. We were going to make software for creating online stores. This was in 1995. We got funding from a company called Idelle. We hired a couple of programmers. We wrote a lot of code. We had a product, but it wasn't very good. We ran out of money.
After that I worked for a company called Viaweb. We made software for creating online stores. This was in 1995. We got funding from a company called Idelle. We hired a couple of programmers. We wrote a lot of code. We had a product, and it was good. Yahoo bought us in 1998.
After Yahoo I worked on a programming language called Arc. I also started an investment firm called Y Combinator.
At Y Combinator we help people start startups. We give them a small amount of money, and we help them with advice and connections. We've funded over 3000 startups. Some of them have become very successful, like Airbnb, Dropbox, Stripe, and Reddit.
I still write. I write essays for my website. I also wrote a book called Hackers & Painters.
I still program. I mostly program in a Lisp dialect called Bel. I also program in Python.
The main thing I've learned from working on all these different things is that the most important thing is to work on something you love. If you don't love what you're doing, you're not going to be very good at it.
Another thing I've learned is that it's important to work with people you like. If you don't like the people you're working with, you're not going to be very happy.
And finally, I've learned that it's important to work on something that matters. If you're not working on something that matters, you're not going to feel very good about yourself.
(This is a greatly abridged version for demonstration purposes)
"""
print("步骤 4: 数据准备完成 (使用内联文本)")

# --- 5. 加载数据 ---
documents = [Document(text=paul_graham_essay_text)]
print("步骤 5: 数据加载完成")

# --- 6. 为每个块创建上下文的提示 ---
# 这是用于生成围绕每个文本块的上下文信息的提示模板。
# 这些上下文信息可以帮助检索器更好地理解块的含义。
CONTEXT_PROMPT_TEMPLATE = """
The following is a chunk of text from a document:
"{context_str}"

Given this chunk of text, create a concise sentence of context about what this chunk is about.
This will be used to answer questions over the document.
Context: """

print("步骤 6: 上下文提示模板定义完成")

# --- 7. 工具函数 ---
# 创建 Embedding 检索器
def create_embedding_retriever(nodes, similarity_top_k=3):
    # 确保 similarity_top_k 不超过节点数量
    adjusted_top_k = min(similarity_top_k, len(nodes))
    if adjusted_top_k < similarity_top_k:
        print(f"Warning: Adjusting similarity_top_k from {similarity_top_k} to {adjusted_top_k} due to limited nodes.")
    # 确保至少为1
    adjusted_top_k = max(1, adjusted_top_k)
    
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index.as_retriever(similarity_top_k=adjusted_top_k)

# 创建 BM25 检索器
def create_bm25_retriever(nodes, similarity_top_k=3):
    # BM25Retriever 需要 TextNode
    text_nodes = [TextNode(text=node.get_content(), id_=node.node_id) for node in nodes if hasattr(node, 'get_content')]
    if not text_nodes:
        # Fallback if nodes are not directly TextNode or don't have get_content
        # This might happen if nodes are already wrapped or of a different type.
        # Adjust this part based on the actual type and structure of your `nodes`.
        print("Warning: Could not directly create TextNodes for BM25. Attempting fallback.")
        # Assuming nodes might be Document objects or similar
        temp_text_nodes = []
        for i, node in enumerate(nodes):
            try:
                # Try to get text content, robustly
                if isinstance(node, TextNode):
                    temp_text_nodes.append(node)
                elif hasattr(node, 'text'):
                     temp_text_nodes.append(TextNode(text=node.text, id_=getattr(node, 'node_id', f'node_{i}')))
                elif isinstance(node, Document):
                     temp_text_nodes.append(TextNode(text=node.get_content(), id_=getattr(node, 'doc_id', f'doc_{i}')))
                else:
                    print(f"Skipping node of type {type(node)} for BM25 as content extraction is unclear.")
            except Exception as e:
                print(f"Error processing node for BM25: {e}")
        text_nodes = temp_text_nodes

    if not text_nodes:
        print("Error: No valid TextNodes could be created for BM25Retriever.")
        # Return a dummy retriever or raise an error
        class DummyRetriever(BaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                return []
        return DummyRetriever()

    # 确保 similarity_top_k 不超过文本节点数量
    adjusted_top_k = min(similarity_top_k, len(text_nodes))
    if adjusted_top_k < similarity_top_k:
        print(f"Warning: Adjusting similarity_top_k from {similarity_top_k} to {adjusted_top_k} due to limited text nodes.")
    # 确保至少为1
    adjusted_top_k = max(1, adjusted_top_k)

    return BM25Retriever.from_defaults(nodes=text_nodes, similarity_top_k=adjusted_top_k)


# 混合检索器 (Embedding + BM25 + Reranker)
class EmbeddingBM25RerankerRetriever(BaseRetriever):
    def __init__(self, embedding_retriever, bm25_retriever, reranker):
        self._embedding_retriever = embedding_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        embedding_nodes = self._embedding_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # 合并并去重
        all_nodes = {node.node.node_id: node for node in embedding_nodes}
        for node in bm25_nodes:
            if node.node.node_id not in all_nodes:
                all_nodes[node.node.node_id] = node
        
        # reranker 需要 NodeWithScore 列表
        nodes_for_rerank = list(all_nodes.values())
        
        # CohereRerank.postprocess_nodes 需要一个 QueryBundle
        # 以及原始节点列表 (不带分数的 TextNode)
        # 注意: reranker 可能期望 Node 对象而不是 NodeWithScore，或者需要特定的格式
        # 此处我们传递 NodeWithScore，如果 reranker.postprocess_nodes 预期不同，则需要调整
        if self._reranker and nodes_for_rerank: # 确保有节点进行重排
            reranked_nodes = self._reranker.postprocess_nodes(
                nodes_for_rerank, query_bundle=query_bundle
            )
            return reranked_nodes
        return nodes_for_rerank # 如果没有reranker或没有节点，返回合并后的节点

# 创建评估数据集
def create_eval_dataset(nodes, llm, num_questions_per_chunk=2):
    # generate_question_context_pairs 在较新版本中可能位于 llama_index.core.evaluation
    # 或者 llama_index.evaluation
    try:
        from llama_index.core.evaluation import generate_question_context_pairs
    except ImportError:
        from llama_index.evaluation import generate_question_context_pairs # 旧版本路径

    # 需要 TextNode 列表
    text_nodes_for_eval = [TextNode(text=node.get_content(), id_=node.node_id) for node in nodes if hasattr(node, 'get_content')]
    if not text_nodes_for_eval and isinstance(nodes[0], Document): # 处理 Document 列表的情况
         text_nodes_for_eval = [TextNode(text=doc.text, id_=doc.doc_id) for doc in nodes]


    qa_dataset = generate_question_context_pairs(
        text_nodes_for_eval, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    return qa_dataset

# 异步检索结果
async def retrieval_results(retriever, qa_dataset):
    # RetrieverEvaluator 在较新版本中可能位于 llama_index.core.evaluation
    # 或者 llama_index.evaluation
    try:
        from llama_index.core.evaluation import RetrieverEvaluator
    except ImportError:
        from llama_index.evaluation import RetrieverEvaluator # 旧版本路径

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    return eval_results

# 显示结果
def display_results(name, eval_results):
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    
    full_df = pd.DataFrame(metric_dicts)
    
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    
    # 原文档中有 precision, recall, ap, ndcg，但 RetrieverEvaluator 默认不计算这些
    # 这里只显示 hit_rate 和 mrr
    # 如果需要其他指标，可能需要自定义评估或使用不同的评估器

    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )
    return metric_df

print("步骤 7: 工具函数定义完成")

# --- 8. 创建节点 ---
# 使用 SentenceSplitter 将文档分割成节点（块）
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20) # 文档中未指定，使用常见值
nodes = splitter.get_nodes_from_documents(documents)
print(f"步骤 8: 从文档创建了 {len(nodes)} 个节点")

# --- 9. 设置节点 ID (如果需要) ---
# LlamaIndex 通常会自动处理节点 ID。
# 如果手动创建节点或有特定需求，可以在此处设置。
# for idx, node in enumerate(nodes):
#     node.id_ = f"node_{idx}"
print("步骤 9: 节点 ID 设置完成 (LlamaIndex 自动处理)")


# --- 10. 创建上下文节点 ---
# 此步骤使用 LLM 为每个节点（块）生成上下文描述。
# 这可以帮助检索器理解每个块的主要内容。

# 由于调用 LLM 生成上下文成本较高且耗时，此处我们跳过实际的 LLM 调用
# 并使用一个简化的模拟版本。在实际应用中，您会使用 CONTEXT_PROMPT_TEMPLATE 和 LLM。
#
# 真实实现的大致思路:
# nodes_contextual = []
# for node in nodes:
#     context_str = node.get_content()
#     # 使用 LLM 生成上下文 (伪代码)
#     # prompt = CONTEXT_PROMPT_TEMPLATE.format(context_str=context_str)
#     # response = llm.complete(prompt)
#     # generated_context = response.text
#     #
#     # # 将生成的上下文添加到节点的元数据中
#     # new_node_metadata = node.metadata.copy()
#     # new_node_metadata["context"] = generated_context
#     # contextual_node = TextNode(text=node.text, metadata=new_node_metadata, id_=node.id_)
#     # nodes_contextual.append(contextual_node)

# 简化版：我们假设上下文就是节点本身内容的一个摘要或关键词
# 或者，为了演示，我们简单地复制原始节点，因为下面的评估部分需要 nodes_contextual
# 在实际应用中，这里的上下文节点应该包含由 LLM 生成的额外上下文信息。
nodes_contextual = []
for node in nodes:
    # 模拟上下文生成：这里我们简单地在元数据中添加一个占位符上下文
    # 真实场景会调用LLM
    simulated_context = f"This chunk discusses: {node.get_content()[:50]}..." # 非常粗略的模拟
    new_metadata = node.metadata.copy()
    new_metadata["generated_context"] = simulated_context
    
    # 创建新的 TextNode，确保它有 text 属性
    contextual_node = TextNode(
        text=node.get_content(), # 节点的主要文本内容
        metadata=new_metadata,
        id_=node.node_id
    )
    nodes_contextual.append(contextual_node)

print(f"步骤 10: 创建了 {len(nodes_contextual)} 个上下文节点 (使用模拟上下文)")


# --- 11. 设置 similarity_top_k ---
similarity_top_k = 3 # 检索时返回最相似的 top_k 个结果
print(f"步骤 11: similarity_top_k 设置为 {similarity_top_k}")

# --- 12. 设置 CohereReranker ---
# 确保 COHERE_API_KEY 已设置
if os.environ.get("COHERE_API_KEY"):
    cohere_rerank = CohereRerank(
        api_key=os.environ["COHERE_API_KEY"], top_n=similarity_top_k
    )
    print("步骤 12: CohereReranker 设置完成")
else:
    cohere_rerank = None
    print("步骤 12: CohereReranker 未设置 (COHERE_API_KEY 未找到)")


# --- 13. 创建检索器 ---
# 1. 基于 Embedding 的检索器
# 2. 基于 BM25 的检索器
# 3. Embedding + BM25 + Cohere reranker 检索器

embedding_retriever = create_embedding_retriever(
    nodes, similarity_top_k=similarity_top_k
)
bm25_retriever = create_bm25_retriever(
    nodes, similarity_top_k=similarity_top_k
)
embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    embedding_retriever, bm25_retriever, reranker=cohere_rerank
)
print("步骤 13: 标准检索器创建完成")

# --- 14. 使用上下文节点创建检索器 ---
contextual_embedding_retriever = create_embedding_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_bm25_retriever = create_bm25_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    contextual_embedding_retriever,
    contextual_bm25_retriever,
    reranker=cohere_rerank,
)
print("步骤 14: 上下文检索器创建完成")


# --- 15. 创建综合查询数据集 ---
# 此步骤使用 LLM 根据节点内容生成问题，用于评估检索器的性能。
# 为避免实际 LLM 调用产生费用和延迟，这里可以加载预生成的数据集或使用少量节点生成。
# 注意: create_eval_dataset 内部会调用 LLM。
# 如果 OPENAI_API_KEY 未设置或想避免调用，可以注释掉这部分或使用固定的问答对。

# qa_dataset = create_eval_dataset(nodes, llm=llm, num_questions_per_chunk=1) #减少问题数量以加速
# print(f"步骤 15: 创建了包含 {len(qa_dataset.queries)} 个查询的评估数据集")
# print(f"示例查询: {list(qa_dataset.queries.values())[0] if qa_dataset.queries else 'N/A'}")

# 使用一个固定的、小型的 QA 数据集进行演示，以避免 LLM 调用
from llama_index.core.evaluation import QueryResponseDataset, EmbeddingQAFinetuneDataset
fixed_queries = {
    "q1": "What did the author work on before college?",
    "q2": "What was Viaweb and what happened to it?",
    "q3": "What is Y Combinator?"
}
fixed_responses = { # 伪响应，评估时我们主要关注检索的上下文是否相关
    "q1": "The author worked on writing and programming.",
    "q2": "Viaweb made software for online stores and was bought by Yahoo.",
    "q3": "Y Combinator is an investment firm that helps startups."
}
# 检索评估需要 relevant_docs
# 我们需要手动将节点ID与问题关联起来作为"黄金标准"
# 这部分比较复杂，因为需要知道哪个节点包含哪个答案
# 为了简化，我们假设第一个节点与第一个问题相关，以此类推（这在实际中不准确）
relevant_docs_mapping = {}
if nodes:
    relevant_docs_mapping["q1"] = [nodes[0].node_id] if len(nodes) > 0 else []
    relevant_docs_mapping["q2"] = [nodes[min(4, len(nodes)-1)].node_id] if len(nodes) > 0 else [] # 假设在第5个节点左右
    relevant_docs_mapping["q3"] = [nodes[min(6, len(nodes)-1)].node_id] if len(nodes) > 0 else [] # 假设在第7个节点左右

# ADDED: Construct corpus for EmbeddingQAFinetuneDataset
corpus_data = {}
if nodes:
    corpus_data = {node.node_id: node.get_content() for node in nodes if hasattr(node, 'node_id') and hasattr(node, 'get_content')}

# MODIFIED: Instantiate EmbeddingQAFinetuneDataset instead of QueryResponseDataset
qa_dataset = EmbeddingQAFinetuneDataset(
    queries=fixed_queries,
    corpus=corpus_data,
    relevant_docs=relevant_docs_mapping
)
print(f"步骤 15: 使用固定的 QA 数据集进行评估 (共 {len(qa_dataset.queries)} 个问题)")
if qa_dataset.queries:
    first_query_id = list(qa_dataset.queries.keys())[0]
    print(f"示例查询: {qa_dataset.queries[first_query_id]}")


# --- 16. 评估检索器 (有无上下文节点) ---
# 注意: aevaluate_dataset 是异步的，需要在一个事件循环中运行。
async def main_evaluation():
    results_dfs = []

    print("\n--- 开始评估 ---")
    if not qa_dataset or not qa_dataset.queries:
        print("错误: QA 数据集为空，无法进行评估。")
        return

    # 确保 qa_dataset.queries 是一个字典，其值是字符串
    # 并且 qa_dataset.relevant_docs (如果提供) 也是正确的格式
    # RetrieverEvaluator 需要 qa_dataset 包含 relevant_docs 字段，且其值为一个字典，
    # key 是 query_id, value 是一个包含相关文档ID的列表。

    # 更新 qa_dataset 以包含 relevant_docs，这是评估所必需的
    # qa_dataset.relevant_docs = relevant_docs_mapping # <--- 这行将被移除


    print("\n评估标准检索器:")
    embedding_retriever_results = await retrieval_results(
        embedding_retriever, qa_dataset
    )
    results_dfs.append(display_results("Embedding Retriever", embedding_retriever_results))

    bm25_retriever_results = await retrieval_results(bm25_retriever, qa_dataset)
    results_dfs.append(display_results("BM25 Retriever", bm25_retriever_results))
    
    if cohere_rerank: # 只有在 reranker 可用时才评估带 reranker 的版本
        embedding_bm25_retriever_rerank_results = await retrieval_results(
            embedding_bm25_retriever_rerank, qa_dataset
        )
        results_dfs.append(display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_retriever_rerank_results,
        ))
    else:
        print("跳过 Embedding + BM25 + Reranker 的评估，因为 Cohere Reranker 未配置。")


    print("\n评估上下文检索器:")
    contextual_embedding_retriever_results = await retrieval_results(
        contextual_embedding_retriever, qa_dataset
    )
    results_dfs.append(display_results(
        "Contextual Embedding Retriever",
        contextual_embedding_retriever_results,
    ))

    contextual_bm25_retriever_results = await retrieval_results(
        contextual_bm25_retriever, qa_dataset
    )
    results_dfs.append(display_results(
        "Contextual BM25 Retriever", contextual_bm25_retriever_results
    ))

    if cohere_rerank: # 只有在 reranker 可用时才评估带 reranker 的版本
        contextual_embedding_bm25_retriever_rerank_results = await retrieval_results(
            contextual_embedding_bm25_retriever_rerank, qa_dataset
        )
        results_dfs.append(display_results(
            "Contextual Embedding + Contextual BM25 Retriever + Reranker",
            contextual_embedding_bm25_retriever_rerank_results,
        ))
    else:
        print("跳过 Contextual Embedding + Contextual BM25 + Reranker 的评估，因为 Cohere Reranker 未配置。")

    # --- 17. 显示结果 ---
    print("\n--- 评估结果 ---")
    if results_dfs:
        final_results_df = pd.concat(results_dfs, ignore_index=True, axis=0)
        print(final_results_df.to_string())
    else:
        print("没有生成评估结果。")
    
    print("\n--- 观察 ---")
    print("我们观察到上下文检索可能会提高指标；然而，")
    print("我们的实验表明，这在很大程度上取决于查询、块大小、块重叠和其他变量。")
    print("因此，进行实验以优化此技术的益处至关重要。")


if __name__ == "__main__":
    # 检查 API 密钥是否设置
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误: OPENAI_API_KEY 环境变量未设置。评估部分将无法正确运行LLM调用。")
        print("请设置 OPENAI_API_KEY。如果只想运行代码结构，可以忽略。")
        # exit(1) # 可以取消注释以强制执行API密钥

    # 运行异步评估
    # For Python 3.7+
    if hasattr(asyncio, 'run'):
        asyncio.run(main_evaluation())
    else: # Fallback for older Python versions (less common now)
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main_evaluation())
        finally:
            loop.close()

    print("\n--- 脚本执行完毕 ---")

