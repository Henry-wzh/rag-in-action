#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
上下文检索（Contextual Retrieval）与Milvus实现
基于Anthropic提出的方法，解决传统RAG中语义隔离问题
"""

# 导入必要的库
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction

from typing import List, Dict, Any
from typing import Callable
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
import json
import anthropic
import os
import dotenv
dotenv.load_dotenv()

class MilvusContextualRetriever:
    """
    Milvus上下文检索器类
    
    支持标准检索、混合检索、上下文检索和重排序功能
    """
    def __init__(
        self,
        uri="milvus.db",
        collection_name="contexual_bgem3",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        anthropic_client=None,
        use_reranker=False,
        rerank_function=None,
    ):
        """
        初始化检索器
        
        参数:
            uri: Milvus服务地址
            collection_name: 集合名称
            dense_embedding_function: 密集向量嵌入函数
            use_sparse: 是否使用稀疏向量
            sparse_embedding_function: 稀疏向量嵌入函数
            use_contextualize_embedding: 是否使用上下文嵌入
            anthropic_client: Anthropic客户端
            use_reranker: 是否使用重排序
            rerank_function: 重排序函数
        """
        self.collection_name = collection_name

        # 对于Milvus-lite，uri是本地路径，如"./milvus.db"
        # 对于Milvus独立服务，uri类似"http://localhost:19530"
        # 对于Zilliz Cloud，请设置`uri`和`token`
        self.client = MilvusClient(uri)

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        self.anthropic_client = anthropic_client

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif use_sparse is True and sparse_embedding_function is None:
            raise ValueError(
                "稀疏嵌入函数不能为空，如果use_sparse为True"
            )
        else:
            pass

    def build_collection(self):
        """构建Milvus集合"""
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,
        )
        if self.use_sparse is True:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        if self.use_sparse is True:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        """
        插入数据到Milvus
        
        参数:
            chunk: 文本块内容
            metadata: 元数据
        """
        dense_vec = self.embedding_function([chunk])[0]
        data = {
            "dense_vector": dense_vec,
            **metadata
        }
        
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def insert_contextualized_data(self, doc_content, chunk_content, metadata):
        """
        插入上下文化的数据
        
        参数:
            doc_content: 整个文档内容
            chunk_content: 当前文本块内容
            metadata: 元数据
        """
        # 通过LLM处理，将整个文档内容作为上下文添加到每个块之前
        prompt = f"""
        <文档>
        {doc_content}
        </文档>
        <块>
        {chunk_content}
        </块>
        
        我需要你对上述<块>进行丰富，使用<文档>中的内容提供背景和上下文信息。
        你的回答应该包含<块>的完整内容，并确保语义连贯。只返回丰富后的文本内容，不要添加任何说明或解释。
        """
        
        message = self.anthropic_client.messages.create( 
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        contextualized_chunk = message.content[0].text.strip()
        
        # 使用上下文化后的内容生成嵌入并插入
        dense_vec = self.embedding_function([contextualized_chunk])[0]
        data = {
            "dense_vector": dense_vec,
            "contextualized_content": contextualized_chunk,
            **metadata
        }
        
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([contextualized_chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def search(self, query, k=5):
        """
        搜索相关内容
        
        参数:
            query: 查询文本
            k: 返回结果数量
        
        返回:
            搜索结果列表
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # 生成查询的嵌入向量
        dense_vec = self.embedding_function([query])[0]
        
        # 执行标准密集向量搜索
        res = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vec],
            limit=k,
            output_fields=["content", "contextualized_content"],
            search_params=search_params,
        )
        
        # 使用重排序器进一步优化结果
        if self.use_reranker:
            # 提取文档内容
            docs = []
            for hit in res[0]:
                # 使用上下文化内容（如果存在）或原始内容
                content = hit["entity"].get("contextualized_content", hit["entity"].get("content", ""))
                docs.append(content)
            
            # 应用重排序
            rerank_results = self.rerank_function(query, docs)
            
            # 根据重排序结果重新排序原始结果
            reranked_results = []
            for result in rerank_results:
                idx = result.index  # 使用 .index 属性而不是字典访问
                reranked_results.append(res[0][idx])
            
            res = [reranked_results]
        
        return res


def evaluate_retrieval(eval_data, retrieval_function, db, k=5):
    """
    评估检索性能
    
    参数:
        eval_data: 评估数据集
        retrieval_function: 检索函数
        db: 数据库实例
        k: 评估的top-k结果数
        
    返回:
        评估结果
    """
    total_score = 0
    total_queries = 0
    
    for item in tqdm(eval_data, desc="Evaluating retrieval"):
        total_queries += 1
        query = item["query"]
        
        # 获取黄金标准内容
        golden_contents = []
        for ref in item["references"]:
            doc_uuid = ref["doc_uuid"]
            chunk_index = ref["chunk_index"]
            
            # 查找对应的原始文档和块
            golden_doc = next(
                (
                    doc
                    for doc in dataset
                    if doc.get("original_uuid") == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                print(f"警告：未找到UUID为{doc_uuid}的黄金文档")
                continue
                
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["original_index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                print(f"警告：在文档{doc_uuid}中未找到索引为{chunk_index}的黄金块")
                continue
                
            golden_contents.append(golden_chunk["content"].strip())
            
        if not golden_contents:
            print(f"警告：未找到查询的黄金内容：{query}")
            continue
            
        # 使用检索函数获取检索结果
        retrieved_docs = retrieval_function(query, db, k=k)
        
        # 计算有多少黄金块在top-k检索文档中
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:
                content_field = "content"
                if "contextualized_content" in doc["entity"]:
                    # 跳过上下文内容，仅比较原始内容
                    content_field = "content"
                retrieved_content = doc["entity"][content_field].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break
                    
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
        
    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries,
    }


def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    """基础检索函数"""
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件并返回字典列表"""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db, original_jsonl_path: str, k):
    """评估数据库的检索性能"""
    # 加载原始JSONL数据作为查询和真实标签
    original_data = load_jsonl(original_jsonl_path)
    
    # 评估检索
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"总分: {results['average_score']}")
    print(f"总查询数: {results['total_queries']}")
    
    return results


def download_data():
    """下载示例数据"""
    import urllib.request
    
    # 检查文件是否已存在
    if not os.path.exists("codebase_chunks.json"):
        print("下载codebase_chunks.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
            "codebase_chunks.json"
        )
    
    if not os.path.exists("evaluation_set.jsonl"):
        print("下载evaluation_set.jsonl...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
            "evaluation_set.jsonl"
        )
    
    print("数据下载完成！")


def main():
    """主函数 - 运行所有实验"""
    # 替换这些为你的实际API密钥
    cohere_api_key = os.getenv("COHERE_API_KEY")
    anthropic_api_key = os.getenv("CLAUDE_API_KEY")
    
    # 下载数据
    download_data()
    
    # 加载数据集
    global dataset
    with open("codebase_chunks.json", "r") as f:
        dataset = json.load(f)
    
    # 只使用前5个文档进行测试
    dataset = dataset[:5]
    
    # 初始化模型
    dense_ef = SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh')  # 使用中文优化的模型
    cohere_rf = CohereRerankFunction(api_key=cohere_api_key)
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    # 实验一：标准检索
    print("\n===== 实验一：标准检索 =====")
    standard_retriever = MilvusContextualRetriever(
        uri="standard.db", 
        collection_name="standard", 
        dense_embedding_function=dense_ef
    )
    
    standard_retriever.build_collection()
    for doc in tqdm(dataset, desc="插入标准检索数据"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)
    
    # 创建简单的评估数据
    eval_data = []
    for doc in dataset[:2]:  # 只使用前2个文档进行评估
        for chunk in doc["chunks"][:2]:  # 每个文档只取前2个块
            eval_data.append({
                "query": chunk["content"][:50],  # 使用块内容的前50个字符作为查询
                "references": [{
                    "doc_uuid": doc["original_uuid"],
                    "chunk_index": chunk["original_index"]
                }]
            })
    
    # 保存评估数据
    with open("evaluation_set.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    standard_results = evaluate_db(standard_retriever, "evaluation_set.jsonl", 5)
    
    # 实验二：上下文检索
    print("\n===== 实验二：上下文检索 =====")
    contextual_retriever = MilvusContextualRetriever(
        uri="contextual.db",
        collection_name="contextual",
        dense_embedding_function=dense_ef,
        use_contextualize_embedding=True,
        anthropic_client=anthropic_client,
    )
    
    contextual_retriever.build_collection()
    for doc in tqdm(dataset, desc="插入上下文检索数据"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            contextual_retriever.insert_contextualized_data(
                doc_content, chunk_content, metadata
            )
    
    contextual_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # 实验三：带重排序的上下文检索
    print("\n===== 实验三：带重排序的上下文检索 =====")
    contextual_retriever.use_reranker = True
    contextual_retriever.rerank_function = cohere_rf
    
    reranker_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # 打印所有结果比较
    print("\n===== 所有实验结果比较 =====")
    print(f"标准检索 Pass@5: {standard_results['pass_at_n']:.2f}%")
    print(f"上下文检索 Pass@5: {contextual_results['pass_at_n']:.2f}%")
    print(f"带重排序的上下文检索 Pass@5: {reranker_results['pass_at_n']:.2f}%")


if __name__ == "__main__":
    main()
