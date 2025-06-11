import os
import pdfplumber
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.llms.deepseek import DeepSeek
from llama_index.core import Document
from typing import List
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


pdf_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"

# 打开 PDF 并解析表格
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# 转换所有表格为 DataFrame 并构建文档
documents: List[Document] = []
if tables:
    # 遍历所有表格
    for i, table in enumerate(tables, 1):
        # 将表格转换为 DataFrame
        df = pd.DataFrame(table)
        
        # 保存到CSV文件
        # csv_filename = f"billionaires_table_{i}.csv"
        # df.to_csv(csv_filename, index=False)
        # print(f"\n表格 {i} 数据已保存到 {csv_filename}")
        
        # 将DataFrame转换为文本
        text = df.to_string()
        
        # 创建Document对象
        doc = Document(text=text, metadata={"source": f"表格{i}"})
        documents.append(doc)

# 创建查询引擎
# 创建 Deepseek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1,  # 较低的温度使输出更加确定
    top_p=0.9,             # 控制随机性的另一种方式
    request_timeout=120,   # API 请求超时时间（秒）
    max_tokens=1024,  # 生成回复的token数
)
# llm = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com/v1",  # DeepSeek API地址
#     model="deepseek-chat",
# )
# 设置全局LLM
Settings.llm = llm
# 使用本地 HuggingFace 嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # 或其他模型
)
# 构建索引
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 示例问答
questions = [
    "2023年谁是最富有的人?",
    "最年轻的富豪是谁?"
]

print("\n===== 问答演示 =====")
for question in questions:
    response = query_engine.query(question)
    print(f"\n问题: {question}")
    print(f"回答: {response}")