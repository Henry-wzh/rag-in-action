import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 模型
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # 或其他模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 创建 Deepseek LLM
llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=0.7,        # 控制输出的随机性
    max_tokens=2048,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)
# 加载 PDF 文件
loader = UnstructuredPDFLoader("data/PDF/uber_10q_march_2022.pdf")
documents = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = FAISS.from_documents(chunks, embeddings)

# 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    verbose=True
)

# 查询示例
query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"
query = "how many COVID-19 response initiatives in year 2021?"
response = qa_chain.invoke({"query": query})

print("\n************LangChain Query Response************")
print("Answer:", response["result"])
print("\nSource Documents:")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"\nDocument {i}:")
    print(doc.page_content[:200] + "...")