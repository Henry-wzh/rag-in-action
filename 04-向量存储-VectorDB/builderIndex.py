from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# 加载数据
documents = SimpleDirectoryReader(input_dir="90-文档-Data", recursive=True).load_data()
print(len(documents), "--"*10)
# 构建索引
index = VectorStoreIndex.from_documents(documents)
print(vars(index))

# 查看Nodes
nodes = index.index_struct.nodes_dict
for node in nodes:
    print(node)


# LlamaIndex自动的切分Nodes
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
nodes = text_splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes) # 从nodes中生成Index
nodes = index.index_struct.nodes_dict
print(len(nodes))
for node in nodes:
    print(node)

# 保存索引到磁盘
index.storage_context.persist(persist_dir="saved_index")