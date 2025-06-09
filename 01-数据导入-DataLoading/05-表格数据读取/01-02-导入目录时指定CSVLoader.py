from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader

loader = DirectoryLoader(
    path="90-文档-Data/黑悟空",  # Specify the directory containing your CSV files
    glob="**/*.csv",                # Use a glob pattern to match CSV files
    loader_cls=CSVLoader            # Specify CSVLoader as the loader class
)

docs = loader.load()
print(f"文档数：{len(docs)}")  # 输出文档总数
print(docs[0])
