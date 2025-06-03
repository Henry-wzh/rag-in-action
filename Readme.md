# RAG (Retrieval-Augmented Generation) 系统研发实战

本项目是一个基于 DeepSeek 的 RAG 系统[实战课程](https://u.geekbang.org/subject/airag/1009927)的代码仓库，实现了一个完整的检索增强生成系统。

课程地址：[RAG系统研发实战](https://u.geekbang.org/subject/airag/1009927)

![基于DeepSeek的RAG系统研发实战课程架构图](92-图片-Pic/RAG.PNG)

## 项目架构

项目采用模块化设计，每个模块负责 RAG 系统的不同方面：

- `00-简单RAG-SimpleRAG`: 基础 RAG 系统实现
- `01-数据导入-DataLoading`: 数据加载和预处理
- `02-文本切块-DocChunking`: 文档分块策略
- `03-向量嵌入-Embedding`: 文本向量化
- `04-向量存储-VectorDB`: 向量数据库操作
- `05-检索前处理-PreRetrieval`: 检索优化
- `06-索引优化-Indexing`: 索引构建和优化
- `07-检索后处理-PostRetrieval`: 检索结果优化
- `08-响应生成-Generation`: 答案生成
- `09-系统评估-Evaluation`: 系统性能评估
- `10-高级RAG-AdvanceRAG`: 高级 RAG 技术实现

## 环境要求

### 硬件要求

#### GPU 版本
- NVIDIA GPU (建议 >= 8GB 显存)
- CUDA 11.8 或更高版本
- cuDNN 8.0 或更高版本

#### CPU 版本
- 建议 >= 16GB RAM
- 多核处理器（建议 >= 4 核）

### 软件要求

#### 操作系统支持
1. Ubuntu (推荐 22.04 或更高)
   - GPU 版本：使用 `requirements_langchain_20250413(Ubuntu-with-GPU).txt`
   - CPU 版本：使用 `requirements_langchain(Ubuntu-with-CPU).txt`

2. MacOS/Windows
   - 使用 `requirements_langchain_无GPU版(Mac,Win).txt`

### 框架选择

1. LangChain 框架
   - 基础版：`requirements_langchain_简单RAG(后续模块还要安装其它包).txt`
   - 完整版（GPU）：`requirements_langchain_20250413(Ubuntu-with-GPU).txt`
   - 完整版（CPU）：`requirements_langchain_无GPU版(Mac,Win).txt`

2. LlamaIndex 框架
   - 基础版：`requirements_llamaindex_简单RAG(后续模块还要安装其它包).txt`
   - 完整版（GPU）：`requirements_llamaindex_20250413(Ubuntu-with-GPU).txt`
   - 完整版（CPU）：`requirements_llamaindex_无GPU版(Mac,Win).txt`

## 环境配置

### Ubuntu (GPU 版本)

```bash
# 创建虚拟环境
python -m venv venv-rag-langchain
source venv-rag-langchain/bin/activate

# 安装依赖
pip install -r 91-环境-Environment/requirements_langchain_20250413\(Ubuntu-with-GPU\).txt
```

### Ubuntu (CPU 版本)

```bash
python -m venv venv-rag-langchain
source venv-rag-langchain/bin/activate
pip install -r 91-环境-Environment/requirements_langchain\(Ubuntu-with-CPU\).txt
```

### MacOS/Windows

```bash
python -m venv venv-rag-langchain
# Windows
.\venv-rag-langchain\Scripts\activate
# MacOS
source venv-rag-langchain/bin/activate

pip install -r "91-环境-Environment/requirements_langchain_无GPU版(Mac,Win).txt"
```

## 特殊依赖说明

1. PDF 处理相关：
   - 使用 `requirements_camelot_20250413.txt` 安装 PDF 处理相关依赖
   - 可能需要额外安装系统级依赖：
     - Ubuntu: `sudo apt-get install ghostscript python3-tk`
     - MacOS: `brew install ghostscript tcl-tk`
     - Windows: 需要手动安装 Ghostscript

2. 标注工具相关：
   - 使用 `requirements_marker_20250413.txt` 安装标注工具相关依赖

## 使用说明

1. 选择合适的环境配置文件并安装依赖
2. 按照模块顺序逐步学习和实践
3. 每个模块都包含独立的示例和说明文档
4. 建议先从 `00-简单RAG-SimpleRAG` 开始，逐步深入

## 注意事项

1. GPU 版本需要确保 CUDA 环境配置正确
2. 不同操作系统可能需要额外的系统级依赖
3. 建议使用虚拟环境管理依赖
4. 部分模块可能需要额外的模型下载或 API 密钥配置

## 常见问题

1. CUDA 相关错误：检查 NVIDIA 驱动和 CUDA 版本是否匹配
2. 内存不足：调整批处理大小或使用 CPU 版本
3. 依赖冲突：使用虚拟环境并严格按照 requirements 文件安装

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

[![佳哥项目 Star History 趋势](https://api.star-history.com/svg?repos=huangjia2019/langchain-in-action,huangjia2019/ai-agents,huangjia2019/let-us-machine-learning,huangjia2019/rag-in-action,huangjia2019/llm-gpt&type=Date)](https://www.star-history.com/#huangjia2019/langchain-in-action&huangjia2019/ai-agents&huangjia2019/let-us-machine-learning&huangjia2019/rag-in-action&huangjia2019/llm-gpt&Date)

## 许可证

本项目采用 MIT 许可证。