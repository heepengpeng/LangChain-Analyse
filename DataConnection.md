# Data Connection

## 1. 关键特性

许多LLM（Language Model）应用程序需要用户特定的数据，这些数据不是模型训练集的一部分。LangChain为您提供了构建块，通过这些构建块可以加载、转换、存储和查询您的数据：

文档加载器（Document loaders）：从多种不同的来源加载文档。
文档转换器（Document transformers）：将文档拆分、转换为问答格式、去除冗余文档等。
文本嵌入模型（Text embedding models）：将非结构化文本转换为向量
向量存储（Vector stores）：存储和搜索嵌入式数据。
检索器（Retrievers）：对数据进行查询。

![DataConnection](https://python.langchain.com/assets/images/data_connection-c42d68c3d092b85f50d08d4cc171fc25.jpg)


### 1.1 Document Loaders


使用文档加载器从数据源加载数据作为文档（Document）。文档是一段文本及其相关的元数据。例如，有文档加载器可以加载简单的 .txt 文件，可以加载任何网页的文本内容，甚至可以加载YouTube视频的转录文本。

文档加载器提供了一个"load"方法，用于从配置的数据源加载数据作为文档。它们还可以选择实现"lazy load"（延迟加载）功能，以便将数据延迟加载到内存中。

内置了以下loader:

* CSVLoader
* DirectoryLoader
* UnstructuredHTMLLoader
* JSONLoader
* UnstructuredMarkdownLoader
* PyPDFLoader

### 1.2  Document Transformers

一旦您加载了文档，通常您会希望对它们进行转换，以更好地适应您的应用程序。

最简单的例子是，您可能希望将长文档分割成较小的块，以适应您模型的上下文窗口。

LangChain内置了许多文档转换器，使得对文档进行分割、合并、筛选和其他操作变得简单易行。

### 1.3 Text embedding models

嵌入模型可以为文本创建一个向量表示。这非常有用，因为这意味着我们可以在向量空间中处理文本，并进行语义搜索，找到在向量空间中最相似的文本片段。

LangChain中的基本Embeddings类公开了两种方法：

一种用于嵌入文档，另一种用于嵌入查询。前者接受多个文本作为输入，而后者接受单个文本作为输入。将这两种方法分开的原因是某些嵌入模型提供者对于文档（用于搜索）和查询（搜索查询本身）有不同的嵌入方法。

### 1.4 Vector stores

在处理非结构化数据的最常见方法之一是将其嵌入并存储生成的嵌入向量，然后在查询时将非结构化查询进行嵌入，并检索与嵌入查询“最相似”的嵌入向量。向量存储负责存储嵌入数据并为您执行向量搜索。

![vector_stores](https://python.langchain.com/assets/images/vector_stores-9dc1ecb68c4cb446df110764c9cc07e0.jpg)

### 1.5 Retrievers

检索器（Retriever）是一个接口，它根据非结构化的查询返回文档。它比向量存储更通用。检索器不需要能够存储文档，只需要能够返回（或检索）文档。


## 2. 代码模块

```
document_loaders/                       # 加载文档模块
    ├── text.py                         # 读取文本文件，构造出Document类，包含两个字段page_content, metadata, page_content 存放文本内容，metadata存放文本来源文件路径。
    ├── csv_loader.py                   # 读取CSV文件
    ├── ......                          # 可扩展
text_splitter.py                        # 文本分割实现模块，包括TextSplitter，CharacterTextSplitter，MarkdownHeaderTextSplitter等
embeddings/                             # 嵌入模块
    ├── openai.py                       # 定义了OpenAIEmbeddings类，对openai的embedding api 进行封装
    ├──......                           # 可扩展，提供其他embedding 实现    
vectorstores/
    ├── chroma.py                       # 定义了chromadb向量数据库的接口
    ├──......                           # 可扩展，提供其他向量数据库接口    
retrievers/                             # retrievers 模块， 提供多种Retriever
    ├──wikipedia.py                     # 提供wikipedia的Retriever实现
    ├──......                           # 可扩展，提供其他数据源的Retriever实现
schema/
    ├── retriever.py                    # 文档检索系统的抽象基类。检索系统被定义为能够接受字符串查询并从某个来源返回最相关的文档。
```


## 3. 代码逻辑