# Building a Multilingual RAG Chatbot for Financial Data Analysis
# 构建多语言 RAG 聊天机器人用于财务数据分析

![Application Screenshot](assets/app_screenshot.png)

## Introduction / 简介

In this article, I'll share how I built a multilingual Retrieval Augmented Generation (RAG) chatbot that specializes in Italian harmonized financial statements (Bilanci Armonizzati). This project combines natural language processing, vector databases, and large language models to create an intelligent assistant capable of answering complex financial questions in both English and Italian.

在本文中，我将分享如何构建一个专注于意大利协调财务报表（Bilanci Armonizzati）的多语言检索增强生成（RAG）聊天机器人。该项目结合了自然语言处理、向量数据库和大型语言模型，创建了一个能够用英语和意大利语回答复杂财务问题的智能助手。

## What is RAG? / 什么是 RAG？

RAG (Retrieval Augmented Generation) is a powerful technique that enhances language models by first retrieving relevant information from a knowledge base, then using that information to generate accurate, contextual responses. Unlike traditional chatbots that rely solely on pre-trained knowledge, RAG systems can access up-to-date, domain-specific information, significantly improving response quality, accuracy, and relevance.

RAG（检索增强生成）是一种强大的技术，它通过首先从知识库中检索相关信息，然后使用该信息生成准确、具有上下文的响应来增强语言模型。与仅依赖预训练知识的传统聊天机器人不同，RAG 系统可以访问最新的、特定领域的信息，显著提高响应质量、准确性和相关性。

## Core Features / 核心功能

### Multilingual Support / 多语言支持

Our chatbot fully supports both English and Italian, making it accessible to a wider audience. The user interface, responses, and even document processing capabilities work seamlessly in both languages.

我们的聊天机器人完全支持英语和意大利语，使其对更广泛的受众可访问。用户界面、响应甚至文档处理功能在两种语言中都能无缝工作。

### Advanced RAG Architecture / 先进的 RAG 架构

The system uses sentence transformers to create vector embeddings of document chunks. When a user asks a question, it's also converted to a vector and compared against the document embeddings to find the most relevant information. This retrieved context is then sent to the LLM along with the original query to generate an informed response.

该系统使用句子转换器为文档块创建向量嵌入。当用户提出问题时，它也会被转换为向量，并与文档嵌入进行比较，以找到最相关的信息。然后将这个检索到的上下文与原始查询一起发送到 LLM，以生成有根据的响应。

```python
# Example of how RAG works in our system:
def get_relevant_context(query, top_k=5):
    # Convert query to vector
    query_embedding = embeddings_model.encode(query)
    
    # Search vector database for similar documents
    context_chunks = vector_db.similarity_search_by_vector(
        query_embedding, k=top_k
    )
    
    # Compile retrieved contexts
    context = "\n\n".join([chunk.page_content for chunk in context_chunks])
    sources = [chunk.metadata["source"] for chunk in context_chunks]
    
    return context, sources
```

### Multiple LLM Options / 多个 LLM 选项

The application supports both cloud-based and local LLMs:
- **Groq API**: Access to powerful models like Llama 3.1 and Gemma2
- **Local Ollama**: Run open-source models locally for privacy and cost savings

应用程序支持基于云的和本地的 LLM：
- **Groq API**：访问强大的模型，如 Llama 3.1 和 Gemma2
- **本地 Ollama**：在本地运行开源模型，以保护隐私和节省成本

### Vector Visualization / 向量可视化

One of the most fascinating features is the ability to visualize document embeddings in 2D space. Users can choose between PCA, t-SNE, or UMAP dimensionality reduction techniques. Each document source is color-coded for easy identification, allowing users to see how similar documents cluster together.

最引人入胜的功能之一是能够在 2D 空间中可视化文档嵌入。用户可以在 PCA、t-SNE 或 UMAP 降维技术之间进行选择。每个文档源都有颜色编码，便于识别，使用户能够看到相似文档如何聚集在一起。

### BDAP API Integration / BDAP API 集成

The application integrates with the Italian BDAP (Banca Dati delle Amministrazioni Pubbliche) APIs to access official financial data for Italian local governments. This allows users to query historical financial information, visualize trends, and analyze structural patterns in government spending.

该应用程序与意大利 BDAP（公共行政数据库）API 集成，以访问意大利地方政府的官方财务数据。这使用户能够查询历史财务信息，可视化趋势，并分析政府支出的结构模式。

## Technical Implementation / 技术实现

### Document Processing Pipeline / 文档处理流程

The system includes a complete pipeline for processing documents:

1. **PDF to Text Conversion**: Extracts text from PDF documents
2. **Text Chunking**: Splits text into manageable chunks with appropriate overlap
3. **Vector Embedding**: Converts text chunks into vector embeddings
4. **Database Storage**: Stores vectors in a SQLite database for efficient retrieval

系统包括完整的文档处理流程：

1. **PDF 到文本转换**：从 PDF 文档中提取文本
2. **文本分块**：将文本分割成具有适当重叠的可管理块
3. **向量嵌入**：将文本块转换为向量嵌入
4. **数据库存储**：将向量存储在 SQLite 数据库中，以便高效检索

```python
def process_text_files():
    # Find all text files
    text_files = list(text_dir.glob("*.txt"))
    
    # Process each text file
    for text_file in text_files:
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, text_file.stem)
        
        # Generate embeddings for all chunks
        embeddings = get_embeddings(chunks, model)
        
        # Store chunks and embeddings in database
        store_in_database(chunks, embeddings)
```

### Streamlit Interface / Streamlit 界面

The application uses Streamlit to create an interactive web interface with multiple tabs:
- **RAG Chat**: The main chat interface
- **Documents**: Upload and manage PDF documents
- **Vector Visualizations**: Interactive 2D visualization of document embeddings
- **Financial Data**: Access to BDAP financial data with visualization tools
- **RAG Settings**: Configuration for chunk size, overlap, and other parameters

应用程序使用 Streamlit 创建具有多个选项卡的交互式 Web 界面：
- **RAG 聊天**：主要聊天界面
- **文档**：上传和管理 PDF 文档
- **向量可视化**：文档嵌入的交互式 2D 可视化
- **财务数据**：访问 BDAP 财务数据，并提供可视化工具
- **RAG 设置**：分块大小、重叠和其他参数的配置

### Web Search Integration / 网络搜索集成

To supplement the internal knowledge base, the application can perform web searches using DuckDuckGo. This allows it to provide up-to-date information even on topics that may not be covered in the document database.

为了补充内部知识库，应用程序可以使用 DuckDuckGo 执行网络搜索。这使它能够提供最新信息，即使是在文档数据库中可能未涵盖的主题。

## Benefits of This Approach / 这种方法的好处

1. **Accuracy**: By retrieving relevant context before generating responses, the chatbot provides more accurate and contextually appropriate answers
2. **Transparency**: Users can see the sources used to generate responses
3. **Customization**: The knowledge base can be easily expanded by adding more documents
4. **Multilingual Capability**: Serve diverse user populations with the same system
5. **Reduced Hallucinations**: The RAG approach significantly reduces the problem of AI hallucinations by grounding responses in retrieved documents

1. **准确性**：通过在生成响应之前检索相关上下文，聊天机器人提供更准确和上下文适当的答案
2. **透明度**：用户可以看到用于生成响应的来源
3. **定制化**：可以通过添加更多文档轻松扩展知识库
4. **多语言能力**：用同一个系统服务不同的用户群体
5. **减少幻觉**：RAG 方法通过将响应建立在检索到的文档基础上，显著减少了 AI 幻觉问题

## Challenges and Solutions / 挑战与解决方案

### Vector Database Schema / 向量数据库模式

One challenge was ensuring compatibility between the database schema used for storing embeddings and the schema expected by the visualization components. We implemented a flexible approach that checks the database structure and adapts accordingly.

一个挑战是确保用于存储嵌入的数据库模式与可视化组件期望的模式之间的兼容性。我们实现了一种灵活的方法，检查数据库结构并相应地进行调整。

### API Reliability / API 可靠性

The BDAP APIs occasionally experience timeouts or temporary unavailability. We implemented robust retry mechanisms with appropriate error handling to ensure a smooth user experience even when API issues occur.

BDAP API 偶尔会遇到超时或临时不可用的情况。我们实现了具有适当错误处理的强大重试机制，以确保即使在发生 API 问题时也能提供流畅的用户体验。

### Visualization Performance / 可视化性能

Visualizing large numbers of embeddings can be computationally intensive. We implemented sampling and pagination strategies to maintain performance while still providing useful insights.

可视化大量嵌入可能会消耗大量计算资源。我们实现了采样和分页策略，以在保持性能的同时仍提供有用的见解。

## Conclusion / 结论

This multilingual RAG chatbot demonstrates the power of combining vector databases, LLMs, and domain-specific data to create intelligent applications. The ability to ground responses in retrieved documents significantly improves accuracy and trustworthiness compared to traditional chatbots.

这个多语言 RAG 聊天机器人展示了结合向量数据库、LLM 和特定领域数据创建智能应用程序的强大功能。与传统聊天机器人相比，将响应建立在检索到的文档基础上的能力显著提高了准确性和可信度。

The project is open-source and available at: [https://github.com/learnbydoingwithsteven/rag_chat_with_ds_streamlit](https://github.com/learnbydoingwithsteven/rag_chat_with_ds_streamlit)

该项目是开源的，可在以下网址获取：[https://github.com/learnbydoingwithsteven/rag_chat_with_ds_streamlit](https://github.com/learnbydoingwithsteven/rag_chat_with_ds_streamlit)

---

### About the Author / 关于作者

I'm a data scientist and machine learning engineer passionate about building AI applications that solve real-world problems. This project combines my interests in natural language processing, financial data analysis, and multilingual applications.

我是一名数据科学家和机器学习工程师，热衷于构建解决现实世界问题的 AI 应用程序。这个项目结合了我对自然语言处理、财务数据分析和多语言应用程序的兴趣。

Feel free to connect with me on GitHub or leave comments below!

欢迎在 GitHub 上与我联系或在下方留言！
