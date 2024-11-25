# Hybrid-LLM-RAG-Application-for-Advanced-PDF-Querying

![RAG Application UI](RAG_Application.png)
---
![Embedding_Visualization](Embedding_Visualization.png)

## 📘 Overview

**Hybrid-LLM-RAG-Application-for-Advanced-PDF-Querying** is a powerful Retrieval-Augmented Generation (RAG) system designed to leverage **OpenAI** and **Ollama** language models for advanced querying of PDF documents. This application processes PDF files, generates embeddings, stores them in a vector database, and answers user queries based on document content using state-of-the-art language models.

---

## 🚀 Features

- **Hybrid LLM Support**: Switch seamlessly between OpenAI and Ollama models.
- **PDF Chunking and Querying**: Automatically processes PDF files into chunks for efficient retrieval.
- **Dynamic OpenAI API Configuration**: Load API keys from a `.env` file or input them directly in the application.
- **Interactive UI**: User-friendly interface built with Streamlit.
- **Real-Time Progress Bar**: Displays step-by-step progress for processing, embedding, and querying.
- **Customizable RAG Settings**: Easily configure the LLM type and model through a sidebar.
- **Embedding Visualization**: Leverage UMAP to visually project query and retrieved embeddings in 2D space, with interactive tooltips displaying document chunks and query text.

---