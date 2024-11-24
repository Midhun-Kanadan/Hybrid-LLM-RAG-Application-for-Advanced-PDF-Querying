# Import necessary libraries
import os
import logging
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import generate as ollama_generate, list as list_ollama_models
import openai
from dotenv import load_dotenv
import re

import openai
from openai import OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Application: OpenAI and Ollama Integration 📘",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper function to sanitize collection names
def sanitize_collection_name(name):
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = sanitized[:63]
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    return sanitized or "default_collection"

# Cache the PDF chunks
@st.cache_data
def read_and_chunk_pdf(file_path):
    """
    Read and chunk the PDF, then return chunks with progress.
    """
    from pypdf import PdfReader

    # Read the PDF
    reader = PdfReader(file_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]  # Remove empty pages

    # Combine text into a single string
    full_text = "\n\n".join(pdf_texts)

    # Initialize the text splitter
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )

    # Initialize the progress bar
    progress_bar = st.progress(0)
    total_length = len(full_text)

    # Chunk the text with progress updates
    chunks = []
    for i, chunk in enumerate(character_splitter.split_text(full_text)):
        chunks.append(chunk)
        progress_bar.progress((i + 1) / total_length)

    progress_bar.empty()  # Remove the progress bar when done
    return chunks

# Cache the ChromaDB client as a resource
@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

# Function to create and load ChromaDB collection
@st.cache_resource
def create_and_load_chroma_collection(_chroma_client, chunks, file_name):
    collection_name = sanitize_collection_name(os.path.splitext(file_name)[0])
    existing_collections = [col.name for col in _chroma_client.list_collections()]
    
    if collection_name in existing_collections:
        _chroma_client.delete_collection(name=collection_name)
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = _chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
    
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    return collection

# Function to query ChromaDB
def query_chroma(collection, query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results, include=['documents'])
    return results

# Function to chat with OpenAI model
def chat_with_openai(prompt, retrieved_documents=None, model="gpt-4o-mini"):
    try:
        information = "\n\n".join(retrieved_documents)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
                "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
            },
            {"role": "user", "content": f"Question: {prompt}. \n Information: {information}"}
        ]
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return f"Error: {e}"

# Function to chat with Ollama model
def chat_with_ollama(model_name, prompt, retrieved_documents=None):
    """
    Generate a response from the selected Ollama model, including context from retrieved documents.
    """
    try:
        # Format the context for the prompt
        if retrieved_documents:
            context = "\n\n".join(retrieved_documents)
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer the question using only the provided context."
        else:
            full_prompt = prompt

        # Generate response from Ollama
        response = ollama_generate(model=model_name, prompt=full_prompt)
        return response.get("response", "No response generated.")
    except Exception as e:
        return f"Error: {e}"

# Function to fetch Ollama models
def get_ollama_models():
    try:
        models_info = list_ollama_models()
        return [model.model for model in models_info.models]
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

# Sidebar configuration
def sidebar():
    st.sidebar.title("🔧 Configuration")
    st.sidebar.markdown("---")
    model_type = st.sidebar.selectbox("Choose the LLM type:", ["OpenAI", "Ollama"], index=0)
    
    selected_model = None
    if model_type == "Ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            # Check if "llama3.2:latest" is in the list and set it as the default if available
            default_index = ollama_models.index("llama3.2:latest") if "llama3.2:latest" in ollama_models else 0
            selected_model = st.sidebar.selectbox("Available Ollama Models", options=ollama_models, index=default_index)
        else:
            st.sidebar.warning("No Ollama models available.")
    elif model_type == "OpenAI":
        selected_model = "gpt-4o-mini"
    
    st.sidebar.markdown("---")
    file_upload = st.sidebar.file_uploader("📁 Upload a PDF file", type=["pdf"])
    
    return model_type, selected_model, file_upload

# Main function
def main():
    st.title("📘 OpenAI and Ollama RAG Playground")
    st.markdown("Welcome to the **RAG Application** leveraging **OpenAI** and **Ollama** models. Upload a PDF and ask questions to get answers based on the document content.")
    st.markdown("---")

    # Sidebar setup
    model_type, selected_model, file_upload = sidebar()
    
    st.sidebar.info(f"**Selected Model Type:** {model_type}")
    if selected_model:
        st.sidebar.info(f"**Selected Model:** {selected_model}")
    
    if file_upload:
        with st.spinner("Reading and processing the PDF..."):
            chunks = read_and_chunk_pdf(file_upload)
        
        st.success("PDF processed into chunks.")
        
        with st.spinner("Creating embeddings and building vector database..."):
            chroma_client = get_chroma_client()
            chroma_collection = create_and_load_chroma_collection(chroma_client, chunks, file_upload.name)
        
        st.success("Vector database created successfully.")
        
        st.markdown("### 🔍 Ask a Question")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Retrieving relevant documents..."):
                retrieval_results = query_chroma(chroma_collection, query)
                retrieved_docs = retrieval_results['documents'][0]
            
            with st.spinner("Generating response..."):
                if model_type == "OpenAI":
                    response = chat_with_openai(query, retrieved_docs, model=selected_model)
                elif model_type == "Ollama" and selected_model:
                    response = chat_with_ollama(selected_model, query, retrieved_documents=retrieved_docs)
                else:
                    response = "No valid model selected."

            st.markdown("### 📄 Answer")
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {response}")

            with st.expander("🔎 Retrieved Documents"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document {idx+1}:**")
                    st.write(doc)
    else:
        st.info("Please upload a PDF file to begin.")

if __name__ == "__main__":
    main()
