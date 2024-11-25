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
import time  # To simulate progress updates
import umap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import textwrap
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")



import openai
from openai import OpenAI
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# default_openai_api_key = os.getenv("OPENAI_API_KEY")
# Ensure the .env file is loaded ONLY if it exists
if os.path.exists(".env"):
    load_dotenv(override=True)  # Forcefully override any system environment variables
    default_openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    default_openai_api_key = None
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
    results = collection.query(query_texts=[query], n_results=n_results, include=['documents', 'embeddings'])
    return results


def chat_with_openai(prompt, retrieved_documents=None, model="gpt-4o-mini"):
    """
    Generate a response from the OpenAI model using retrieved context from the document.
    """
    try:
        information = "\n\n".join(retrieved_documents)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an advanced AI assistant designed to answer user questions based on the provided context. "
                    "You will be shown the user's question along with relevant information extracted from a document. "
                    "Provide accurate and concise answers using only the provided context. "
                    "If the context is insufficient to answer the question, respond with 'The information is not available in the provided document.'"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {prompt}. \n Information: {information}"
            }
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return f"Error: {e}"


def chat_with_ollama(model_name, prompt, retrieved_documents=None):
    """
    Generate a response from the selected Ollama model, including context from retrieved documents.
    """
    try:
        # Format the context for the prompt
        if retrieved_documents:
            context = "\n\n".join(retrieved_documents)
            full_prompt = (
                f"Context: {context}\n\n"
                f"Question: {prompt}\n\n"
                f"Answer the question using only the provided context. "
                f"If the information is insufficient, respond with: 'The information is not available in the provided document.'"
            )
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

# Helper CSS to adjust spacing and font size
st.markdown(
    """
    <style>
        /* Reduce spacing above headers */
        .css-18e3th9 {
            padding-top: 0.5rem;
            padding-bottom: 0rem;
        }
        
        /* Adjust overall font size */
        html, body, .block-container {
            font-size: 14px;
        }

        /* Reduce space between sidebar elements */
        section[data-testid="stSidebar"] .css-1d391kg {
            margin-bottom: 0.5rem;
        }
        /* Hide spinner log outputs like Running create_and_load_chroma_collection */
        .stSpinner > div > div {
            display: none;
        }

        /* Adjust the height of processing steps to make chat area more visible */
        .processing-box {
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



def sidebar():
    st.sidebar.image(
        "logo.png", 
        width=200,  # Set the width to control the size (adjust as needed)
        # caption="RAG Application", 
        output_format="PNG"
    )
    # Compact Sidebar Configuration
    # st.sidebar.title("🔧 Configuration")
    # st.sidebar.markdown("---")
    
    # Compact Model Type Dropdown
    model_type = st.sidebar.selectbox(
        "LLM Type:",
        ["Ollama", "OpenAI"],
        index=0,
        help="Select the language model type"
    )
    # OpenAI API Key Input
    openai_api_key = default_openai_api_key
    if model_type == "OpenAI":
        st.sidebar.markdown("### OpenAI Configuration")
        if not default_openai_api_key:
            openai_api_key = st.sidebar.text_input(
                "Enter OpenAI API Key",
                type="password",
                help="Required to use OpenAI models. This will override the default key.",
            )
        else:
            st.sidebar.success("API key loaded from .env file.")

    # Compact Model Selection
    selected_model = None
    if model_type == "Ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            default_index = ollama_models.index("llama3.2:latest") if "llama3.2:latest" in ollama_models else 0
            selected_model = st.sidebar.selectbox(
                "Model:",
                options=ollama_models,
                index=default_index,
                help="Select a specific Ollama model"
            )
        else:
            st.sidebar.warning("No Ollama models available.")
    elif model_type == "OpenAI":
        selected_model = "gpt-4o-mini"

    # Compact File Uploader
    st.sidebar.markdown("### Upload PDF")
    file_upload = st.sidebar.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Drag and drop a PDF file or browse files",
    )
    # Add a horizontal separator and author information
    st.sidebar.markdown("---")
    # Enhanced author details with an avatar and description
    st.sidebar.image(
        "Midhun.png",  # Replace with the actual path to your avatar image
        width=100,  # Adjust size of avatar image
    )
    st.sidebar.markdown(
        """ 
        **[Midhun Kanadan](https://www.linkedin.com/in/midhunkanadan/)**  
        *ML and AI Enthusiast*  
        """,
        unsafe_allow_html=True,
    )
    
    return model_type, selected_model, file_upload, openai_api_key

# Function to project embeddings using UMAP
def project_embeddings(embeddings, umap_transform):
    """
    Projects embeddings into 2D space using a pre-fitted UMAP transform.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings, desc="Projecting embeddings")):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

# Function to wrap long text with HTML <br> for hover
def wrap_text_with_html(text, width=50):
    return "<br>".join(textwrap.wrap(text, width=width))

# Function to visualize embeddings using Plotly
# def visualize_embeddings_plotly(query, query_embedding, dataset_embeddings, retrieved_embeddings, chunks, retrieved_documents):
#     """
#     Visualizes embeddings using UMAP projection and Plotly.
#     """
#     # Project all embeddings
#     umap_transform = umap.UMAP(n_neighbors=15, n_components=2, random_state=42).fit(dataset_embeddings)
#     projected_dataset_embeddings = project_embeddings(dataset_embeddings, umap_transform)
#     projected_query_embedding = project_embeddings([query_embedding], umap_transform)
#     projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

#     # Prepare data for Plotly visualization
#     dataset_points = {
#         "type": ["Dataset Chunks"] * len(projected_dataset_embeddings),
#         "x": projected_dataset_embeddings[:, 0],
#         "y": projected_dataset_embeddings[:, 1],
#         "text": [wrap_text_with_html(chunk, width=50) for chunk in chunks],  # Dataset chunks
#     }

#     query_point = {
#         "type": ["Query"],
#         "x": projected_query_embedding[:, 0],
#         "y": projected_query_embedding[:, 1],
#         "text": [wrap_text_with_html(query, width=50)],  # Query text
#     }

#     retrieved_points = {
#         "type": ["Retrieved Chunks"] * len(projected_retrieved_embeddings),
#         "x": projected_retrieved_embeddings[:, 0],
#         "y": projected_retrieved_embeddings[:, 1],
#         "text": [wrap_text_with_html(doc, width=50) for doc in retrieved_documents],  # Retrieved documents
#     }

#     # Combine all data
#     data = pd.DataFrame(dataset_points)
#     data = pd.concat([data, pd.DataFrame(query_point), pd.DataFrame(retrieved_points)], ignore_index=True)

#     # Create a Plotly scatter plot
#     fig = px.scatter(
#         data,
#         x="x",
#         y="y",
#         color="type",
#         hover_data={"text": True, "x": False, "y": False, "type": False},  # Show only text on hover
#         title="UMAP Projection of Query and Retrieved Chunks",
#     )
#     fig.update_traces(marker=dict(size=10, opacity=0.8))  # Adjust marker size and transparency
#     fig.update_layout(title_font_size=18, legend=dict(font=dict(size=12)))
#     return fig
def visualize_embeddings_plotly(query, query_embedding, dataset_embeddings, retrieved_embeddings, chunks, retrieved_documents):
    """
    Visualizes embeddings using UMAP projection and Plotly with improved colors and readability.
    """
    # Project all embeddings
    umap_transform = umap.UMAP(n_neighbors=15, n_components=2, random_state=42).fit(dataset_embeddings)
    projected_dataset_embeddings = project_embeddings(dataset_embeddings, umap_transform)
    projected_query_embedding = project_embeddings([query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

    # Prepare data for Plotly visualization
    dataset_points = {
        "type": ["Dataset Chunks"] * len(projected_dataset_embeddings),
        "x": projected_dataset_embeddings[:, 0],
        "y": projected_dataset_embeddings[:, 1],
        "text": [wrap_text_with_html(chunk, width=50) for chunk in chunks],  # Dataset chunks
    }

    query_point = {
        "type": ["Query"],
        "x": projected_query_embedding[:, 0],
        "y": projected_query_embedding[:, 1],
        "text": [wrap_text_with_html(query, width=50)],  # Query text
    }

    retrieved_points = {
        "type": ["Retrieved Chunks"] * len(projected_retrieved_embeddings),
        "x": projected_retrieved_embeddings[:, 0],
        "y": projected_retrieved_embeddings[:, 1],
        "text": [wrap_text_with_html(doc, width=50) for doc in retrieved_documents],  # Retrieved documents
    }

    # Combine all data
    data = pd.DataFrame(dataset_points)
    data = pd.concat([data, pd.DataFrame(query_point), pd.DataFrame(retrieved_points)], ignore_index=True)

    # Define colors for better visibility
    color_map = {
        "Dataset Chunks": "#7f8c8d",  # Gray
        "Query": "#e74c3c",           # Bright Red
        "Retrieved Chunks": "#2ecc71", # Bright Green
    }

    # Create a Plotly scatter plot
    fig = px.scatter(
        data,
        x="x",
        y="y",
        color="type",
        hover_data={"text": True, "x": False, "y": False, "type": False},  # Show only text on hover
        title="UMAP Projection of Query and Retrieved Chunks",
        color_discrete_map=color_map,  # Use custom colors
    )
    fig.update_traces(marker=dict(size=10, opacity=0.9))  # Adjust marker size and transparency
    fig.update_layout(title_font_size=18, legend=dict(font=dict(size=12)))
    return fig


# Function to visualize embeddings
def visualize_embeddings(query, query_embedding, dataset_embeddings, retrieved_embeddings, umap_transform):
    """
    Visualizes embeddings using UMAP projection.
    """
    projected_dataset_embeddings = umap_transform.fit_transform(dataset_embeddings)
    projected_query_embedding = umap_transform.transform([query_embedding])
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

    # Plot the embeddings
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
        label="Dataset Chunks"
    )
    ax.scatter(
        projected_query_embedding[:, 0],
        projected_query_embedding[:, 1],
        s=150,
        marker="X",
        color="red",
        label="Query"
    )
    ax.scatter(
        projected_retrieved_embeddings[:, 0],
        projected_retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="green",
        label="Retrieved Chunks"
    )
    ax.set_title(f"UMAP Projection of Query and Retrieved Chunks", fontsize=14)
    ax.axis("off")
    ax.legend()
    st.pyplot(fig)

# Updated main function with progress bar
def main():
    # Title and Header
    st.title("📘 LLM-RAG-Application-for-Advanced-PDF-Querying")
    st.markdown(
        """
        Welcome to the **RAG Application** leveraging **Ollama** and **OpenAI** models. 
        Upload a PDF and ask questions to get answers based on the document content.
        """
    )
    st.markdown("---")

    # Sidebar setup
    # model_type, selected_model, file_upload = sidebar()
    model_type, selected_model, file_upload, openai_api_key = sidebar()

    # Set OpenAI API Key if provided
    if model_type == "OpenAI" and openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.success("OpenAI API Key is set.")
    elif model_type == "OpenAI" and not openai_api_key:
        st.sidebar.error("Please provide a valid OpenAI API Key.")

    # Main Workspace
    if file_upload:
        st.markdown("### Processing Steps")
        
        # Initialize progress bar
        progress_bar = st.progress(0)  # Starts at 0%

        # Step 1: Read and chunk PDF
        with st.spinner("📖 Reading and processing the PDF..."):
            try:
                time.sleep(1)  # Simulate processing time
                chunks = read_and_chunk_pdf(file_upload)
                progress_bar.progress(33)  # Update progress to 33%
                st.success("✅ PDF processed into chunks.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return

        # Step 2: Create embeddings and vector database
        with st.spinner("🔧 Creating embeddings and building vector database..."):
            try:
                time.sleep(1)  # Simulate processing time
                chroma_client = get_chroma_client()
                chroma_collection = create_and_load_chroma_collection(
                    chroma_client, chunks, file_upload.name
                )
                progress_bar.progress(100)  # Update progress to 66%
                st.success("✅ Vector database created successfully.")
            except Exception as e:
                st.error(f"Error building vector database: {e}")
                return

        # Step 3: Allow user to ask questions
        st.markdown("### 🔍 Ask a Question")
        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("🔎 Retrieving relevant documents..."):
                try:
                    time.sleep(1)  # Simulate processing time
                    retrieval_results = query_chroma(chroma_collection, query)
                    retrieved_docs = retrieval_results["documents"][0]
                    retrieved_embeddings = retrieval_results["embeddings"][0]

                    progress_bar.progress(100)  # Update progress to 100%
                except Exception as e:
                    st.error(f"Error retrieving documents: {e}")
                    return

            with st.spinner("💡 Generating response..."):
                try:
                    if model_type == "OpenAI":
                        response = chat_with_openai(query, retrieved_docs, model=selected_model)
                    elif model_type == "Ollama" and selected_model:
                        response = chat_with_ollama(selected_model, query, retrieved_documents=retrieved_docs)
                    else:
                        response = "⚠️ No valid model selected."
                except Exception as e:
                    response = f"Error generating response: {e}"

            # Display the response
            st.markdown("### 📄 Answer")
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {response}")

            # Display Retrieved Documents
            with st.expander("🔎 Retrieved Documents"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document {idx + 1}:**")
                    st.write(doc)
            # # Embedding Visualization
            # with st.spinner("🔍 Visualizing embeddings..."):
            #     embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
            #     query_embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")([query])[0]
            #     umap_transform = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
            #     visualize_embeddings(query, query_embedding, embeddings, retrieved_embeddings, umap_transform)
            # Embedding Visualization
            with st.spinner("### 🌐 Embedding Visualization"):
                embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
                query_embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")([query])[0]
                # st.markdown("### 🌐 Embedding Visualization")
                query_embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")([query])[0]
                fig = visualize_embeddings_plotly(query, query_embedding, embeddings, retrieved_embeddings, chunks, retrieved_docs)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()
