import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import umap
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import tempfile

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None
if 'embedding_function' not in st.session_state:
    st.session_state.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
if 'umap_model' not in st.session_state:
    st.session_state.umap_model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Reuse your existing functions with slight modifications
def read_pdf(file):
    reader = PdfReader(file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts

def chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )
    character_chunks = character_splitter.split_text("\n\n".join(texts))
    
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50, tokens_per_chunk=256)
    token_chunks = []
    for chunk in character_chunks:
        token_chunks.extend(token_splitter.split_text(chunk))
    return token_chunks

def create_chroma_collection(file_name, embedding_function):
    collection_name = os.path.splitext(file_name)[0]
    chroma_client = chromadb.Client()
    
    existing_collections = [col.name for col in chroma_client.list_collections()]
    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)
    
    collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
    return collection

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    texts = read_pdf(tmp_file_path)
    chunks = chunk_texts(texts)
    
    collection = create_chroma_collection(uploaded_file.name, st.session_state.embedding_function)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    
    embeddings = collection.get(include=['embeddings'])['embeddings']
    umap_model = umap.UMAP(random_state=42).fit(embeddings)
    
    os.unlink(tmp_file_path)
    return collection, embeddings, umap_model

def generate_response(query, retrieved_documents):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    information = "\n\n".join(retrieved_documents)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Answer the user's question using only the provided information from the annual report."
        },
        {"role": "user", "content": f"Question: {query}\nInformation: {information}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content

def visualize_embeddings(dataset_embeddings, query_embedding, retrieved_embeddings, umap_model):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    projected_dataset = umap_model.transform(dataset_embeddings)
    projected_query = umap_model.transform([query_embedding])
    projected_retrieved = umap_model.transform(retrieved_embeddings)
    
    ax.scatter(projected_dataset[:, 0], projected_dataset[:, 1], s=10, color='gray', label='Dataset', alpha=0.5)
    ax.scatter(projected_query[:, 0], projected_query[:, 1], s=150, color='red', marker='X', label='Query')
    ax.scatter(projected_retrieved[:, 0], projected_retrieved[:, 1], s=100, edgecolors='green', facecolors='none', label='Retrieved')
    ax.legend()
    ax.set_title('Document Embedding Space Visualization')
    
    return fig

# Streamlit UI
st.set_page_config(layout="wide", page_title="RAG Document Analysis")

# Sidebar
with st.sidebar:
    st.title("Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            st.session_state.chroma_collection, st.session_state.embeddings, st.session_state.umap_model = process_pdf(uploaded_file)
        st.success("PDF processed successfully!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses RAG (Retrieval-Augmented Generation) to analyze PDF documents and answer questions about their content.
    
    1. Upload a PDF document
    2. Ask questions in the chat
    3. View relevant document sections
    4. See visualization of document embeddings
    """)

# Main content area
st.title("Document Q&A System")

# Initialize chat interface
if uploaded_file is None:
    st.info("Please upload a PDF document to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query the vector store
                results = st.session_state.chroma_collection.query(
                    query_texts=[prompt],
                    n_results=5,
                    include=['documents', 'embeddings']
                )
                retrieved_docs = results['documents'][0]
                retrieved_embeddings = results['embeddings'][0]
                
                # Generate response
                response = generate_response(prompt, retrieved_docs)
                
                # Display response
                st.markdown(response)
                
                # Create columns for source documents and visualization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### Relevant Document Sections")
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.expander(f"Source {i}"):
                            st.markdown(doc)
                
                with col2:
                    st.markdown("#### Document Embedding Space")
                    fig = visualize_embeddings(
                        st.session_state.embeddings,
                        st.session_state.embedding_function([prompt])[0],
                        retrieved_embeddings,
                        st.session_state.umap_model
                    )
                    st.pyplot(fig)
        
        # Append assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})