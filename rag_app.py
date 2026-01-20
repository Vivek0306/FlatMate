# rag.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

from config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    TOP_K,
    TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    PDF_DIR
)

def load_documents():
    all_documents = []
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    for txt_file in txt_files:
        print(f"Loading text file: {txt_file}...")
        loader = TextLoader(os.path.join(DATA_DIR, txt_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata['source'] = txt_file
        all_documents.extend(docs)

    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            print(f"Loading PDF file: {pdf_file}...")
            try:
                loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))
                docs = loader.load()

                for doc in docs:
                    doc.metadata['source'] = pdf_file
                all_documents.extend(docs)
                print(f"   ✓ Loaded {len(docs)} pages from {pdf_file}")
            except Exception as e:
                print(f"   ⚠️ Failed to load {pdf_file}: {e}")
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def create_vectorstore():
    print("Loading documents.....")
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found! Please add text files or PDFs to the data/ directory.")
    
    print(f"\nLoaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nSplit into {len(chunks)} chunks of text.")
    return chunks


def load_or_create_vectorstore(embeddings):
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        if vectorstore._collection.count() == 0:
            raise ValueError("Empty vector store")

        print("✓ Loaded existing vector database")
        return vectorstore

    except Exception:
        print("⚠️ No valid vector DB found. Creating a new one...")

        vectorstore = Chroma.from_documents(
            documents=create_vectorstore(),
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        print("✓ Vector database created")
        return vectorstore



def create_rag_chain():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = load_or_create_vectorstore(embeddings)

    # LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
        }
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are FlatMate, an expert assistant for property buying in Kerala, India.

INSTRUCTIONS:
1. Answer ONLY using the information from the context below
2. Be thorough and include ALL relevant details from the context
3. After each fact or piece of information, cite the source document in [brackets]
4. If multiple documents mention the same thing, cite all of them
5. If the context doesn't contain enough information, acknowledge what's missing

Context:
{context}

Question:
{input}

Answer with citations:
""")


    # RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return create_retrieval_chain(retriever, document_chain)
