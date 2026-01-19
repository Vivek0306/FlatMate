# rag.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
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
)

def create_vectorstore():
    print("Loading documents.....")
    loader = TextLoader("data/kerala_property_guide.txt")
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
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
You are FlatMate (Application Name), a helpful assistant for property buying in Kerala, India.
Answer ONLY using the context below.
If information is missing, say:
"I'm sorry, I don't have that information. I can help with property buying in Kerala."

Context:
{context}

Question:
{input}

Answer:
""")

    # RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return create_retrieval_chain(retriever, document_chain)
