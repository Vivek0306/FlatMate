# rag.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    TOP_K,
    TEMPERATURE,
)

def create_rag_chain():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Vector store
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

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
