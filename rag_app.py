from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import threading
import itertools
import sys
import time


load_dotenv()


def spinner(stop_event):
    for char in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r🤖 Thinking {char}")
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write("\r" + " " * 30 + "\r")


# Step 1: Load the existing vector database
print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print("✓ Vector database loaded!")

# Step 2: Set up the AI model (Ollama with API key)
print("\nSetting up AI model...")
llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="https://api.ollama.cloud",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    }
)

# Step 3: Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for property buying in Kerala, India.
Answer the question based on the context provided below.
Be specific and helpful. 
Dont answer any other information outside the context.
If out of scope, politely say "I am sorry, I don't have that information. But I can help you with property buying in Kerala."

Context: {context}

Question: {input}

Answer:
""")

# Step 4: Create the RAG chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("✓ RAG system ready!")

# Step 5: Ask questions!
print("\n" + "="*60)
print("KERALA PROPERTY BUYING ASSISTANT")
print("="*60)




while True:
    question = input("\n❓ Ask a question or type 'exit' to quit: ")
    if question.lower() == 'exit':
        print("Exiting the assistant. Goodbye!")
        break
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()
    try:
        result = retrieval_chain.invoke({"input": question})
        print(f"\n✅ Answer:\n{result['answer']}\n")
        print("\n" + "-"*160)
    finally:
        stop_event.set()
        spinner_thread.join()

