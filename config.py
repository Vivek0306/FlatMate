# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = "llama3.2"
# OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "https://api.ollama.cloud"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

TOP_K = 8
TEMPERATURE = 0.2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

APP_NAME = "FlatMate"
APP_TAGLINE = "Kerala Property Buying Assistant"


DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "pdf")
