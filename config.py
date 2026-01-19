# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "https://api.ollama.cloud"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

TOP_K = 3
TEMPERATURE = 0.3

APP_NAME = "FlatMate"
APP_TAGLINE = "Kerala Property Buying Assistant"

