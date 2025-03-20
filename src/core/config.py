import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Memory settings
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "10"))

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True) 