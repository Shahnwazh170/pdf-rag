import os

from dotenv import load_dotenv

load_dotenv()

# API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash-lite"

# PDF
FILE_NAME = "ShahnwazHusain_Resume.pdf"

# Vector Store
DB_PATH = "./vectordb"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# Retrieval
TOP_K = 3

# LLM
TEMPERATURE = 0
MAX_RETRIES = 2
