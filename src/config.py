import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    
    # Internal paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")

# Ensure required directories exist
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
