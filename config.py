# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to hold all settings for the Pharma Launch AI system.
    Loads sensitive data from environment variables.
    """
    
    # --- OpenAI Configuration ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4-turbo"  # Or another model like "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # --- LangSmith Configuration ---
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "pharma-launch-assistant")

    # --- Data Ingestion & RAG Configuration ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    VECTOR_STORE_PATH = "./data/vector_store"

    # --- External API Endpoints ---
    FDA_BASE_URL = "https://www.fda.gov"
    CLINICALTRIALS_API = "https://clinicaltrials.gov/api"