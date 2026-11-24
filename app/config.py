
import os

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 30)


EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


TOP_K = _int("TOP_K", 5)                  
CANDIDATES = _int("CANDIDATES", 20)         
MIN_DENSE_SCORE = _float("MIN_DENSE_SCORE", 0.20) 
MAX_CHUNK_CHARS = _int("MAX_CHUNK_CHARS", 1200)


DB_PATH = os.getenv("DB_PATH", "data/index.db")
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin")  

HR_CONTACT = os.getenv("HR_CONTACT", "HR at hr@example.com")
