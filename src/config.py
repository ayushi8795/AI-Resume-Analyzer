import os 
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma_db")
    

settings = settings()

if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")