from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/documents"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_retries: int = 1
    max_upload_mb: int = 15
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
