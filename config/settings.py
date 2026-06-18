"""Environment and LLM configuration for the graph-reasoning agent."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o", description="Primary chat model")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256)

    # Agent loop
    max_iterations: int = Field(default=5, ge=1, le=20)
    sandbox_timeout_seconds: float = Field(default=30.0, ge=1.0)

    # ChromaDB
    chroma_persist_directory: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="macro_financial_corpus")

    # Graph
    graph_data_path: str = Field(default="./data/structured_graph.json")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings singleton."""
    return Settings()
