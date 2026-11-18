from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LUMEN_", env_file=".env", extra="ignore")

    provider: Literal["echo", "openai", "anthropic"] = "echo"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_per_minute: int = Field(default=120, ge=1)
    cache_enabled: bool = True

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LUMEN_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LUMEN_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )

    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=384, ge=8, le=8192)

    data_dir: Path = Path("data")
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, ge=1, le=50)
    max_context_chars: int = Field(default=6000, ge=500)

    agent_max_steps: int = Field(default=6, ge=1, le=20)
    agent_token_budget: int = Field(default=20000, ge=100)
    max_input_chars: int = Field(default=20000, ge=100)

    api_key: str | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def vector_db_path(self) -> Path:
        return self.data_dir / "vectors.sqlite3"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache.sqlite3"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
