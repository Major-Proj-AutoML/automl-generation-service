# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://automl:automl_dev_pw@localhost:5432/automl"
    redis_url: str = "redis://localhost:6379/0"
    data_service_url: str = "http://localhost:8001"
    # LLM provider: "openrouter" (default) or "ollama". The worker reads these
    # from the environment; see src/experiments/runner.py in automl-reusables.
    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_url: str = "https://openrouter.ai/api/v1"
    ollama_url: str = "http://localhost:11434"
    service_port: int = 8003
    log_level: str = "INFO"
    default_max_iter: int = 3
    default_timeout_seconds: int = 300
    http_timeout_seconds: float = 30.0

    minio_endpoint: str = ""
    minio_access_key: str = "automl"
    minio_secret_key: str = "automl_dev_pw"
    minio_secure: bool = False
    minio_scripts_bucket: str = "scripts"
    minio_models_bucket: str = "models"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
