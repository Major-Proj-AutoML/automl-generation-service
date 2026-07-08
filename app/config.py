from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://automl:automl_dev_pw@localhost:5432/automl"
    redis_url: str = "redis://localhost:6379/0"
    data_service_url: str = "http://localhost:8001"
    ollama_url: str = "http://localhost:11434"
    service_port: int = 8003
    log_level: str = "INFO"
    default_max_iter: int = 3
    default_timeout_seconds: int = 300
    http_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
