from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nemetron_base_url: str = "http://localhost:11434"
    nemetron_api_key: str = "not-needed"
    nemetron_model: str = "nemetron-30b"

    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8000

    default_max_tokens: int = 8192
    max_output_tokens: int = 32768

    allowed_tools: str = (
        "read_file,read_multiple_files,write_file,edit_file,"
        "list_directory,search_codebase,run_commands,create_folder,fetch_web_content"
    )

    log_level: str = "INFO"

    @property
    def allowed_tool_set(self) -> set[str]:
        return {name.strip() for name in self.allowed_tools.split(",") if name.strip()}


settings = Settings()
