from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nemetron_base_url: str = "http://10.33.11.12:8103"
    nemetron_api_key: str = "not-needed"
    nemetron_model: str = "nemetron-30b"

    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8000

    # Output token handling (progressive expansion)
    initial_max_tokens: int = 8192
    max_output_tokens: int = 262000
    enable_token_expansion: bool = True

    allowed_tools: str = (
        "read_file,read_multiple_files,write_file,edit_file,"
        "list_directory,search_codebase,run_commands,create_folder,fetch_web_content"
    )

    log_level: str = "INFO"

    default_system_prompt: str = (
        "You are a helpful coding assistant. "
        "Answer questions directly and completely when you can. "
        "Use tools only when you need to read files, search code, run commands, or fetch web content. "
        "If the user asks you to explain code, analyze it, or answer a question, respond directly without asking how to assist. "
        "Be proactive: if the user mentions a file path, read it. If they mention a codebase, search it. "
        "Do not ask clarifying questions when the task is clear. Provide thorough, complete answers. "
        "IMPORTANT: After you have used tools and received their results, STOP calling tools and provide your final answer directly. "
        "Do not call the same tool more than once with the same arguments. "
        "When you have enough information to answer, write your final response without any tool calls. "
        "WORK INCREMENTALLY: When creating files or doing multi-step tasks, do ONE step at a time. "
        "Create one file, acknowledge it was created, then move to the next step. "
        "Do not try to do everything at once. After each action, briefly state what you did and what's next."
    )

    @property
    def allowed_tool_set(self) -> set[str]:
        return {name.strip() for name in self.allowed_tools.split(",") if name.strip()}


settings = Settings()
