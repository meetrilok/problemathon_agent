from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    db_path: Path
    faiss_path: Path
    secret_key: str
    manager_username: str
    manager_password: str
    match_threshold: float
    azure_endpoint: str | None
    azure_api_key: str | None
    azure_api_version: str
    azure_embedding_deployment: str | None
    azure_chat_deployment: str | None

    @classmethod
    def from_env(cls) -> "AppConfig":
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            db_path=data_dir / "ideas.db",
            faiss_path=data_dir / "faiss_index.pkl",
            secret_key=os.getenv("FLASK_SECRET_KEY", "dev-secret"),
            manager_username=os.getenv("MANAGER_USERNAME", "manager"),
            manager_password=os.getenv("MANAGER_PASSWORD", "manager123"),
            match_threshold=float(os.getenv("MATCH_THRESHOLD", "0.30")),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        )
