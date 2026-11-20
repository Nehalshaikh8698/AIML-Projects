from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "medscan"
    model_path: str = "app/models/densenet121_nih14.pth"
    model_threshold: float = 0.50
    segmentation_model_path: str = "app/models/unet3d_msd_lung.pth"
    max_volume_mb: int = 200
    orthanc_url: str = "http://localhost:8042"
    orthanc_username: str = "demo"
    orthanc_password: str = "demo-change-me"
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 20
    demo_mode: bool = True
    storage_dir: str = "storage"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    @property
    def resolved_segmentation_model_path(self) -> Path:
        path = Path(self.segmentation_model_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    @property
    def resolved_storage_dir(self) -> Path:
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


@lru_cache

def get_settings() -> Settings:
    return Settings()
