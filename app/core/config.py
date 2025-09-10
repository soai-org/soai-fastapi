from pydantic_settings import BaseSettings
from pathlib import Path
class Settings(BaseSettings):
    MODEL_DIR: str = "app/models"
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    model_config = {
        "extra": "ignore"
    }
settings = Settings()