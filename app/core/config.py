from pydantic_settings import BaseSettings
from pathlib import Path
class Settings(BaseSettings):
    # 디렉토리 설정
    HF_TOKEN: str          # 반드시 대문자 HF_TOKEN으로 선언
    ORTHANC_URL: str       # 반드시 대문자 ORTHANC_URL으로 선언
    MODEL_DIR: str = "app/models"
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMPLATE_DIR: Path = BASE_DIR / "templates"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    # DeepFill 설정
    DEEPFILL_MODEL_NAME: str = "resnet18-5c106cde.pth"
    DEEPFILL_BATCH_SIZE: int = 1
    DEEPFILL_NUM_WORKERS: int = 0
    # 전처리 설정
    TARGET_DIMS: tuple = (400, 400)
    FOREGROUND_THRESHOLD: float = 0.2
    SHADE_THRESHOLD: int = 6
    ALLOWED_NAMES: list = ['app', 'rlq']
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {'.png', '.jpg', '.jpeg', '.bmp'}
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"  # 정의되지 않은 필드는 거부
settings = Settings()