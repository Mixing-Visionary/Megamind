from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = '0.1.0'
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Neural Photo API"
    DEBUG: bool = True

    # CORS Settings
    ALLOW_ORIGINS: List[str] = ["*"]

    # Model Settings
    BASE_MODEL_PATH: str = "stabilityai/sd-turbo"  # "models/base_model/base_model.safetensors"
    LORA_MODELS_DIR: str = "models/lora/"
    LORA_STYLES: List[str] = ["cyberpunk", "anime", "ghibli", "noir", "gogh"]

    # Performance Optimizations
    ENABLE_MODEL_CPU_OFFLOAD: bool = True
    ENABLE_ATTENTION_SLICING: bool = True
    SAFETY_CHECKER: Optional[bool] = None
    WATERMARKS: Optional[bool] = None
    USE_HALF_PRECISION: bool = True
    DISABLE_VAE_TILING: bool = True
    DISABLE_VAE_SLICING: bool = True

    # Processing Settings
    DEFAULT_STRENGTH: float = 1
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_STEPS: int = 2
    DEFAULT_CONTROLNET_SCALE: float = 0.3
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension for input images

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10

    # Security
    API_KEY_REQUIRED: bool = True
    API_KEYS_PATH: str = "resources/api_keys/api_keys.txt"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
