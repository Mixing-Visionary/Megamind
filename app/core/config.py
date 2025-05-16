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
    BASE_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"  # "models/base_model/base_model.safetensors"
    CONTROLNET_DEPTH_MODEL: str = "diffusers/controlnet-depth-sdxl-1.0-small"
    CONTROLNET_CANNY_MODEL: str = "diffusers/controlnet-canny-sdxl-1.0-small"
    VAE: str = "madebyollin/sdxl-vae-fp16-fix"
    PROMPT_MODEL: str = "Salesforce/blip-image-captioning-large"
    DEPTH_DETECTOR: str = "lllyasviel/ControlNet"

    LORA_MODELS_DIR: str = "models/lora/"
    PRELOAD_BASE_MODEL: bool = True
    MASK_STRATEGY: str = "depth" # canny, depth

    # Performance Optimizations
    ENABLE_MODEL_CPU_OFFLOAD: bool = True
    SAFETY_CHECKER: Optional[bool] = None
    USE_HALF_PRECISION: bool = True

    # Processing Settings
    DEFAULT_STRENGTH: float = 1
    DEFAULT_GUIDANCE_SCALE: float = 10
    DEFAULT_STEPS: int = 15
    DEFAULT_CONTROLNET_SCALE: float = 0.5
    MAX_IMAGE_SIZE: int = 2048  # Maximum dimension for input images

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10

    # Security
    API_KEY_REQUIRED: bool = True
    API_KEYS_PATH: str = "C:/Users/Anton/Desktop/Megamind/resources/api_keys/api_keys.txt"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
