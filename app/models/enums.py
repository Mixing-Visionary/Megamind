from enum import Enum
from pathlib import Path
from app.core.config import settings


# Стили которые есть
class ProcessingStyle(str, Enum):
    ANIME = "anime"
    CYBERPUNK = "cyberpunk"
    GHIBLI = "ghibli"
    NOIR = "noir"
    GOGH = "gogh"

    def get_path(self):
        base_dir = Path.cwd()
        return str(base_dir / settings.LORA_MODELS_DIR / f'{self.value}.safetensors')

