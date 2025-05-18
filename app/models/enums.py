from enum import Enum
from pathlib import Path
from app.core.config import settings
import json


# Стили которые есть
class ProcessingStyle(str, Enum):
    ANIME = "anime"
    CYBERPUNK = "cyberpunk"
    GHIBLI = "ghibli"
    NOIR = "noir"
    GOGH = "gogh"
    COMIX = "comix"
    FLAT_COLOR = "flat_color"
    POINTILISM = "pointilism"

    def get_path(self):
        base_dir = Path.cwd()
        return str(base_dir / settings.LORA_MODELS_DIR / f'{self.value}.safetensors')

    def get_prompt(self):
        with open('C:/Users/Anton/Desktop/Megamind/resources/prompts/styles.json') as f:
            d = json.load(f)
            return d[self.value]

    def get_negative_prompt(self):
        with open('C:/Users/Anton/Desktop/Megamind/resources/prompts/styles.json') as f:
            d = json.load(f)
            return d['negative']


