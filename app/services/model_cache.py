import os
import json
from datetime import datetime
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """Manages metadata and caching for models"""

    def __init__(self):
        self.cache_file = os.path.join("app", "tmp", "model_cache.json")
        self.cache_data = self._load_cache()

    def _load_cache(self):
        """Load existing cache data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model cache: {e}")

        # Initialize empty cache
        return {
            "last_updated": datetime.now().isoformat(),
            "base_model_info": {},
            "lora_models": {}
        }

    def save_cache(self):
        """Save cache data to disk"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache_data, f)

    def update_model_info(self, model_type, model_id, info):
        """Update info for a specific model"""
        if model_type == "base":
            self.cache_data["base_model_info"] = {
                **info,
                "last_used": datetime.now().isoformat()
            }
        elif model_type == "lora":
            self.cache_data["lora_models"][model_id] = {
                **info,
                "last_used": datetime.now().isoformat()
            }

        self.cache_data["last_updated"] = datetime.now().isoformat()
        self.save_cache()

    def get_model_info(self, model_type, model_id=None):
        """Get info for a specific model"""
        if model_type == "base":
            return self.cache_data.get("base_model_info", {})
        elif model_type == "lora" and model_id:
            return self.cache_data.get("lora_models", {}).get(model_id, {})
        return {}