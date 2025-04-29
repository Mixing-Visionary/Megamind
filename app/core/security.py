from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings
from typing import Optional

# Обращение к апи если есть ключ

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def validate_api_key(api_key: Optional[str] = None):
    """Validate API key if required"""
    if not settings.API_KEY_REQUIRED:
        return True

    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="API key is required",
        )
    with open(settings.API_KEYS_PATH) as file:
        api_keys = [line.rstrip() for line in file]

    if api_keys and api_key not in api_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return True