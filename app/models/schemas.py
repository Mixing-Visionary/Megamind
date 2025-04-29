from pydantic import BaseModel, Field, field_validator
import base64
import re
from typing import Optional
from app.models.enums import ProcessingStyle


class ImageProcessingRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    style: ProcessingStyle
    strength: Optional[float] = Field(default=0.5, ge=0, le=1,
                                      description="How much to transform the image (0-1)")
    api_key: Optional[str] = Field(description="API key")

    @field_validator('image')
    def validate_base64(cls, v):
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', v):
            raise ValueError('Invalid base64 encoding')
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 data')
        return v


class ImageProcessingResponse(BaseModel):
    processed_image: str
    processing_time: float
    style: str

    class Config:
        json_schema_extra = {
            "example": {
                "processed_image": "base64_encoded_string",
                "processing_time": 5.32,
                "style": "style1"
            }
        }


class APIError(BaseModel):
    detail: str


class GetStylesResponse(BaseModel):
    styles: list

    class Config:
        json_schema_extra = {
            "styles": ["anime", "cyberpunk", "ghibli", "noir", "gogh"]
        }
