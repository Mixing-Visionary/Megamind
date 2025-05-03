from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
import time
import asyncio

from pydantic import Field

from app.models.schemas import ImageProcessingRequest, ImageProcessingResponse, APIError, GetStylesResponse
from app.services.image_processor import ImageProcessor
from app.models.enums import ProcessingStyle
from app.core.config import settings
from app.core.security import validate_api_key
import base64
from typing import Optional, Dict, List, Any

router = APIRouter()
processor = ImageProcessor()


@router.post("/process",
             response_model=ImageProcessingResponse,
             responses={400: {"model": APIError}, 429: {"model": APIError}, 500: {"model": APIError}})
async def process_image(
        request: ImageProcessingRequest,
):
    print(request.api_key)
    """Process an image using a selected style with the neural network"""
    validate_api_key(request.api_key)

    return await processor.process_image(request)


@router.get("/styles", response_model=GetStylesResponse)
async def get_available_styles():
    """Get a list of all available processing styles"""
    return GetStylesResponse(styles=[style.value for style in ProcessingStyle])

