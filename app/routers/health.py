from fastapi import APIRouter, Depends, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import platform
import psutil
import time
import torch
from app.core.config import settings
from app.core.security import validate_api_key
from app.services.model_loader import ModelLoader
import os

router = APIRouter(tags=["Health"])


# Models for health response
class ModelStatus(BaseModel):
    model_type: str
    is_loaded: bool
    name: str
    device: str
    used_memory_mb: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    cpu_usage: float
    ram_usage: float
    total_ram_gb: float
    disk_usage: float
    python_version: str
    os_info: str


class DetailedHealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    system: SystemStatus
    models: List[ModelStatus]
    gpu_available: bool
    gpu_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str

# Store start time for uptime calculation
START_TIME = time.time()


# Basic health check
@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="ok",
        version=settings.API_VERSION
    )


# Detailed health status
@router.get("/status", response_model=DetailedHealthResponse)
async def detailed_health(
        include_models: bool = Query(True, description="Include model status information"),
        api_key: Optional[str] = Header(None, description="Api key for detailed diagnostics")
):
    """
    Detailed health status including system information and model status.

    This endpoint provides detailed diagnostics about:
    - System resources (CPU, RAM, disk)
    - Model loading status
    - GPU availability and usage
    """
    # For enhanced security, you could require an api key for sensitive details
    validate_api_key(api_key)

    # System status
    system_status = SystemStatus(
        cpu_usage=psutil.cpu_percent(),
        ram_usage=psutil.virtual_memory().percent,
        total_ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
        disk_usage=psutil.disk_usage('/').percent,
        python_version=platform.python_version(),
        os_info=f"{platform.system()} {platform.release()}"
    )

    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_info = None

    if gpu_available:
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
        }

        # Add memory info if available
        if hasattr(torch.cuda, 'memory_allocated') and hasattr(torch.cuda, 'max_memory_allocated'):
            gpu_info.update({
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
                "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            })

    # Model status
    models = []
    if include_models:
        # Check base model
        model_loader = ModelLoader()
        base_model_loaded = hasattr(model_loader, 'base_model') and model_loader.base_model is not None

        models.append(ModelStatus(
            model_type="base",
            is_loaded=base_model_loaded,
            name=os.path.basename(settings.BASE_MODEL),
            device=model_loader.device if hasattr(model_loader, 'device') else "unknown",
            used_memory_mb=round(torch.cuda.memory_allocated() / (1024 ** 2), 2) if gpu_available else None
        ))

        # Check LoRA models
        if hasattr(model_loader, 'current_lora_style') and model_loader.current_lora_style:
            models.append(ModelStatus(
                model_type="lora",
                is_loaded=True,
                name=model_loader.current_lora_style,
                device=model_loader.device if hasattr(model_loader, 'device') else "unknown"
            ))

    return DetailedHealthResponse(
        status="ok",
        version=settings.API_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 2),
        system=system_status,
        models=models,
        gpu_available=gpu_available,
        gpu_info=gpu_info
    )
