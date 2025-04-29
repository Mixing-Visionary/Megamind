from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.routers import api_v1, health
from app.core.config import settings
import logging
import time
from contextlib import asynccontextmanager

# Configure logging
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="[%(name)s] - %(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


# Lifespan for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model cache, prepare environment
    logger.info("Starting Neural Photo API")
    try:
        # Pre-load base model if configured
        if settings.PRELOAD_BASE_MODEL:
            from app.services.model_loader import ModelLoader
            logger.info("Pre-loading base model")
            ModelLoader().load_base_model()
            logger.info("Base model loaded successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down Neural Photo API")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for neural network image processing with predefined styles",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Request error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


# Include routers
app.include_router(api_v1.router, prefix=settings.API_V1_STR)
app.include_router(health.router, prefix="/health")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)