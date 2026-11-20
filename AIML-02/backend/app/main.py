import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import create_store
from app.services.model_service import ModelService
from app.services.segmentation_service import SegmentationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.resolved_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    app.state.store = create_store(settings.mongo_uri, settings.mongo_database)
    app.state.model = ModelService(settings.resolved_model_path, settings.model_threshold)
    app.state.segmentation_model = SegmentationService(settings.resolved_segmentation_model_path)
    yield


app = FastAPI(
    title="MedScan AI API",
    description="Educational chest X-ray classification and review API. Not for clinical use.",
    version="1.0.0",
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/media", StaticFiles(directory=str(settings.resolved_storage_dir), check_dir=False), name="media")


@app.get("/")
def root():
    return {"name": "MedScan AI", "docs": "/docs", "notice": "Educational use only — not for clinical use."}
