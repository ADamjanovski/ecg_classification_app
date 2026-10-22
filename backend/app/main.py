import torch
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .config import MODEL_CONFIGS
from .model_service import registry
from .model import router
from .database import connect_db, close_db
from src.BLSTM import BiLSTMClassifier
from CNN1d import CNN1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()

    for name, conf in MODEL_CONFIGS.items():
        path = conf["path"]
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            continue

        args = {k: v for k, v in conf.items() if k != "path"}
        if "cnn" in name:
            model = CNN1d(**args)

            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)

        else:
            model = BiLSTMClassifier(**args)

            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

        registry.register(name, model)
        logger.info(f"Loaded model: {name}")

    yield

    await close_db()
    registry._registry.clear()
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)