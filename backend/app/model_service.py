import asyncio
import logging
import wfdb
import torch
import torch.nn as nn
import tempfile
import zipfile
import io
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fastapi import HTTPException
from .schemas import ModelState, ModelEntry
from .config import LABEL_MAP, SUPERCLASSES, settings, RETRAIN_DEFAULTS, MODEL_CONFIGS
from .database import list_predictions
from torch.utils.data import Dataset, DataLoader, random_split
from .minio_client import get_bytes
from src.train import train as train_model


logger = logging.getLogger(__name__)


def _object_name_from_input_ref(input_ref: str, bucket: str) -> str:
    prefix = f"{bucket}/"
    if input_ref.startswith(prefix):
        return input_ref[len(prefix):]
    return input_ref

class ModelRegistry:
    def __init__(self):
        self._registry: dict[str, ModelEntry] = {}

    def register(self, model_name: str, model):
        model.eval()
        self._registry[model_name] = ModelEntry(model=model)

    def available(self) -> list[str]:
        """Only return models that are ready — hides ones retraining."""
        return [
            name for name, entry in self._registry.items()
            if entry.state == ModelState.READY
        ]

    def get(self, model_name: str) -> ModelEntry:
        entry = self._registry.get(model_name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        if entry.state == ModelState.TRAINING:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' is currently retraining")
        return entry

    async def retrain(self, model_name: str):
        entry = self._registry.get(model_name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        if entry.state == ModelState.TRAINING:
            raise HTTPException(status_code=409, detail=f"Model '{model_name}' is already retraining")

        async def run():
            try:
                entry.state = ModelState.TRAINING
                with entry.lock:
                    entry.model.train()
                    model_path = MODEL_CONFIGS[model_name]["path"]
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    train_loader, val_loader = await build_training_loaders(
                        val_split=RETRAIN_DEFAULTS["val_split"],
                        batch_size=RETRAIN_DEFAULTS["batch_size"],
                    )

                    if "cnn" in model_name.lower():
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            entry.model.train_model,
                            train_loader,
                            val_loader,
                            RETRAIN_DEFAULTS["num_epochs"],
                            RETRAIN_DEFAULTS["learning_rate"],
                            RETRAIN_DEFAULTS["patience"],
                            Adam,
                            RETRAIN_DEFAULTS["weight_decay"],
                            nn.BCEWithLogitsLoss,
                            None,
                        )
                        torch.save(entry.model.state_dict(), model_path)
                    else:
                        optimizer = Adam(
                            entry.model.parameters(),
                            lr=RETRAIN_DEFAULTS["learning_rate"],
                            weight_decay=RETRAIN_DEFAULTS["weight_decay"],
                        )
                        criterion = nn.BCEWithLogitsLoss()
                        scheduler = ReduceLROnPlateau(
                            optimizer,
                            mode="max",
                            factor=0.5,
                            patience=3,
                        )

                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            train_model,
                            entry.model,
                            train_loader,
                            val_loader,
                            criterion,
                            optimizer,
                            scheduler,
                            torch.device("cpu"),
                            RETRAIN_DEFAULTS["num_epochs"],
                            RETRAIN_DEFAULTS["patience"],
                            model_path,
                        )
                    entry.model.eval()
                entry.state = ModelState.READY
            except Exception as e:
                entry.state = ModelState.READY
                logger.exception("Retraining failed for model '%s'", model_name)

        asyncio.create_task(run())



def logits_to_prediction(
    logits: np.ndarray,
) -> tuple[list[str], list[float], dict[str, float]]:
    """
    Convert raw logits → predicted classes, confidences, probabilities.
    Uses sigmoid for multilabel (each class is independent).
    """
    probs      = 1 / (1 + np.exp(-logits)) 
    probs      = probs[0]                    
    threshold  = 0.5

    predicted   = [LABEL_MAP[i] for i, p in enumerate(probs) if p >= threshold]
    confidences = [float(probs[i]) for i, p in enumerate(probs) if p >= threshold]

    if not predicted:
        top         = int(np.argmax(probs))
        predicted   = [LABEL_MAP[top]]
        confidences = [float(probs[top])]

    prob_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

    return predicted, confidences, prob_dict

def read_signal(data_path: str | Path, record_path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(str(Path(data_path) / record_path))
    return signal.astype(np.float32)

def run_inference(signal: np.ndarray, entry: ModelEntry) -> tuple[list[str], list[float], dict[str, float]]:
    tensor = torch.from_numpy(signal).unsqueeze(0)
    
    with entry.lock:
        with torch.no_grad():
            logits = entry.model(tensor).numpy()  

    return logits_to_prediction(logits)


def run_inference_logits(signal: np.ndarray, entry: ModelEntry) -> np.ndarray:
    tensor = torch.from_numpy(signal).unsqueeze(0)

    with entry.lock:
        with torch.no_grad():
            logits = entry.model(tensor).numpy()

    return logits





# ── Dataset ───────────────────────────────────────────────────────────────────
class ConfirmedPredictionDataset(Dataset):
    def __init__(
        self,
        records:  list[dict],
        mean:     np.ndarray | None = None,
        std:      np.ndarray | None = None,
    ):
        self.records = records
        self.mean    = mean
        self.std     = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        object_name = _object_name_from_input_ref(record["input_ref"], settings.data_bucket)

        zip_bytes = get_bytes(
            bucket      = settings.data_bucket,
            object_name = object_name,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(tmp_dir)
            hea_files = list(Path(tmp_dir).rglob("*.hea"))
            signal    = read_signal(tmp_dir, hea_files[0].stem)  

        if self.mean is not None and self.std is not None:
            signal = (signal - self.mean) / self.std

        label = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for cls in record["confirmed_class"]:
            if cls in SUPERCLASSES:
                label[SUPERCLASSES.index(cls)] = 1.0

        signal_tensor = torch.from_numpy(signal.astype(np.float32))
        label_tensor  = torch.from_numpy(label)

        return signal_tensor, label_tensor


async def build_training_loaders(
    val_split:  float = 0.2,
    batch_size: int   = 32,
) -> tuple[DataLoader, DataLoader]:
    records, _ = await list_predictions(confirmed_only=True, limit=10_000)

    if not records:
        raise HTTPException(status_code=422, detail="No confirmed predictions available for retraining")

    val_size   = max(1, int(len(records) * val_split))
    train_size = len(records) - val_size

    train_records, val_records = random_split(records, [train_size, val_size])
    
    mean, std = compute_normalization_stats(list(train_records))

    train_dataset = ConfirmedPredictionDataset(list(train_records), mean=mean, std=std)
    val_dataset   = ConfirmedPredictionDataset(list(val_records),   mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def compute_normalization_stats(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-lead mean/std over all training records and timesteps."""
    sums = np.zeros(12, dtype=np.float64)
    sum_squares = np.zeros(12, dtype=np.float64)
    n_values = 0

    for record in records:
        object_name = _object_name_from_input_ref(record["input_ref"], settings.data_bucket)

        zip_bytes = get_bytes(
            bucket      = settings.data_bucket,
            object_name = object_name,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(tmp_dir)
            hea_files = list(Path(tmp_dir).rglob("*.hea"))
            signal    = read_signal(tmp_dir, hea_files[0].stem)

        sums += signal.sum(axis=0)
        sum_squares += np.square(signal, dtype=np.float64).sum(axis=0)
        n_values += signal.shape[0]

    mean = sums / n_values
    variance = np.maximum(sum_squares / n_values - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)

registry = ModelRegistry()
