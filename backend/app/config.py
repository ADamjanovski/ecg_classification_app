from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False

    data_bucket: str = "data"

    # Minimum confirmed examples before retraining is allowed
    min_train_samples: int = 3

    mongo_uri: str
    db_name: str

    model_config = {"env_file": ".env", "protected_namespaces": ()}


settings = Settings()


LABEL_MAP = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

MODELS_DIR = Path("/app/models")

MODEL_CONFIGS = {
    "bilstm_attention": {
        "path": MODELS_DIR / "bilstm_attention.pt",
        "use_attention": True,
    },
    "bilstm_baseline": {
        "path": MODELS_DIR / "bilstm_baseline.pt",
        "dropout": 0.3,
    },
    "bilstm_regularized": {
        "path": MODELS_DIR / "bilstm_regularized.pt",
        "dropout": 0.5,
    },
    "bilstm_smaller": {
        "path": MODELS_DIR / "bilstm_smaller.pt",
        "hidden_size": 64,
    },
    "cnn_baseline": {
        "path": MODELS_DIR / "cnn_baseline.pth",
        "in_channels":   12,
        "num_classes":   5,
        "out_channels":  [64, 128, 256, 256],
        "kernel_sizes":  [7, 5, 5, 3],
        "pool_sizes":    [2, 2, 2, 2],
        "dense_units":   [512, 256],
        "dropout_rate":  0.4,
        "use_batch_norm": True,
    },
    "cnn_small": {
        "path": MODELS_DIR / "cnn_small.pth",
        "in_channels":   12,
        "num_classes":   5,
        "out_channels":  [32, 64, 128],
        "kernel_sizes":  [7, 5, 3],
        "pool_sizes":    [2, 2, 2],
        "dense_units":   [128],
        "dropout_rate":  0.4,
        "use_batch_norm": True,
    },
}
RETRAIN_DEFAULTS = {
    "num_epochs":    50,
    "learning_rate": 0.001,
    "patience":      5,
    "weight_decay":  1e-4,
    "val_split":     0.2,
    "batch_size":    32,
}

MONGO_URI = settings.mongo_uri
DB_NAME   = settings.db_name