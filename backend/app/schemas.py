import threading
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime, timezone
_no_ns = ConfigDict(protected_namespaces=())


class PredictionBase(BaseModel):
    model_config = _no_ns

    conversation_id :str
    prediction_id: str
    model_name: Optional[str] = None
    predicted_class: list[str]
    confidence: list[float]  
    probabilities: dict[str, float]
    input_ref: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Prediction(PredictionBase):
    confirmed_class: Optional[list[str]] = None
    confirmed: bool = False
    confirmed_at:  Optional[datetime] = None 
    notes: Optional[str] = None


class Model(BaseModel):
    name : str

class ConfirmedPredictionRequest(BaseModel):
    model_config = _no_ns
    prediction_id : str
    confirmed_class : list[str]
    notes : str

class HealthStatus(BaseModel):
    model_config = _no_ns
    status : str

class ErrorResponse(BaseModel):
    status_code: int
    message: str

class ModelState(Enum):
    READY    = "ready"
    TRAINING = "training"

@dataclass
class ModelEntry:
    model:      any
    state:      ModelState        = ModelState.READY
    lock:       threading.Lock    = field(default_factory=threading.Lock)

class PredictionListRequest(BaseModel):
    limit:          int  = Field(default=20, ge=1, le=100)
    offset:         int  = Field(default=0, ge=0)
    confirmed_only: bool = Field(default=False)

class PredictionListResponse(BaseModel):
    total:   int
    limit:   int
    offset:  int
    results: list[Prediction]

class ConfirmPredictionRequest(BaseModel):
    confirmed_class: list[str]        # checklist of true classes
    notes:           Optional[str] = None

class RetrainResponse(BaseModel):
    status:     str
    model_name: str


class IndividualPrediction(BaseModel):
    predicted_class: list[str]
    confidence: list[float]
    probabilities: dict[str, float]


class PredictAllResponse(PredictionBase):
    individual_predictions: dict[str, IndividualPrediction]

    
