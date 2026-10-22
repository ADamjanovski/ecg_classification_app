import tempfile
import zipfile
import io
import uuid
import numpy as np
from pathlib import Path
from fastapi import APIRouter,Depends,  UploadFile, File, Form, HTTPException
from .schemas import PredictionListResponse, Prediction, PredictionBase, Model, PredictionListRequest, ConfirmPredictionRequest, RetrainResponse, PredictAllResponse, IndividualPrediction
from .config import settings
from .model_service import read_signal, run_inference, run_inference_logits, logits_to_prediction, registry
from .database import *
from .minio_client import upload_bytes
router = APIRouter()

@router.get("/predictions", response_model=PredictionListResponse)
async def get_predictions(params: PredictionListRequest= Depends()):
    records, total = await list_predictions(
        limit          = params.limit,
        offset         = params.offset,
        confirmed_only = params.confirmed_only,
    )
    return PredictionListResponse(
        total   = total,
        limit   = params.limit,
        offset  = params.offset,
        results = records,
    )

@router.get("/predictions/{prediction_id}", response_model=Prediction)
async def get_predictiction_endpoint(prediction_id: str):
    prediction = await get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return Prediction(**prediction)

@router.get("/models", response_model=list[Model])
async def get_models():
    return [Model(name=name) for name in registry.available()]

@router.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    records = await get_conversation(conversation_id)
    if not records:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "predictions": records}

@router.post("/predict", response_model=PredictionBase)
async def predict(
    file:            UploadFile = File(...),
    model_name:      str        = Form(...),
    conversation_id: str | None = Form(default=None),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=422, detail="Upload must be a .zip file")

    zip_bytes      = await file.read()
    prediction_id  = str(uuid.uuid4()) 

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)

        records = list(Path(tmp_dir).rglob("*.hea"))
        if not records:
            raise HTTPException(status_code=422, detail="No .hea record found in zip")
        if len(records) > 1:
            raise HTTPException(status_code=422, detail="Expected exactly one record, got multiple")

        signal = read_signal(tmp_dir, records[0].stem)  

    input_ref = f"{settings.data_bucket}/{prediction_id}/record.zip"
    try:
        upload_bytes(
            bucket       = settings.data_bucket,
            object_name  = f"{prediction_id}/record.zip",
            data         = zip_bytes,
            content_type = "application/zip",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage error: {str(e)}")

    entry  = registry.get(model_name)
    predicted_class, confidences, prob_dict = run_inference(signal, entry)

    prediction = PredictionBase(
        conversation_id = conversation_id or str(uuid.uuid4()),
        prediction_id   = prediction_id,
        model_name      = model_name,
        predicted_class = predicted_class,
        confidence      = confidences,
        probabilities   = prob_dict,
        input_ref       = input_ref,
        created_at      = datetime.now(timezone.utc),
    )

    await insert_prediction(Prediction(**prediction.model_dump()))

    return prediction

@router.post("/predictions/{prediction_id}/confirm", response_model=Prediction)
async def confirm_prediction_endpoint(
    prediction_id: str,
    body:          ConfirmPredictionRequest,
):
    updated = await confirm_prediction(
        prediction_id   = prediction_id,
        confirmed_class = body.confirmed_class,
        notes           = body.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction = await get_prediction(prediction_id)
    return Prediction(**prediction)


@router.post("/predict_all", response_model=PredictAllResponse)
async def predict_all(
    file:            UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=422, detail="Upload must be a .zip file")

    model_names = registry.available()
    if not model_names:
        raise HTTPException(status_code=503, detail="No ready models available")

    zip_bytes      = await file.read()
    prediction_id  = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)

        records = list(Path(tmp_dir).rglob("*.hea"))
        if not records:
            raise HTTPException(status_code=422, detail="No .hea record found in zip")
        if len(records) > 1:
            raise HTTPException(status_code=422, detail="Expected exactly one record, got multiple")

        signal = read_signal(tmp_dir, records[0].stem)

    input_ref = f"{settings.data_bucket}/{prediction_id}/record.zip"
    try:
        upload_bytes(
            bucket       = settings.data_bucket,
            object_name  = f"{prediction_id}/record.zip",
            data         = zip_bytes,
            content_type = "application/zip",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage error: {str(e)}")

    all_logits = []
    individual_predictions: dict[str, IndividualPrediction] = {}
    for model_name in model_names:
        entry = registry.get(model_name)
        logits = run_inference_logits(signal, entry)
        all_logits.append(logits)

        predicted_class, confidences, prob_dict = logits_to_prediction(logits)
        individual_predictions[model_name] = IndividualPrediction(
            predicted_class=predicted_class,
            confidence=confidences,
            probabilities=prob_dict,
        )

    avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    predicted_class, confidences, prob_dict = logits_to_prediction(avg_logits)

    prediction = PredictAllResponse(
        conversation_id = conversation_id or str(uuid.uuid4()),
        prediction_id   = prediction_id,
        model_name      = "ensemble",
        predicted_class = predicted_class,
        confidence      = confidences,
        probabilities   = prob_dict,
        input_ref       = input_ref,
        created_at      = datetime.now(timezone.utc),
        individual_predictions = individual_predictions,
    )

    await insert_prediction(Prediction(**prediction.model_dump(exclude={"individual_predictions"})))

    return prediction

@router.post("/retrain/{model_name}", response_model=RetrainResponse)
async def retrain_endpoint(model_name: str):
    await registry.retrain(model_name=model_name)

    return RetrainResponse(status="retraining started", model_name=model_name)

@router.get("/health")
async def health():
    return {"status": "ok"}
