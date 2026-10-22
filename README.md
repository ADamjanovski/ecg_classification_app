# PTB-XL ECG Inference App

## What the application does

This application lets a domain expert upload a PTB-XL ECG record and receive a multi-label superclass classification from one of six trained models. The five superclasses — Normal, Myocardial Infarction, ST/T-wave Changes, Conduction Disturbance, and Hypertrophy — can be active simultaneously, making this a multi-label problem rather than a single-class one. After receiving a prediction, the user can confirm or correct the labels, providing human-in-the-loop feedback that is stored persistently and used as training data for on-demand model retraining. The app also exposes an ensemble mode that runs all available models and averages their logits into a single combined prediction.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v24+)
- [Docker Compose](https://docs.docker.com/compose/) (v2+, included with Docker Desktop)

No Python installation is required on the host - everything runs inside containers.

---

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/LokaHQ/ml-interns-mk-2026.git
cd ml-interns-mk-2026/interns/andrej/week_4

# 2. Copy the environment file and fill in credentials if needed
cp .env.example .env

# 3. Place model weight files in backend/models/
#    Required files:
#      bilstm_attention.pt
#      bilstm_baseline.pt
#      bilstm_regularized.pt
#      bilstm_smaller.pt
#      cnn_baseline.pth
#      cnn_small.pth

# 4. Start all services
docker compose up --build
```

The first build downloads PyTorch (~700 MB) and will take 10-20 minutes depending on your internet speed. Subsequent builds are fast due to Docker layer caching.

Once running:
- Streamlit frontend: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001

To stop: `docker compose down`

---

## How to use

1. **Submit a prediction** — go to the New Prediction page, upload a ZIP archive containing a PTB-XL record (`.dat` + `.hea` pair), select a model or choose Ensemble mode, and click Predict.
2. **Review the result** — the predicted active superclasses, per-class probabilities, and a bar chart are shown immediately.
3. **Confirm or correct** — adjust the checkboxes if the prediction is wrong, optionally add a note, and click Confirm. The corrected labels are saved to the database.
4. **Browse history** — go to Prediction History to see all past predictions with filters by confirmation status and model family. Click any row to expand the full record.
5. **Retrain a model** — go to Retrain, select a model, and click Start Retraining. Retraining uses all confirmed predictions as training data and runs in the background.

---

## Architecture decisions

Models are loaded once at FastAPI startup using a lifespan handler and held in an in-memory `ModelRegistry`. This avoids the latency of reloading weights on every request, which would be prohibitive for BiLSTM and CNN models of this size. Raw ECG input files are stored as ZIP archives in MinIO (keyed by prediction UUID) so the original signal is always retrievable for retraining without duplicating data in MongoDB. MongoDB stores only the prediction metadata and confirmed labels — no tensors or binary blobs. The async MongoDB client (`motor`) is used throughout so inference and database operations do not block each other under concurrent requests. For the CNN models, weights were saved as full model objects during training rather than state dicts; loading uses `weights_only=False` with the `CNN1d` class registered in the backend, which is safe because the weights come from our own training runs.

---

## Known limitations

1. **Retraining excludes the original dataset and does not persist weights** — on-demand retraining uses only confirmed predictions stored in MongoDB, not the original PTB-XL training data. With few confirmed examples the retrained model will likely perform worse than the original. Additionally, retrained weights are saved to disk inside the container and are not uploaded to MinIO, so a container restart silently reverts the model to its original weights with no indication to the user.
2. **Input signal identity is not surfaced in the history view** — the MinIO object key for each uploaded ECG file is stored in MongoDB (`input_ref`) but is not shown in the Prediction History page. Users cannot tell which ECG file produced which prediction, making it difficult to cross-reference results when the same model is run on multiple different inputs.

---

## Frontend-Backend contract

| Field | Value |
|---|---|
| Predict endpoint | `POST /predict` (multipart/form-data) |
| Ensemble endpoint | `POST /predict_all` (multipart/form-data) |
| File field name | `file` |
| File format | ZIP archive containing one `.dat` + one `.hea` file |
| Model field name | `model_name` |
| Model identifier strings | `bilstm_attention`, `bilstm_regularized`, `bilstm_baseline`, `bilstm_smaller`, `cnn_baseline`, `cnn_small` |
| `predicted_class` format | List of active class name strings e.g. `["NORM", "STTC"]` |
| `confirmed_class` format | List of active class name strings (same order, sent on confirm) |
| Class order | NORM, MI, STTC, CD, HYP |
| Backend service name | `backend` (frontend connects to `http://backend:8000`) |
