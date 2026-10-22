from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from typing import Optional
from .schemas import Prediction
from .config import settings
from datetime import datetime, timezone

# ── Connection ────────────────────────────────────────────────


client: AsyncIOMotorClient = None
db = None

async def connect_db():
    """Call on FastAPI startup."""
    global client, db
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.db_name]
    # Index so lookups by prediction_id are fast
    await db.predictions.create_index("prediction_id", unique=True)
    await db.predictions.create_index("created_at")

async def close_db():
    """Call on FastAPI shutdown."""
    client.close()


# ── CRUD helpers ──────────────────────────────────────────────

async def insert_prediction(pred: Prediction) -> str:
    """Insert a new prediction document. Returns the prediction_id."""
    doc = pred.model_dump()
    await db.predictions.insert_one(doc)
    return pred.prediction_id


async def get_prediction(prediction_id: str) -> Optional[dict]:
    """Fetch one prediction by its ID."""
    return await db.predictions.find_one(
        {"prediction_id": prediction_id},
        {"_id": 0}          # exclude Mongo's internal _id from result
    )


async def confirm_prediction(
    prediction_id: str,
    confirmed_class: list[str],
    notes: str,
) -> bool:
    result = await db.predictions.update_one(
        {"prediction_id": prediction_id},
        {"$set": {
            "confirmed_class": confirmed_class,
            "confirmed": True,
            "notes": notes,
            "confirmed_at" : datetime.now(timezone.utc)
        }}
    )
    return result.modified_count == 1


async def list_predictions(
    limit: int = 20,
    offset: int = 0,
    confirmed_only: bool = False,
) -> tuple[list[dict], int]:
    """
    Return a paginated list of predictions.
    Returns (records, total_count) so the caller can build pagination metadata.
    """
    query = {"confirmed": True} if confirmed_only else {}

    total = await db.predictions.count_documents(query)

    cursor = (
        db.predictions
        .find(query, {"_id": 0})
        .sort("created_at", DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    records = await cursor.to_list(length=limit)

    return records, total


async def delete_prediction(prediction_id: str) -> bool:
    result = await db.predictions.delete_one({"prediction_id": prediction_id})
    return result.deleted_count == 1

async def get_conversation(conversation_id: str) -> list[dict]:
    cursor = (
        db.predictions
        .find({"conversation_id": conversation_id}, {"_id": 0})
        .sort("created_at", ASCENDING)
    )
    return await cursor.to_list(length=None)