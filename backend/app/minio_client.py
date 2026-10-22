import io
import json
import time

from minio import Minio
from minio.error import S3Error

from .config import settings

_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def ensure_buckets(retries: int = 10, delay: float = 2.0) -> None:
    client = get_client()
    for attempt in range(retries):
        try:
            for bucket in [settings.data_bucket, settings.models_bucket]:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"MinIO not ready (attempt {attempt + 1}/{retries}): {exc}")
            time.sleep(delay)


def upload_bytes(bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    get_client().put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type)


def get_bytes(bucket: str, object_name: str) -> bytes:
    response = get_client().get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def upload_json(bucket: str, object_name: str, data: dict) -> None:
    encoded = json.dumps(data, ensure_ascii=False).encode()
    upload_bytes(bucket, object_name, encoded, content_type="application/json")


def get_json(bucket: str, object_name: str) -> dict:
    return json.loads(get_bytes(bucket, object_name))


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    try:
        return [obj.object_name for obj in get_client().list_objects(bucket, prefix=prefix, recursive=True)]
    except S3Error:
        return []


def object_exists(bucket: str, object_name: str) -> bool:
    try:
        get_client().stat_object(bucket, object_name)
        return True
    except S3Error:
        return False