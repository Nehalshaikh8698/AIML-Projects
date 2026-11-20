import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

try:
    from bson import ObjectId
    from pymongo import MongoClient, DESCENDING
except ImportError:  # MongoDB is optional for local in-memory development.
    ObjectId = None
    MongoClient = None
    DESCENDING = -1

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    if "_id" in result:
        result["id"] = str(result.pop("_id"))
    return result


class MemoryStore:
    """Fallback store for local API work when MongoDB is not reachable."""

    def __init__(self) -> None:
        self.studies: dict[str, dict[str, Any]] = {}
        self.audit: dict[str, list[dict[str, Any]]] = {}
        self.lock = Lock()
        self.persistent = False

    def create_study(self, document: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            study_id = document.get("id") or uuid4().hex
            saved = {**document, "id": study_id}
            self.studies[study_id] = saved
            return saved

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.studies.get(study_id)

    def list_studies(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(self.studies.values(), key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    def append_audit(self, study_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.audit.setdefault(study_id, []).append(event)

    def get_audit(self, study_id: str) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.audit.get(study_id, []))

    def update_review(self, study_id: str, review: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            if study_id not in self.studies:
                return None
            self.studies[study_id]["review"] = review
            self.studies[study_id]["review_status"] = review["decision"]
            self.studies[study_id]["updated_at"] = utc_now()
            return self.studies[study_id]


class MongoStore:
    def __init__(self, uri: str, database: str) -> None:
        if MongoClient is None:
            raise RuntimeError("pymongo is not installed; using in-memory fallback")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=2500)
        self.client.admin.command("ping")
        db = self.client[database]
        self.studies = db["studies"]
        self.audit = db["audit_events"]
        self.persistent = True
        self.studies.create_index("created_at")
        self.audit.create_index([("study_id", 1), ("created_at", 1)])

    def create_study(self, document: dict[str, Any]) -> dict[str, Any]:
        saved = dict(document)
        saved.pop("id", None)
        result = self.studies.insert_one(saved)
        saved["id"] = str(result.inserted_id)
        return saved

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        if ObjectId is None or not ObjectId.is_valid(study_id):
            return None
        return serialize_document(self.studies.find_one({"_id": ObjectId(study_id)}))

    def list_studies(self, limit: int = 50) -> list[dict[str, Any]]:
        return [serialize_document(x) for x in self.studies.find().sort("created_at", DESCENDING).limit(limit)]

    def append_audit(self, study_id: str, event: dict[str, Any]) -> None:
        self.audit.insert_one({"study_id": study_id, **event})

    def get_audit(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.audit.find({"study_id": study_id}).sort("created_at", 1)
        return [serialize_document(row) for row in rows]

    def update_review(self, study_id: str, review: dict[str, Any]) -> dict[str, Any] | None:
        if ObjectId is None or not ObjectId.is_valid(study_id):
            return None
        self.studies.update_one(
            {"_id": ObjectId(study_id)},
            {"$set": {"review": review, "review_status": review["decision"], "updated_at": utc_now()}},
        )
        return self.get_study(study_id)


def create_store(uri: str, database: str):
    try:
        store = MongoStore(uri, database)
        logger.info("Connected to MongoDB; audit events are persistent.")
        return store
    except Exception as exc:  # fallback keeps local dev usable
        logger.warning("MongoDB unavailable (%s). Using in-memory storage; data is temporary.", exc)
        return MemoryStore()
