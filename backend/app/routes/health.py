from fastapi import APIRouter, HTTPException
from app.db import client

router = APIRouter()

@router.get("/")
def root():
    return {"message": "FastAPI Backend Running"}

@router.get("/test-db")
def test_db_connection():
    try:
        client.admin.command('ping')
        return {"status": "success", "message": "Database connection successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
