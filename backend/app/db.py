from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_NAME = os.getenv("DB_NAME", "fetal_detection")

client = MongoClient(DATABASE_URL)
db = client[DB_NAME]

def get_db():
    return db
