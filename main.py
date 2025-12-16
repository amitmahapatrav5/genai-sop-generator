from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.driver import extract_features


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def fun(file: UploadFile):
    features: BaseModel = extract_features(file.file)
    return features.model_dump()

# Created By Amit Mahapatra