import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model, process_text
from src.llm_validator import validate_sarcasm

app = FastAPI()
model = load_model()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    ml_prediction: str
    gemini_validation: str

@app.get("/")
def read_root():
    return {"message": "Sarcasm Detection API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOut)
def predict(text_in: TextIn):
    try:
        ml_prediction = process_text(text_in.text, model)
        gemini_validation = validate_sarcasm(text_in.text, ml_prediction)
        return {"ml_prediction": ml_prediction, "gemini_validation": gemini_validation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

