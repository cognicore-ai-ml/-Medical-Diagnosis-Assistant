import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Medical Diagnosis Assistant API",
    description="An ML-powered API for preliminary symptom analysis.",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomRequest(BaseModel):
    symptoms: List[str]
    days_active: int

# Global variables to hold our ML components
model = None
vectorizer = None

# Load the trained model when the API starts up
@app.on_event("startup")
def load_ml_model():
    global model, vectorizer
    model_path = 'models/symptom_model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("🤖 Machine Learning Model loaded successfully!")
    else:
        print("⚠️ Model files not found. Run models/train.py first to generate them.")

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "message": "Medical Diagnosis Assistant API is running"
    }

@app.post("/api/analyze")
def analyze_symptoms(request: SymptomRequest):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Symptom list cannot be empty")
    
    # If the model hasn't been trained/saved yet, fallback gracefully
    if model is None or vectorizer is None:
        return {
            "analysis": {
                "insights": ["Model is in training mode. Logged: " + ", ".join(request.symptoms)],
                "urgency_level": "low",
                "disclaimer": "Fallback system active. Please train the ML model."
            }
        }
    
    # 1. Prepare input text by joining symptoms into a single string space-separated
    input_text = " ".join([s.lower().strip() for s in request.symptoms])
    
    # 2. Transform the text using our vectorizer and predict using our model
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)[0]
    
    # 3. Handle urgency scaling based on duration or specific high-risk triggers
    urgency = "low"
    if request.days_active > 5 or "stiff neck" in input_text or "shortness of breath" in input_text:
        urgency = "medium"
    if "high risk" in prediction.lower() or request.days_active > 10:
        urgency = "high"

    return {
        "analysis": {
            "insights": [f"Based on our statistical pattern matching, your symptoms align closely with: {prediction}."],
            "urgency_level": urgency,
            "disclaimer": "This is an automated machine learning insight. It is not an alternative to real clinical consultation."
        }
                                                                               }
    
