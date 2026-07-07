
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Medical Diagnosis Assistant API",
    description="A prototype API for preliminary symptom analysis.",
    version="0.1.0"
)

# Enable CORS so your frontend (localhost:5173) can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the structure of the incoming user request
class SymptomRequest(BaseModel):
    symptoms: List[str]
    days_active: int

# Root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical Diagnosis Assistant API is running"}

# Analysis endpoint (Mock logic for prototyping)
@app.post("/api/analyze")
def analyze_symptoms(request: SymptomRequest):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Symptom list cannot be empty")
    
    # Normalize inputs to lowercase for basic matching
    symptoms_lower = [s.lower() for s in request.symptoms]
    
    # Placeholder rule-based engine before we integrate real ML models
    preliminary_insights = []
    urgency = "low"
    
    if "fever" in symptoms_lower or "cough" in symptoms_lower:
        preliminary_insights.append("Possible mild respiratory infection.")
    if "headache" in symptoms_lower and request.days_active > 3:
        preliminary_insights.append("Persistent headache detected. Monitor hydration and screen time.")
        urgency = "medium"
    
    # Fallback if no rules match
    if not preliminary_insights:
        preliminary_insights.append("Symptoms logged. General fatigue or minor strain.")

    return {
        "analysis": {
            "insights": preliminary_insights,
            "urgency_level": urgency,
            "disclaimer": "This is a machine-generated prototype insight. Consult a doctor."
        }
    }
