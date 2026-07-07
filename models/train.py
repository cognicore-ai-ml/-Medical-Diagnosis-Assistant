import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

# 1. Sample Dataset for Prototyping
# In a full-scale app, this would load a massive CSV file of medical cases.
symptoms_data = [
    "fever cough fatigue sore throat",
    "cough sneezing runny nose congestion",
    "fever chills cough shortness of breath",
    "headache nausea sensitivity to light",
    "itchy eyes sneezing runny nose watery eyes",
    "fever body aches fatigue chills cough",
    "abdominal pain nausea vomiting diarrhea",
    "headache stiff neck high fever confusion"
]

diagnoses = [
    "Common Cold",
    "Allergies",
    "Infection/Flu-like symptoms",
    "Migraine",
    "Allergies",
    "Influenza (Flu)",
    "Gastroenteritis",
    "High Risk - Medical Evaluation Required"
]

def train_diagnostic_model():
    print("Initializing model training...")
    
    # 2. Text Preprocessing: Convert symptom strings into numbers the AI can read
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(symptoms_data)
    y = diagnoses

    # 3. Model Selection: Train a Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    print("Model training complete!")

    # 4. Create the models directory if it doesn't exist locally
    os.makedirs('models', exist_ok=True)

    # 5. Save both the model and the vectorizer to disk so FastAPI can use them later
    with open('models/symptom_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
    with open('models/vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
        
    print("Saved 'symptom_model.pkl' and 'vectorizer.pkl' successfully.")

if __name__ == "__main__":
    train_diagnostic_model()
  
