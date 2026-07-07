# Medical Diagnosis Assistant 🩺

An open-source AI assistant designed to help analyze symptoms and medical data to provide preliminary diagnostic insights. 

> ⚠️ **CRITICAL DISCLAIMER:** This project is strictly for educational, research, and prototyping purposes. It is NOT a certified medical device and must NOT be used for clinical decision-making or to replace professional medical advice, diagnosis, or treatment.

---

## 🏗️ Project Architecture

The repository is organized as a monorepo split into modular components:

* **`backend/`**: A Python-based API (FastAPI) that serves the machine learning models.
* **`frontend/`**: A user interface (React or Vue) for symptoms input and displaying insights.
* **`models/`**: Scripts for data preprocessing, model training, and evaluation.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* Node.js 18+
* Docker (Optional, for containerization)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/medical-diagnosis-assistant.git](https://github.com/YOUR_USERNAME/medical-diagnosis-assistant.git)
   cd medical-diagnosis-assistant



### 📡 Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
### 💻 Frontend Setup

```bash
cd ../frontend
npm install
npm run dev
```




   
