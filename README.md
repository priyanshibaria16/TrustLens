# TrustLens

**An Explainable NLP Framework for Assessing the Trustworthiness of Online Reviews**

## Project Overview
TrustLens is an academic research project that provides an explainable AI framework to estimate the textual trustworthiness of online reviews. Using the **Ott Deceptive Opinion Spam Corpus** as the sole supervised learning dataset, it integrates a sophisticated NLP feature extraction pipeline with classical machine learning models. 

The framework does not claim to establish factual truth. Instead, it highlights observable stylistic, linguistic, and semantic signals—such as specific details, personal experiences, exaggerated language, and promotional phrasing—that contribute to the overall textual trustworthiness of a review.

## Architecture
- **Backend**: Python, FastAPI, Scikit-Learn, spaCy
- **Frontend**: React, Vite, Tailwind CSS, Recharts
- **Database**: SQLite (for review history)
- **Dataset**: Ott Deceptive Opinion Spam Corpus (~1,600 balanced hotel reviews)

## Installation & Setup (Windows PowerShell)

### 1. Python Environment Setup
```powershell
cd TrustLens
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```
*(Dependencies include fastapi, uvicorn, pydantic, sqlalchemy, pandas, scikit-learn, spacy, etc.)*
Download the spaCy model:
```powershell
python -m spacy download en_core_web_sm
```

### 2. Dataset Preparation
Ensure `deceptive-opinion.csv` is located in `ml/data/raw/ott/`.
Run the data preparation script:
```powershell
python ml\scripts\prepare_ott.py
```

### 3. Training & Evaluation Pipeline
Run the ML pipeline to extract features, train the Linear SVM, and evaluate:
```powershell
python ml\scripts\train.py
python ml\scripts\evaluate.py
```
*Note: This will generate `model_comparison.csv` and ROC/Confusion Matrix charts in the `ml/evaluation` folder.*

### 4. Running the Backend API
Start the FastAPI server:
```powershell
uvicorn backend.app.main:app --reload
```
*The API runs at `http://localhost:8000`. Swagger UI is at `http://localhost:8000/docs`.*

### 5. Running the Frontend
In a new PowerShell window:
```powershell
cd TrustLens\frontend
npm install
npm run dev
```
*The React app will be available at `http://localhost:5173`.*

## Research Methodology & Limitations
- The system was trained on a single dataset (hotel reviews) and may not generalize optimally to all domains.
- High model confidence reflects stylistic alignment with the dataset's deceptive patterns, not factual verification.
- Trust Score heuristics and thresholds are framework-defined prototypes for academic demonstration.

## Citation
If using this project for academic reference, please note it relies on:
Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.
