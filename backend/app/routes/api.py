from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json
import os
from ..database import get_db
from ..models.review import ReviewHistory
from ..schemas.review import ReviewRequest, AnalysisResponse, ReviewHistorySchema
from ..services.prediction_service import prediction_service

router = APIRouter(prefix="/api")

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/analyze", response_model=AnalysisResponse)
def analyze_review(request: ReviewRequest, db: Session = Depends(get_db)):
    result = prediction_service.analyze_review(request.review_text)
    
    # Save to history
    db_review = ReviewHistory(
        review_id=result["review_id"],
        review_text=request.review_text,
        trust_score=result["trust_score"],
        trust_level=result["trust_level"],
        sentiment_label=result["sentiment"]["label"],
        sentiment_polarity=result["sentiment"]["polarity"],
        confidence=result["confidence"]
    )
    db.add(db_review)
    db.commit()
    
    return result

@router.get("/reviews", response_model=List[ReviewHistorySchema])
def get_reviews(db: Session = Depends(get_db)):
    return db.query(ReviewHistory).order_by(ReviewHistory.timestamp.desc()).all()

@router.get("/reviews/{review_id}", response_model=ReviewHistorySchema)
def get_review(review_id: str, db: Session = Depends(get_db)):
    review = db.query(ReviewHistory).filter(ReviewHistory.review_id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

@router.delete("/reviews/{review_id}")
def delete_review(review_id: str, db: Session = Depends(get_db)):
    review = db.query(ReviewHistory).filter(ReviewHistory.review_id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    db.delete(review)
    db.commit()
    return {"success": True}

@router.get("/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    reviews = db.query(ReviewHistory).all()
    total = len(reviews)
    if total == 0:
        return {"total_reviews": 0}
        
    avg_score = sum([r.trust_score for r in reviews]) / total
    high_trust = len([r for r in reviews if r.trust_level == "High"])
    mod_trust = len([r for r in reviews if r.trust_level == "Moderate"])
    low_trust = len([r for r in reviews if r.trust_level == "Low"])
    
    # Sentiment distribution
    pos = len([r for r in reviews if r.sentiment_label == "positive"])
    neg = len([r for r in reviews if r.sentiment_label == "negative"])
    neu = len([r for r in reviews if r.sentiment_label == "neutral"])
    mixed = len([r for r in reviews if r.sentiment_label == "mixed"])
    
    return {
        "total_reviews": total,
        "average_trust_score": round(avg_score, 1),
        "high_trust_reviews": high_trust,
        "moderate_trust_reviews": mod_trust,
        "low_trust_reviews": low_trust,
        "sentiment_distribution": {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "mixed": mixed
        }
    }

@router.get("/model-performance")
def get_model_performance():
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models")
    EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "evaluation")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "experiments")
    
    try:
        with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
            metadata = json.load(f)
            
        import pandas as pd
        comparison_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
        comparison = comparison_df.to_dict('records')
        
        with open(os.path.join(EVAL_DIR, "classification_report.json"), "r") as f:
            report = json.load(f)
            
        return {
            "metadata": metadata,
            "comparison": comparison,
            "classification_report": report
        }
    except Exception as e:
        return {"error": str(e), "message": "Evaluation data not available yet. Please run training pipeline."}

@router.get("/methodology")
def get_methodology():
    return {
        "dataset": "Ott Deceptive Opinion Spam Corpus (approx. 1,600 balanced hotel reviews)",
        "preprocessing": "Tokenization, lemmatization, stopword removal, syntax parsing using spaCy",
        "feature_extraction": "Extraction of text features (specificity, personal experience, evidence), sentiment via VADER, and style metrics.",
        "modeling": "TF-IDF + extracted numerical features using standard ML algorithms (Random Forest, SVM, Logistic Regression)",
        "trust_score": "Heuristic combination of positive linguistic signals and negative stylistic signals bounded to 0-100."
    }
