from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ReviewRequest(BaseModel):
    review_text: str

class SentimentSchema(BaseModel):
    label: str
    polarity: float

class FeaturesSchema(BaseModel):
    specificity: float
    personal_experience: float
    evidence: float
    linguistic_quality: float
    promotion: float
    exaggeration: float

class AnalysisResponse(BaseModel):
    success: bool
    review_id: str
    trust_score: int
    trust_level: str
    confidence: float
    sentiment: SentimentSchema
    features: FeaturesSchema
    positive_contributors: List[str]
    negative_contributors: List[str]
    explanation: str
    warnings: List[str]

class ReviewHistorySchema(BaseModel):
    id: int
    review_id: str
    review_text: str
    trust_score: int
    trust_level: str
    sentiment_label: str
    sentiment_polarity: float
    confidence: float
    timestamp: datetime

    class Config:
        from_attributes = True
