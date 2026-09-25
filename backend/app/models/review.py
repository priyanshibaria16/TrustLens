from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from datetime import datetime
from ..database import Base

class ReviewHistory(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(String, unique=True, index=True)
    review_text = Column(Text, nullable=False)
    trust_score = Column(Integer)
    trust_level = Column(String)
    sentiment_label = Column(String)
    sentiment_polarity = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
