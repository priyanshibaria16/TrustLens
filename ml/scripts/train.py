import pandas as pd
import numpy as np
import json
import os
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add backend directory to sys.path to import FeatureExtractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.app.nlp.feature_extraction import FeatureExtractor

DATA_PATH = os.path.join("ml", "data", "processed", "ott_reviews.csv")
MODELS_DIR = os.path.join("ml", "models")
RESULTS_DIR = os.path.join("ml", "experiments")

def extract_all_features(df):
    extractor = FeatureExtractor()
    print("Extracting NLP features...")
    features_list = []
    for text in df['review_text']:
        features_list.append(extractor.extract_features(text))
        
    features_df = pd.DataFrame(features_list)
    # Combine with original df
    return pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Extract features
    df = extract_all_features(df)
    
    # Features to use for modeling
    numeric_features = [
        "char_count", "word_count", "sentence_count", "avg_sentence_length", 
        "avg_word_length", "lexical_diversity", "repetition_ratio",
        "specificity_score", "personal_experience_score", "evidence_score",
        "sentiment_polarity", "promotional_language_score", "exaggeration_score",
        "linguistic_quality", "uppercase_ratio", "exclamation_count",
        "question_mark_count", "repeated_punct"
    ]
    
    X_text = df['review_text']
    X_numeric = df[numeric_features]
    y = df['label']
    
    # Train/Test Split
    print("Splitting data...")
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test set for evaluate.py
    os.makedirs(os.path.join(MODELS_DIR, "test_data"), exist_ok=True)
    X_text_test.to_csv(os.path.join(MODELS_DIR, "test_data", "X_text_test.csv"), index=False)
    X_num_test.to_csv(os.path.join(MODELS_DIR, "test_data", "X_num_test.csv"), index=False)
    y_test.to_csv(os.path.join(MODELS_DIR, "test_data", "y_test.csv"), index=False)
    
    # Fit TF-IDF on train only
    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5, max_df=0.7, stop_words='english')
    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf = tfidf.transform(X_text_test)
    
    # Save TF-IDF Vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    
    # Scale numeric features
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # Combine TF-IDF and scaled numeric features
    X_train_combined = np.hstack((X_text_train_tfidf.toarray(), X_num_train_scaled))
    X_test_combined = np.hstack((X_text_test_tfidf.toarray(), X_num_test_scaled))
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Linear SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    best_f1 = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        y_prob = model.predict_proba(X_test_combined)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    # Save Best Model
    print(f"Best model based on F1: {best_model_name}")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    
    # Save model metadata
    model_metadata = {
        "model_name": best_model_name,
        "dataset": "ott",
        "features": {
            "text": "TF-IDF (max 2000 features, unigrams+bigrams)",
            "linguistic": numeric_features
        },
        "random_seed": 42,
        "evaluation_metrics": results_df[results_df['Model'] == best_model_name].to_dict('records')[0],
        "training_date": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=4)
        
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
