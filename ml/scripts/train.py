import pandas as pd
import numpy as np
import json
import os
import joblib
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.app.nlp.feature_extraction import FeatureExtractor

DATA_PATH = os.path.join("ml", "data", "processed", "ott_reviews.csv")
MODELS_DIR = os.path.join("ml", "models")
RESULTS_DIR = os.path.join("ml", "experiments")


def clean_text(text):
    """Strip trailing/leading whitespace and newlines from reviews."""
    if isinstance(text, str):
        return text.strip().replace('\n', ' ').replace('\r', ' ')
    return text


def extract_all_features(df):
    extractor = FeatureExtractor()
    print("Extracting NLP features...")
    features_list = []
    total = len(df)
    for i, text in enumerate(df['review_text']):
        features_list.append(extractor.extract_features(text))
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{total} reviews...")
    features_df = pd.DataFrame(features_list)
    return pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Clean text (remove trailing newlines from CSV multi-line fields)
    df['review_text'] = df['review_text'].apply(clean_text)

    # Extract engineered features
    df = extract_all_features(df)

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

    # ── Train / Test Split ─────────────────────────────────────────────────
    print("Splitting data (80/20, stratified)...")
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test set for evaluate.py
    test_data_dir = os.path.join(MODELS_DIR, "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    X_text_test.to_csv(os.path.join(test_data_dir, "X_text_test.csv"), index=False)
    X_num_test.to_csv(os.path.join(test_data_dir, "X_num_test.csv"), index=False)
    y_test.to_csv(os.path.join(test_data_dir, "y_test.csv"), index=False)

    # ── TF-IDF: Word n-grams + Character n-grams (Experiment 3 feature set) ─
    print("Fitting TF-IDF vectorizers...")

    # Word-level TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        stop_words='english'
    )
    X_word_train = tfidf_word.fit_transform(X_text_train)
    X_word_test = tfidf_word.transform(X_text_test)

    # Character-level TF-IDF (captures style/punctuation patterns)
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb',
        max_features=3000,
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )
    X_char_train = tfidf_char.fit_transform(X_text_train)
    X_char_test = tfidf_char.transform(X_text_test)

    # Save vectorizers
    joblib.dump(tfidf_word, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_char, os.path.join(MODELS_DIR, "tfidf_char_vectorizer.joblib"))

    # ── Scale Numeric Features ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # ── Combine all features (sparse + dense) ─────────────────────────────
    X_train_combined = hstack([X_word_train, X_char_train, csr_matrix(X_num_train_scaled)])
    X_test_combined  = hstack([X_word_test,  X_char_test,  csr_matrix(X_num_test_scaled)])

    # ── Hyperparameter Tuning for best model ──────────────────────────────
    print("\nTuning LinearSVC with GridSearchCV (5-fold stratified CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid_svm = {'base_estimator__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    base_svm = LinearSVC(max_iter=3000, random_state=42, dual=True)
    cal_svm = CalibratedClassifierCV(base_svm, cv=cv, method='sigmoid')

    # Tune C directly on LinearSVC before calibration for speed
    param_grid_svm2 = {'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    grid_svm = GridSearchCV(LinearSVC(max_iter=3000, random_state=42, dual=True),
                            param_grid_svm2, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_svm.fit(X_train_combined, y_train)
    best_C = grid_svm.best_params_['C']
    print(f"Best SVM C = {best_C}")

    # Final calibrated SVM with best C
    tuned_svm = CalibratedClassifierCV(
        LinearSVC(C=best_C, max_iter=3000, random_state=42, dual=True),
        cv=cv, method='sigmoid'
    )

    # ── Train All Models ───────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(C=10, random_state=42, max_iter=2000, solver='lbfgs'),
        "Linear SVM (Tuned)": tuned_svm,
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    results = []
    best_f1 = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        y_prob = model.predict_proba(X_test_combined)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ROC-AUC={roc:.4f}")

        # 5-fold CV F1 on training data
        cv_f1 = cross_val_score(model, X_train_combined, y_train, cv=cv, scoring='f1', n_jobs=-1)
        print(f"  Cross-val F1 (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc,
            "CV_F1_Mean": cv_f1.mean(),
            "CV_F1_Std": cv_f1.std()
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\n✅ Best model: {best_model_name} (F1 = {best_f1:.4f})")

    # Save Best Model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    # Save feature config so prediction_service can match dimensions
    feature_config = {
        "word_tfidf_features": tfidf_word.max_features,
        "char_tfidf_features": tfidf_char.max_features,
        "numeric_features": numeric_features,
        "use_char_tfidf": True
    }
    with open(os.path.join(MODELS_DIR, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=4)

    # Save model metadata
    best_row = results_df[results_df['Model'] == best_model_name].to_dict('records')[0]
    model_metadata = {
        "model_name": best_model_name,
        "dataset": "ott",
        "features": {
            "text_word": f"TF-IDF word n-grams (max {tfidf_word.max_features}, 1-3grams, sublinear_tf)",
            "text_char": f"TF-IDF char n-grams (max {tfidf_char.max_features}, 3-5grams)",
            "linguistic": numeric_features
        },
        "best_svm_C": best_C,
        "random_seed": 42,
        "evaluation_metrics": best_row,
        "training_date": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=4)

    print("\nAll models, vectorizers, and metadata saved to ml/models/")
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    train_and_evaluate()
