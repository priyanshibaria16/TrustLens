import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

MODELS_DIR = os.path.join("ml", "models")
EVAL_DIR = os.path.join("ml", "evaluation")
RESEARCH_RESULTS_DIR = os.path.join("research", "results")

def evaluate_model():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(RESEARCH_RESULTS_DIR, exist_ok=True)
    
    print("Loading models and test data...")
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    
    X_text_test = pd.read_csv(os.path.join(MODELS_DIR, "test_data", "X_text_test.csv"))['review_text']
    X_num_test = pd.read_csv(os.path.join(MODELS_DIR, "test_data", "X_num_test.csv"))
    y_test = pd.read_csv(os.path.join(MODELS_DIR, "test_data", "y_test.csv"))['label']
    
    print("Transforming test data...")
    X_text_test_tfidf = tfidf.transform(X_text_test)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_test_combined = np.hstack((X_text_test_tfidf.toarray(), X_num_test_scaled))
    
    y_pred = model.predict(X_test_combined)
    y_prob = model.predict_proba(X_test_combined)[:, 1]
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred, target_names=["Truthful", "Deceptive"], output_dict=True)
    with open(os.path.join(EVAL_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Truthful", "Deceptive"], yticklabels=["Truthful", "Deceptive"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
    plt.savefig(os.path.join(RESEARCH_RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_curve.png"))
    plt.savefig(os.path.join(RESEARCH_RESULTS_DIR, "roc_curve.png"))
    plt.close()
    
    print("Evaluation artifacts generated in ml/evaluation and research/results")

if __name__ == "__main__":
    evaluate_model()
