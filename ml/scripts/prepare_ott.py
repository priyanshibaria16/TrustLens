import pandas as pd
import numpy as np
import json
import os

RAW_PATH = os.path.join("ml", "data", "raw", "ott", "deceptive-opinion.csv")
PROCESSED_PATH = os.path.join("ml", "data", "processed", "ott_reviews.csv")
REPORT_PATH = os.path.join("ml", "data", "processed", "data_quality_report.json")

def prepare_data():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)
    
    initial_count = len(df)
    
    # 1. Clean missing texts
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    valid_count = len(df)
    
    # 2. Handle Duplicates
    # We will keep duplicates for tracking in report, but we will remove exact duplicates for training dataset
    dup_count = df.duplicated(subset=['text']).sum()
    df = df.drop_duplicates(subset=['text'])
    
    # 3. Create review lengths (for report)
    df['review_length'] = df['text'].apply(lambda x: len(x.split()))
    
    # 4. Normalize Labels
    # deceptive column contains 'truthful' and 'deceptive'
    # 0 = truthful, 1 = deceptive
    df['label'] = df['deceptive'].apply(lambda x: 0 if x == 'truthful' else 1)
    df['label_name'] = df['deceptive']
    df['dataset'] = 'ott'
    
    # 5. Create Review IDs
    df = df.reset_index(drop=True)
    df['review_id'] = [f"{i+1:04d}" for i in range(len(df))]
    
    # Rename columns to match requested structure
    df = df.rename(columns={'text': 'review_text'})
    
    # Final Columns
    cols_to_keep = ['review_id', 'review_text', 'label', 'label_name', 'dataset']
    final_df = df[cols_to_keep]
    
    print(f"Saving {len(final_df)} records to {PROCESSED_PATH}")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_PATH, index=False)
    
    # Generate Data Quality Report
    truthful_count = len(df[df['label'] == 0])
    deceptive_count = len(df[df['label'] == 1])
    min_length = int(df['review_length'].min())
    max_length = int(df['review_length'].max())
    avg_length = float(df['review_length'].mean())
    
    report = {
        "total_records_initial": initial_count,
        "total_records_processed": len(final_df),
        "truthful_count": truthful_count,
        "deceptive_count": deceptive_count,
        "missing_records": initial_count - valid_count,
        "duplicate_records_removed": int(dup_count),
        "minimum_review_length_words": min_length,
        "maximum_review_length_words": max_length,
        "average_review_length_words": avg_length,
        "class_distribution": {
            "truthful_percentage": round(truthful_count / len(final_df) * 100, 2),
            "deceptive_percentage": round(deceptive_count / len(final_df) * 100, 2)
        }
    }
    
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Data Quality Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    prepare_data()
