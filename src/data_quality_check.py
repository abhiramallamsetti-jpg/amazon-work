import os
import pandas as pd
from datetime import datetime

def check_data_quality():
    processed_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'selected_brand_tweets.csv')
    
    if not os.path.exists(processed_file):
        print(f"Error: {processed_file} not found. Run filter_brand.py first.")
        return
        
    print("Loading processed dataset...")
    df = pd.read_csv(processed_file)
    
    print("\n--- Data Quality Report ---")
    
    # 1. Missing text
    missing_text = df['text'].isna().sum()
    print(f"Missing text entries: {missing_text}")
    
    # 2. Duplicate rows (exact)
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")
    
    # 3. Invalid timestamps
    invalid_timestamps = 0
    try:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        invalid_timestamps = df['created_at'].isna().sum()
    except Exception as e:
        invalid_timestamps = "Could not determine reliably"
    print(f"Invalid timestamps: {invalid_timestamps}")
    
    # 4. Extremely short messages
    # Messages with less than 2 characters
    if missing_text == 0:
        short_msgs = df[df['text'].str.len() < 2].shape[0]
    else:
        short_msgs = df[df['text'].notna() & (df['text'].str.len() < 2)].shape[0]
    print(f"Extremely short messages (<2 chars): {short_msgs}")
    
    # 5. Extremely long messages (Twitter max is 280, let's say > 300 is weird)
    if missing_text == 0:
        long_msgs = df[df['text'].str.len() > 300].shape[0]
    else:
        long_msgs = df[df['text'].notna() & (df['text'].str.len() > 300)].shape[0]
    print(f"Extremely long messages (>300 chars): {long_msgs}")
    
    # 6. Missing reply relationships
    missing_reply = df['in_response_to_tweet_id'].isna().sum()
    print(f"Missing reply relationships (could be start of convo): {missing_reply}")
    
    # 7. Unexpected values in inbound (should be boolean)
    unexpected_inbound = df[~df['inbound'].isin([True, False, 'True', 'False'])].shape[0]
    print(f"Unexpected values in 'inbound' column: {unexpected_inbound}")
    
    print("---------------------------")
    print("Data quality check complete. No rows were automatically deleted.")

if __name__ == "__main__":
    check_data_quality()
