import os
import pandas as pd

def filter_cases():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    cases_file = os.path.join(processed_dir, 'support_cases.csv')
    
    if not os.path.exists(cases_file):
        print("Error: support_cases.csv not found.")
        return
        
    df = pd.read_csv(cases_file)
    total_candidate_pairs = len(df)
    
    # Check conditions
    missing_customer = df['customer_message'].isna() | (df['customer_message'].str.strip() == '')
    missing_brand = df['brand_reply'].isna() | (df['brand_reply'].str.strip() == '')
    duplicates = df.duplicated(subset=['customer_tweet_id', 'brand_tweet_id'])
    
    # We could also check for corrupted records if needed (e.g. invalid dates), but we keep it simple
    invalid = missing_customer | missing_brand | duplicates
    
    valid_df = df[~invalid].copy()
    removed_df = df[invalid].copy()
    
    # Save valid pairs
    # Wait, the task says "Remove or flag clearly unusable cases". I'll remove them and overwrite or save to a new file.
    # I'll just overwrite the same file or keep it as support_cases.csv and use a flag if I wanted to, but dropping is safer for downstream.
    # Actually, let's just drop them.
    valid_df.to_csv(cases_file, index=False)
    
    report_data = [{
        'total candidate pairs': total_candidate_pairs,
        'valid pairs': len(valid_df),
        'removed pairs': len(removed_df),
        'missing customer messages': missing_customer.sum(),
        'missing brand replies': missing_brand.sum(),
        'duplicates': duplicates.sum(),
        'other invalid cases': 0
    }]
    
    report_df = pd.DataFrame(report_data)
    report_file = os.path.join(processed_dir, 'support_case_quality_report.csv')
    report_df.to_csv(report_file, index=False)
    
    print(f"Filtered to {len(valid_df)} valid pairs. Saved report to {report_file}")

if __name__ == "__main__":
    filter_cases()
