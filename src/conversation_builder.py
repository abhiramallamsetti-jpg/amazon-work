import os
import pandas as pd

def build_conversations():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    brand_file = os.path.join(processed_dir, 'selected_brand_tweets.csv')
    
    if not os.path.exists(brand_file):
        print("Error: selected_brand_tweets.csv not found.")
        return
        
    df = pd.read_csv(brand_file)
    
    # Customer tweets (inbound == True)
    customers = df[df['inbound'] == True]
    
    # Brand tweets (inbound == False)
    brands = df[df['inbound'] == False]
    
    # A valid pair is Customer Tweet -> Brand Reply
    # The Brand Reply has in_response_to_tweet_id == Customer Tweet's tweet_id
    
    # Merge brand replies with customer tweets
    cases = pd.merge(
        customers,
        brands,
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_cust', '_brand')
    )
    
    # Calculate reply time
    cases['created_at_cust'] = pd.to_datetime(cases['created_at_cust'], errors='coerce')
    cases['created_at_brand'] = pd.to_datetime(cases['created_at_brand'], errors='coerce')
    
    cases['reply_time_minutes'] = (cases['created_at_brand'] - cases['created_at_cust']).dt.total_seconds() / 60.0
    
    # Select useful columns
    support_cases = cases[[
        'tweet_id_cust', 
        'tweet_id_brand', 
        'author_id_cust', 
        'author_id_brand',
        'text_cust', 
        'text_brand', 
        'created_at_cust', 
        'created_at_brand', 
        'reply_time_minutes'
    ]].copy()
    
    support_cases.rename(columns={
        'tweet_id_cust': 'customer_tweet_id',
        'tweet_id_brand': 'brand_tweet_id',
        'author_id_cust': 'customer_user',
        'author_id_brand': 'brand_account',
        'text_cust': 'customer_message',
        'text_brand': 'brand_reply',
        'created_at_cust': 'customer_timestamp',
        'created_at_brand': 'brand_timestamp'
    }, inplace=True)
    
    # Generate conversation_id (we can just use the customer_tweet_id as the root)
    support_cases['conversation_id'] = support_cases['customer_tweet_id'].astype(str)
    
    # Create case_id
    support_cases['case_id'] = 'CASE_' + support_cases['customer_tweet_id'].astype(str)
    
    out_file = os.path.join(processed_dir, 'support_cases.csv')
    support_cases.to_csv(out_file, index=False)
    print(f"Created {len(support_cases)} support cases and saved to {out_file}")

if __name__ == "__main__":
    build_conversations()
