import os
import pandas as pd

def main():
    base_dir = os.path.dirname(__file__)
    
    # Read selected brand
    brand_file = os.path.join(base_dir, '..', 'data', 'selected_brand.txt')
    if not os.path.exists(brand_file):
        print("Error: selected_brand.txt not found. Run brand analysis first.")
        return
        
    selected_brand = None
    with open(brand_file, 'r') as f:
        for line in f:
            if line.startswith("Selected brand:"):
                selected_brand = line.split(":", 1)[1].strip()
                break
                
    if not selected_brand:
        print("Error: Could not parse selected brand.")
        return
        
    print(f"Filtering dataset for brand: {selected_brand}")
    
    raw_file = os.path.join(base_dir, '..', 'data', 'raw', 'twcs', 'twcs.csv')
    if not os.path.exists(raw_file):
        print(f"File not found: {raw_file}")
        return
        
    print("Loading raw dataset...")
    df = pd.read_csv(raw_file)
    
    # The brand's own tweets
    brand_tweets = df[df['author_id'] == selected_brand]
    
    # We also need the customer tweets that are part of the conversation.
    # To keep it simple but functional: 
    # Get all tweets where the customer is replying to the brand's tweets
    brand_tweet_ids = set(brand_tweets['tweet_id'])
    
    # Tweets responding to the brand
    customer_replies_to_brand = df[df['in_response_to_tweet_id'].isin(brand_tweet_ids)]
    
    # Customer tweets that the brand responded to
    customer_tweet_ids_brand_responded_to = set(brand_tweets['in_response_to_tweet_id'].dropna())
    customer_tweets_brand_responded_to = df[df['tweet_id'].isin(customer_tweet_ids_brand_responded_to)]
    
    # Combine and drop duplicates
    filtered_df = pd.concat([brand_tweets, customer_replies_to_brand, customer_tweets_brand_responded_to]).drop_duplicates(subset=['tweet_id'])
    
    out_file = os.path.join(base_dir, '..', 'data', 'processed', 'selected_brand_tweets.csv')
    filtered_df.to_csv(out_file, index=False)
    
    print(f"Saved {len(filtered_df)} tweets related to {selected_brand} to {out_file}")

if __name__ == "__main__":
    main()
