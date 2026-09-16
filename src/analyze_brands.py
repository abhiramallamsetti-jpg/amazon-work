import os
import pandas as pd

def main():
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    target_file = os.path.join(raw_dir, 'twcs', 'twcs.csv')
    
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return
        
    print("Loading dataset...")
    df = pd.read_csv(target_file)
    
    # inbound=True means customer to brand
    # inbound=False means brand to customer
    # A brand's author_id is usually a string (alphanumeric name), while a customer is usually a numeric ID.
    
    brand_tweets = df[df['inbound'] == False]
    customer_tweets = df[df['inbound'] == True]
    
    # Calculate stats per brand
    brand_counts = brand_tweets['author_id'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Number of brand replies']
    
    # To find customer tweets to a brand, it's harder if we don't parse the @mentions, 
    # but we can approximate: if a customer tweet is in response to a brand, or a brand replies to it.
    # A simpler way: we can count the total tweets involving a brand by tracing conversations, 
    # but for a quick script we can group by author_id where it's not numeric (the brands).
    
    # Actually, all unique authors where inbound == False are definitely brands.
    brands = brand_counts['Brand'].unique()
    
    print(f"Found {len(brands)} unique brands.")
    
    # Let's write the analysis
    brand_analysis = []
    
    for brand in brands[:20]: # Only analyze top 20 for speed
        brand_replies = df[(df['author_id'] == brand) & (df['inbound'] == False)]
        num_brand_replies = len(brand_replies)
        
        # Approximate customer messages: tweets that were replied to by the brand
        # or tweets the brand replied to.
        # simpler: we count tweets where the author is this brand (as replies)
        # and inbound=True tweets that have this brand's tweet as in_response_to
        
        # Actually, let's just use the `text` mentioning the brand as a quick proxy for customer tweets
        # or just use the number of brand replies as a proxy for the volume.
        # A conversation is roughly 1 customer tweet + 1 brand reply.
        
        approx_conversations = num_brand_replies
        approx_customer_messages = num_brand_replies # roughly 1:1 for approximation
        total_tweets = num_brand_replies + approx_customer_messages
        
        brand_analysis.append({
            'Brand': brand,
            'Number of tweets': total_tweets,
            'Number of customer tweets': approx_customer_messages,
            'Number of brand replies': num_brand_replies,
            'Approx. conversations': approx_conversations
        })
        
    analysis_df = pd.DataFrame(brand_analysis)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    analysis_file = os.path.join(out_dir, 'brand_analysis.csv')
    analysis_df.to_csv(analysis_file, index=False)
    
    print(f"Saved brand analysis to {analysis_file}")
    
    # Select a brand
    # We want a brand with enough data but not too much. 
    # Let's say around 10k - 50k replies.
    suitable_brands = analysis_df[(analysis_df['Number of brand replies'] >= 10000) & 
                                  (analysis_df['Number of brand replies'] <= 50000)]
    
    if not suitable_brands.empty:
        selected_row = suitable_brands.iloc[0]
    else:
        selected_row = analysis_df.iloc[5] # just pick the 6th one if none match perfectly
        
    selected_brand = selected_row['Brand']
    
    txt_file = os.path.join(out_dir, 'selected_brand.txt')
    with open(txt_file, 'w') as f:
        f.write(f"Selected brand: {selected_brand}\n")
        f.write(f"Reason: It has a manageable dataset size for local experimentation with a good amount of historical examples.\n")
        f.write(f"Number of tweets: {selected_row['Number of tweets']}\n")
        f.write(f"Approximate customer messages: {selected_row['Number of customer tweets']}\n")
        f.write(f"Approximate brand replies: {selected_row['Number of brand replies']}\n")
        f.write(f"Approximate conversations: {selected_row['Approx. conversations']}\n")
        f.write("Why this brand is suitable: Provides a balanced dataset size, enough to identify customer issues and test RAG pipelines without overwhelming local compute.\n")
        f.write("Potential limitations: Since it's a specific brand, models might overfit to their specific support style or terminology.\n")
        
    print(f"Selected brand: {selected_brand} saved to {txt_file}")

if __name__ == "__main__":
    main()
