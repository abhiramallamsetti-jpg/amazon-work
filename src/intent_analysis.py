import os
import pandas as pd
import re

def analyze_intents():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    cases_file = os.path.join(processed_dir, 'support_cases.csv')
    
    if not os.path.exists(cases_file):
        print("Error: support_cases.csv not found.")
        return
        
    df = pd.read_csv(cases_file)
    
    # Define keywords for SpotifyCares
    intent_rules = {
        'billing_issue': ['charge', 'charged', 'billing', 'pay', 'payment', 'credit card', 'refund', 'money'],
        'subscription_issue': ['premium', 'family', 'student', 'subscription', 'plan', 'upgrade', 'downgrade', 'cancel'],
        'account_access': ['login', 'log in', 'password', 'hacked', 'email', 'account', 'sign in'],
        'playback_issue': ['play', 'playing', 'stop', 'pause', 'skip', 'skipping', 'offline', 'download', 'buffer', 'quality'],
        'app_bug': ['crash', 'crashing', 'bug', 'update', 'glitch', 'won\'t open', 'blank', 'error'],
        'feature_inquiry': ['lyrics', 'podcast', 'playlist', 'add', 'feature', 'when', 'missing']
    }
    
    def assign_intent(text):
        if pd.isna(text):
            return 'unknown'
        text_lower = text.lower()
        for intent, keywords in intent_rules.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return 'other'
        
    df['intent'] = df['customer_message'].apply(assign_intent)
    
    # 1. Intent Examples
    examples_list = []
    for intent in df['intent'].unique():
        # sample up to 5 examples per intent
        subset = df[df['intent'] == intent].head(5)
        for _, row in subset.iterrows():
            examples_list.append({
                'example_id': 'EX_' + str(row['case_id']),
                'customer_message': row['customer_message'],
                'intent': row['intent'],
                'conversation_id': row['conversation_id'],
                'brand_reply': row['brand_reply']
            })
            
    examples_df = pd.DataFrame(examples_list)
    examples_file = os.path.join(processed_dir, 'intent_examples.csv')
    examples_df.to_csv(examples_file, index=False)
    
    # 2. Intent Distribution
    distribution = df['intent'].value_counts().reset_index()
    distribution.columns = ['intent', 'count']
    distribution['percentage'] = (distribution['count'] / len(df)) * 100
    
    dist_file = os.path.join(processed_dir, 'intent_distribution.csv')
    distribution.to_csv(dist_file, index=False)
    
    # 3. Intent Definitions
    definitions = []
    for intent in df['intent'].unique():
        definitions.append({
            'intent': intent,
            'description': f"Customer issues related to {intent.replace('_', ' ')}",
            'include_examples': "See intent_examples.csv for examples",
            'exclude_examples': "N/A",
            'expected_action': "Provide relevant troubleshooting or account link",
            'escalation_risk': "low"
        })
    
    def_df = pd.DataFrame(definitions)
    def_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'intent_definitions.csv')
    def_df.to_csv(def_file, index=False)
    
    print(f"Discovered {len(df['intent'].unique())} intents. Saved examples, distribution, and definitions.")

if __name__ == "__main__":
    analyze_intents()
