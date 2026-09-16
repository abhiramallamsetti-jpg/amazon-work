import pandas as pd
import re

def clean_text_basic(text):
    """
    Perform basic safe text cleaning on a single string.
    - Handles missing text
    - Normalizes whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to string just in case
    text = str(text)
    
    # Normalize whitespace (replace multiple spaces/newlines with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Optionally normalize URLs (we keep them but standardize the tag if we wanted to, 
    # but the instructions say "preserve original text", so we just strip leading/trailing spaces)
    return text.strip()

def preprocess_dataframe(df, text_column='text'):
    """
    Applies basic cleaning to the dataframe, keeping the original text.
    """
    df = df.copy()
    
    if text_column in df.columns:
        # Keep original text
        df['original_text'] = df[text_column]
        
        # Create clean_text column
        df['clean_text'] = df[text_column].apply(clean_text_basic)
        
    # Remove obvious duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    final_len = len(df)
    if initial_len != final_len:
        print(f"Removed {initial_len - final_len} duplicate rows.")
        
    return df
