import os
import pandas as pd
import glob

def main():
    # 1. Detect dataset files inside data/raw/
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    csv_files = glob.glob(os.path.join(raw_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("No CSV files found in data/raw/")
        return
        
    # We will pick the largest file, which is likely the main dataset
    csv_files.sort(key=os.path.getsize, reverse=True)
    target_file = csv_files[0]
    
    print(f"Detected dataset: {target_file}")
    
    # 2 & 3. Load the dataset safely
    try:
        # For a very large dataset, we just read the first few rows for inspection if memory is an issue,
        # but here we'll load the full dataset to get complete stats.
        print("Loading dataset (this may take a moment)...")
        df = pd.read_csv(target_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin-1...")
        df = pd.read_csv(target_file, encoding='latin-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. Print stats
    print("-" * 50)
    print(f"Filename: {os.path.basename(target_file)}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate row count: {df.duplicated().sum()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("-" * 50)

    # 5. Identify brand/company column
    brand_cols = [col for col in df.columns if 'brand' in col.lower() or 'company' in col.lower() or 'account' in col.lower() or 'author' in col.lower()]
    print(f"Potential brand/company columns: {brand_cols}")

    # 6. Identify conversation/reply relationship column
    reply_cols = [col for col in df.columns if 'reply' in col.lower() or 'conversation' in col.lower() or 'in_response' in col.lower()]
    print(f"Potential reply relationship columns: {reply_cols}")

    # 7. Identify timestamp columns
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'created' in col.lower()]
    print(f"Potential timestamp columns: {time_cols}")
    print("-" * 50)

if __name__ == "__main__":
    main()
