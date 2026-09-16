import pandas as pd
import json
import os

# Check golden set
g = pd.read_csv('data/golden/golden_evaluation_set.csv')
print('=== GOLDEN SET ===')
print(f'Rows: {len(g)}')
print(f'Cols: {list(g.columns)}')
print('Intents:', sorted(g['intent'].unique()))
print(g['intent'].value_counts())
print(f'Duplicate messages: {g["customer_message"].duplicated().sum()}')
print(f'Empty messages: {g["customer_message"].isna().sum()}')

# Check note: file is named golden_evaluation_set.csv, NOT golden_set.csv
golden_set_path = 'data/golden/golden_set.csv'
print(f'\ngolden_set.csv exists: {os.path.exists(golden_set_path)}')

# Check splits
print('\n=== SPLITS ===')
for split in ['train', 'validation', 'golden_test']:
    path = f'data/splits/{split}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        has_intent = 'intent' in df.columns
        print(f'{split}: {len(df)} rows, has intent: {has_intent}')
    else:
        print(f'{split}: FILE MISSING')

# Check intent_definitions
print('\n=== INTENT DEFINITIONS ===')
defs = pd.read_csv('data/intent_definitions.csv')
print(list(defs['intent']))

# Cross-check golden intents vs definitions
print('\n=== INTENT MISMATCH CHECK ===')
golden_intents = set(g['intent'].unique())
defined_intents = set(defs['intent'].unique())
print(f'Golden intents: {golden_intents}')
print(f'Defined intents: {defined_intents}')
print(f'In golden but not defined: {golden_intents - defined_intents}')
print(f'In defined but not in golden: {defined_intents - golden_intents}')
