import os
import json
import subprocess
import sys
import pandas as pd


def evaluate_all():
    base_dir    = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, '..', 'evaluation', 'results')

    # Run Majority baseline
    print('=' * 60)
    print('Running Majority Baseline...')
    subprocess.run(
        [sys.executable, os.path.join(base_dir, 'baseline_majority.py')],
        check=True,
    )

    # Run TF-IDF baseline
    print()
    print('=' * 60)
    print('Running TF-IDF + Logistic Regression Baseline...')
    subprocess.run(
        [sys.executable, os.path.join(base_dir, 'baseline_tfidf.py')],
        check=True,
    )

    # Load results
    with open(os.path.join(results_dir, 'majority_baseline.json')) as f:
        maj = json.load(f)
    with open(os.path.join(results_dir, 'tfidf_baseline.json')) as f:
        tfidf = json.load(f)

    # Build comparison table
    comparison = pd.DataFrame([
        {
            'model':            'Majority Baseline',
            'accuracy':         maj['accuracy'],
            'macro_precision':  maj['macro_precision'],
            'macro_recall':     maj['macro_recall'],
            'macro_f1':         maj['macro_f1'],
            'weighted_f1':      maj['weighted_f1'],
        },
        {
            'model':            'TF-IDF + Logistic Regression',
            'accuracy':         tfidf['accuracy'],
            'macro_precision':  tfidf['macro_precision'],
            'macro_recall':     tfidf['macro_recall'],
            'macro_f1':         tfidf['macro_f1'],
            'weighted_f1':      tfidf['weighted_f1'],
        },
    ])

    comp_path = os.path.join(results_dir, 'baseline_comparison.csv')
    comparison.to_csv(comp_path, index=False)

    print()
    print('=' * 60)
    print('BASELINE COMPARISON')
    print('=' * 60)
    print(comparison.to_string(index=False))
    print(f'\nSaved to {comp_path}')


if __name__ == '__main__':
    evaluate_all()
