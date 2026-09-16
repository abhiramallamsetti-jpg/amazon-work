import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def evaluate_tfidf_baseline():
    base_dir    = os.path.dirname(__file__)
    splits_dir  = os.path.join(base_dir, '..', 'data', 'splits')
    results_dir = os.path.join(base_dir, '..', 'evaluation', 'results')

    train_df  = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    golden_df = pd.read_csv(os.path.join(splits_dir, 'golden_test.csv'))

    # Fill NaN text with empty string so TF-IDF doesn't crash
    train_df['customer_message']  = train_df['customer_message'].fillna('')
    golden_df['customer_message'] = golden_df['customer_message'].fillna('')

    X_train = train_df['customer_message']
    y_train = train_df['intent']
    X_test  = golden_df['customer_message']
    y_test  = golden_df['intent']

    # ----------------------------------------------------------------
    # Pipeline: TF-IDF → Logistic Regression
    # TF-IDF is fitted ONLY on training data (inside pipeline.fit)
    # ----------------------------------------------------------------
    pipeline = Pipeline([
        ('tfidf',      TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred        = pipeline.predict(X_test)
    y_prob        = pipeline.predict_proba(X_test)
    confidences   = np.max(y_prob, axis=1)

    labels = sorted(y_test.unique())

    # Full metric suite -----------------------------------------------
    accuracy        = float(accuracy_score(y_test, y_pred))
    macro_precision = float(precision_score(y_test, y_pred, average='macro',    zero_division=0))
    macro_recall    = float(recall_score(   y_test, y_pred, average='macro',    zero_division=0))
    macro_f1        = float(f1_score(       y_test, y_pred, average='macro',    zero_division=0))
    weighted_f1     = float(f1_score(       y_test, y_pred, average='weighted', zero_division=0))

    per_f1 = f1_score(       y_test, y_pred, average=None, labels=labels, zero_division=0)
    per_p  = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    per_r  = recall_score(   y_test, y_pred, average=None, labels=labels, zero_division=0)

    results = {
        'model':                    'TF-IDF + Logistic Regression',
        'accuracy':                 accuracy,
        'macro_precision':          macro_precision,
        'macro_recall':             macro_recall,
        'macro_f1':                 macro_f1,
        'weighted_f1':              weighted_f1,
        'per_intent_f1':            dict(zip(labels, per_f1.tolist())),
        'per_intent_precision':     dict(zip(labels, per_p.tolist())),
        'per_intent_recall':        dict(zip(labels, per_r.tolist())),
    }

    with open(os.path.join(results_dir, 'tfidf_baseline.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # ----------------------------------------------------------------
    # Predictions file
    # ----------------------------------------------------------------
    predictions_df = pd.DataFrame({
        'example_id':       golden_df['example_id'].values,
        'customer_message': golden_df['customer_message'].values,
        'true_intent':      y_test.values,
        'predicted_intent': y_pred,
        'correct':          (y_test.values == y_pred),
        'confidence':       confidences,
    })
    predictions_df.to_csv(os.path.join(results_dir, 'tfidf_predictions.csv'), index=False)

    # Verify count matches golden set size
    assert len(predictions_df) == len(golden_df), (
        f"Prediction count {len(predictions_df)} != golden set size {len(golden_df)}"
    )

    # ----------------------------------------------------------------
    # Confusion matrix — CSV + PNG
    # ----------------------------------------------------------------
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'true_intent'
    cm_df.to_csv(os.path.join(results_dir, 'tfidf_confusion_matrix.csv'))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix — TF-IDF + Logistic Regression', fontsize=14)
    ax.set_ylabel('True Intent',      fontsize=12)
    ax.set_xlabel('Predicted Intent', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tfidf_confusion_matrix.png'), dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Error analysis — only incorrect predictions, sorted by confidence
    # ----------------------------------------------------------------
    errors = predictions_df[predictions_df['correct'] == False].copy()
    # Add source_case_id from golden set
    golden_id_map = golden_df.set_index('example_id')['source_case_id']
    errors['source_case_id'] = errors['example_id'].map(golden_id_map)
    errors = errors.sort_values(by='confidence', ascending=False)
    errors.to_csv(os.path.join(results_dir, 'tfidf_errors.csv'), index=False)

    # Print most common confusion pairs
    print('\nTop confusion pairs (true -> predicted):')
    err_pairs = errors.groupby(['true_intent', 'predicted_intent']).size().sort_values(ascending=False)
    print(err_pairs.head(5).to_string())

    # ----------------------------------------------------------------
    # Confidence analysis — binned correct/incorrect counts
    # ----------------------------------------------------------------
    bins      = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]   # 1.01 so 1.0 is included
    bin_labels = ['0.0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    predictions_df['conf_bin'] = pd.cut(
        predictions_df['confidence'], bins=bins, labels=bin_labels, include_lowest=True
    )
    conf_analysis = (
        predictions_df.groupby(['conf_bin', 'correct'], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    # Rename columns for clarity regardless of True/False ordering
    rename_map = {True: 'correct_predictions', False: 'incorrect_predictions'}
    conf_analysis = conf_analysis.rename(columns={k: v for k, v in rename_map.items() if k in conf_analysis.columns})
    conf_analysis.to_csv(os.path.join(results_dir, 'confidence_analysis.csv'))

    print('\nTF-IDF Baseline Evaluation Complete.')
    print(f'  Accuracy        : {accuracy:.4f}')
    print(f'  Macro Precision : {macro_precision:.4f}')
    print(f'  Macro Recall    : {macro_recall:.4f}')
    print(f'  Macro F1        : {macro_f1:.4f}')
    print(f'  Weighted F1     : {weighted_f1:.4f}')
    print(f'  Errors          : {len(errors)} / {len(golden_df)}')


if __name__ == '__main__':
    evaluate_tfidf_baseline()
