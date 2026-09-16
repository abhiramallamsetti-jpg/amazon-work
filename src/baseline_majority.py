import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_majority_baseline():
    base_dir   = os.path.dirname(__file__)
    splits_dir  = os.path.join(base_dir, '..', 'data', 'splits')
    results_dir = os.path.join(base_dir, '..', 'evaluation', 'results')

    train_df  = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    golden_df = pd.read_csv(os.path.join(splits_dir, 'golden_test.csv'))

    # The majority class is determined ONLY from the training data
    majority_class = train_df['intent'].mode()[0]

    y_true = golden_df['intent'].values
    y_pred = [majority_class] * len(y_true)

    labels = sorted(set(y_true))

    accuracy         = float(accuracy_score(y_true, y_pred))
    macro_precision  = float(precision_score(y_true, y_pred, average='macro',    zero_division=0))
    macro_recall     = float(recall_score(   y_true, y_pred, average='macro',    zero_division=0))
    macro_f1         = float(f1_score(       y_true, y_pred, average='macro',    zero_division=0))
    weighted_f1      = float(f1_score(       y_true, y_pred, average='weighted', zero_division=0))

    per_f1  = f1_score(       y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_p   = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_r   = recall_score(   y_true, y_pred, average=None, labels=labels, zero_division=0)

    results = {
        'model':            'Majority Baseline',
        'majority_class':   majority_class,
        'accuracy':         accuracy,
        'macro_precision':  macro_precision,
        'macro_recall':     macro_recall,
        'macro_f1':         macro_f1,
        'weighted_f1':      weighted_f1,
        'per_intent_f1':        dict(zip(labels, per_f1.tolist())),
        'per_intent_precision': dict(zip(labels, per_p.tolist())),
        'per_intent_recall':    dict(zip(labels, per_r.tolist())),
    }

    with open(os.path.join(results_dir, 'majority_baseline.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print('Majority Baseline Evaluation Complete.')
    print(f"  Predicted class : {majority_class}")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Macro Precision : {macro_precision:.4f}")
    print(f"  Macro Recall    : {macro_recall:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")


if __name__ == '__main__':
    evaluate_majority_baseline()
