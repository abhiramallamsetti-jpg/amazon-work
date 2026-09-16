"""
tests/test_baselines.py

Standard unittest tests for Day 3 baselines.
Run with:
    python -m unittest discover -s tests -p "test_*.py" -v
or:
    python tests/test_baselines.py
"""

import os
import json
import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Shared paths — computed relative to this file so tests work from any cwd
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, '..')
SPLITS_DIR   = os.path.join(_ROOT, 'data', 'splits')
GOLDEN_DIR   = os.path.join(_ROOT, 'data', 'golden')
RESULTS_DIR  = os.path.join(_ROOT, 'evaluation', 'results')

VALID_INTENTS = {
    'billing_issue', 'subscription_issue', 'account_access',
    'playback_issue', 'app_bug', 'feature_inquiry', 'other',
}


class TestDataExists(unittest.TestCase):
    """Verify required files exist before running anything."""

    def test_train_csv_exists(self):
        path = os.path.join(SPLITS_DIR, 'train.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_validation_csv_exists(self):
        path = os.path.join(SPLITS_DIR, 'validation.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_golden_test_csv_exists(self):
        path = os.path.join(SPLITS_DIR, 'golden_test.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_golden_set_csv_exists(self):
        path = os.path.join(GOLDEN_DIR, 'golden_set.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_majority_baseline_json_exists(self):
        path = os.path.join(RESULTS_DIR, 'majority_baseline.json')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_tfidf_baseline_json_exists(self):
        path = os.path.join(RESULTS_DIR, 'tfidf_baseline.json')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_tfidf_predictions_csv_exists(self):
        path = os.path.join(RESULTS_DIR, 'tfidf_predictions.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_confusion_matrix_png_exists(self):
        path = os.path.join(RESULTS_DIR, 'tfidf_confusion_matrix.png')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')

    def test_baseline_comparison_csv_exists(self):
        path = os.path.join(RESULTS_DIR, 'baseline_comparison.csv')
        self.assertTrue(os.path.exists(path), f'Missing: {path}')


class TestDataNonEmpty(unittest.TestCase):
    """Verify splits are non-empty and have the required columns."""

    def setUp(self):
        self.train  = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
        self.val    = pd.read_csv(os.path.join(SPLITS_DIR, 'validation.csv'))
        self.golden = pd.read_csv(os.path.join(SPLITS_DIR, 'golden_test.csv'))

    def test_train_not_empty(self):
        self.assertGreater(len(self.train), 0)

    def test_validation_not_empty(self):
        self.assertGreater(len(self.val), 0)

    def test_golden_not_empty(self):
        self.assertGreater(len(self.golden), 0)

    def test_golden_in_allowed_range(self):
        n = len(self.golden)
        self.assertGreaterEqual(n, 150, f'Golden set too small: {n}')
        self.assertLessEqual(  n, 250, f'Golden set too large: {n}')

    def test_train_has_intent_column(self):
        self.assertIn('intent', self.train.columns)

    def test_golden_has_intent_column(self):
        self.assertIn('intent', self.golden.columns)

    def test_golden_has_customer_message_column(self):
        self.assertIn('customer_message', self.golden.columns)


class TestIntentLabels(unittest.TestCase):
    """Verify intent labels are valid and consistent."""

    def setUp(self):
        self.train  = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
        self.golden = pd.read_csv(os.path.join(SPLITS_DIR, 'golden_test.csv'))

    def test_golden_intent_labels_valid(self):
        invalid = set(self.golden['intent'].unique()) - VALID_INTENTS
        self.assertEqual(invalid, set(), f'Invalid intents in golden: {invalid}')

    def test_train_intent_labels_valid(self):
        invalid = set(self.train['intent'].unique()) - VALID_INTENTS
        self.assertEqual(invalid, set(), f'Invalid intents in train: {invalid}')

    def test_no_null_intents_in_golden(self):
        null_count = self.golden['intent'].isna().sum()
        self.assertEqual(null_count, 0, f'{null_count} null intents in golden set')

    def test_no_duplicate_messages_in_golden(self):
        dup_count = self.golden['customer_message'].dropna().duplicated().sum()
        self.assertEqual(dup_count, 0, f'{dup_count} duplicate messages in golden set')

    def test_no_leakage_train_vs_golden(self):
        golden_msgs = set(self.golden['customer_message'].dropna())
        leaked = self.train['customer_message'].isin(golden_msgs).sum()
        self.assertEqual(leaked, 0, f'{leaked} training messages appear in golden set (leakage!)')


class TestMajorityBaseline(unittest.TestCase):
    """Verify the majority baseline results file is correct."""

    def setUp(self):
        path = os.path.join(RESULTS_DIR, 'majority_baseline.json')
        with open(path) as f:
            self.results = json.load(f)
        self.golden = pd.read_csv(os.path.join(SPLITS_DIR, 'golden_test.csv'))

    def test_accuracy_present(self):
        self.assertIn('accuracy', self.results)

    def test_macro_f1_present(self):
        self.assertIn('macro_f1', self.results)

    def test_weighted_f1_present(self):
        self.assertIn('weighted_f1', self.results)

    def test_accuracy_range(self):
        acc = self.results['accuracy']
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(   acc, 1.0)

    def test_majority_class_is_most_common_in_train(self):
        train = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
        expected_majority = train['intent'].mode()[0]
        self.assertEqual(self.results['majority_class'], expected_majority)


class TestTfidfBaseline(unittest.TestCase):
    """Verify the TF-IDF baseline: training, prediction, and metrics."""

    def setUp(self):
        self.train  = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
        self.golden = pd.read_csv(os.path.join(SPLITS_DIR, 'golden_test.csv'))
        path = os.path.join(RESULTS_DIR, 'tfidf_baseline.json')
        with open(path) as f:
            self.results = json.load(f)

    def test_model_can_fit_and_predict(self):
        """Actually trains the pipeline from scratch and checks predictions."""
        X_train = self.train['customer_message'].fillna('')
        y_train = self.train['intent']
        X_test  = self.golden['customer_message'].fillna('')

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf',   LogisticRegression(random_state=42, max_iter=1000)),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        self.assertEqual(len(preds), len(self.golden))

    def test_predictions_count_matches_golden(self):
        pred_df = pd.read_csv(os.path.join(RESULTS_DIR, 'tfidf_predictions.csv'))
        self.assertEqual(len(pred_df), len(self.golden))

    def test_accuracy_present_and_valid(self):
        acc = self.results['accuracy']
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(   acc, 1.0)

    def test_macro_precision_present(self):
        self.assertIn('macro_precision', self.results)

    def test_macro_recall_present(self):
        self.assertIn('macro_recall', self.results)

    def test_macro_f1_present(self):
        self.assertIn('macro_f1', self.results)

    def test_weighted_f1_present(self):
        self.assertIn('weighted_f1', self.results)

    def test_per_intent_f1_present(self):
        self.assertIn('per_intent_f1', self.results)

    def test_per_intent_precision_present(self):
        self.assertIn('per_intent_precision', self.results)

    def test_per_intent_recall_present(self):
        self.assertIn('per_intent_recall', self.results)

    def test_tfidf_beats_majority_on_accuracy(self):
        with open(os.path.join(RESULTS_DIR, 'majority_baseline.json')) as f:
            maj = json.load(f)
        self.assertGreater(
            self.results['accuracy'], maj['accuracy'],
            'TF-IDF should outperform the majority baseline on accuracy.',
        )

    def test_metrics_reproducible(self):
        """Re-runs evaluation and checks accuracy matches saved JSON."""
        X_train = self.train['customer_message'].fillna('')
        y_train = self.train['intent']
        X_test  = self.golden['customer_message'].fillna('')
        y_true  = self.golden['intent']

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf',   LogisticRegression(random_state=42, max_iter=1000)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        recalc_acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(
            recalc_acc, self.results['accuracy'], places=6,
            msg='Accuracy is not reproducible — check random seed.',
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
