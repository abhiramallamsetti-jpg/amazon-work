import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Same keyword rules as create_golden_set.py — kept consistent
INTENT_RULES = {
    'billing_issue':      ['charge', 'charged', 'billing', 'pay', 'payment', 'credit card', 'refund', 'money'],
    'subscription_issue': ['premium', 'family', 'student', 'subscription', 'plan', 'upgrade', 'downgrade', 'cancel'],
    'account_access':     ['login', 'log in', 'password', 'hacked', 'email', 'account', 'sign in'],
    'playback_issue':     ['play', 'playing', 'stop', 'pause', 'skip', 'skipping', 'offline', 'download', 'buffer', 'quality'],
    'app_bug':            ['crash', 'crashing', 'bug', 'update', 'glitch', "won't open", 'blank', 'error'],
    'feature_inquiry':    ['lyrics', 'podcast', 'playlist', 'add', 'feature', 'when', 'missing'],
}

def assign_intent(text):
    if pd.isna(text):
        return 'other'
    text_lower = text.lower()
    for intent, keywords in INTENT_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return 'other'


def create_splits_and_check_leakage():
    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, '..', 'data', 'processed')
    golden_dir    = os.path.join(base_dir, '..', 'data', 'golden')
    splits_dir    = os.path.join(base_dir, '..', 'data', 'splits')

    # Load full pool
    df = pd.read_csv(os.path.join(processed_dir, 'support_cases.csv'))
    df['intent'] = df['customer_message'].apply(assign_intent)

    # Load golden set (canonical name)
    golden_df = pd.read_csv(os.path.join(golden_dir, 'golden_set.csv'))

    # ------------------------------------------------------------------
    # STEP 1 — Remove golden examples from training pool by case ID
    # ------------------------------------------------------------------
    golden_case_ids = set(golden_df['source_case_id'].dropna())
    remaining_df = df[~df['case_id'].isin(golden_case_ids)].copy()

    # ------------------------------------------------------------------
    # STEP 2 — Data leakage checks
    # ------------------------------------------------------------------
    print('--- Data Leakage Report ---')

    # Check 1: exact duplicate customer messages
    golden_messages = set(golden_df['customer_message'].dropna())
    exact_dup_mask  = remaining_df['customer_message'].isin(golden_messages)
    exact_dup_count = int(exact_dup_mask.sum())
    print(f'Exact duplicate messages between pool and golden: {exact_dup_count}')

    # Check 2: conversation_id overlap
    golden_conv_ids = set(
        df.loc[df['case_id'].isin(golden_case_ids), 'conversation_id'].dropna()
    )
    convo_leak_count = int(remaining_df['conversation_id'].isin(golden_conv_ids).sum())
    print(f'Conversation leakage between pool and golden:    {convo_leak_count}')

    # Remove any exact-duplicate messages from training pool to be safe
    if exact_dup_count > 0:
        remaining_df = remaining_df[~exact_dup_mask]
        print(f'Removed {exact_dup_count} exact-duplicate messages from training pool.')

    # ------------------------------------------------------------------
    # STEP 3 — Train / validation split (stratified, seed=42)
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=0.15,
        random_state=42,
        stratify=remaining_df['intent'],
    )

    # ------------------------------------------------------------------
    # STEP 4 — Save splits
    # ------------------------------------------------------------------
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'),       index=False)
    val_df.to_csv(  os.path.join(splits_dir, 'validation.csv'),  index=False)
    # golden_test.csv is a copy of golden_set.csv kept in splits/ for consistent access
    golden_df.to_csv(os.path.join(splits_dir, 'golden_test.csv'), index=False)

    print(f'Splits created — Train: {len(train_df)}, Val: {len(val_df)}, Golden: {len(golden_df)}')


if __name__ == '__main__':
    create_splits_and_check_leakage()
