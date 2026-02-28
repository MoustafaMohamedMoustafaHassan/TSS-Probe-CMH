#!/usr/bin/env python3
"""
================================================================================
TSS: Data Preparation & Feature Engineering Pipeline
================================================================================
A unified pipeline for transforming raw multi-source text data (Reddit, Twitter)
into a rigorous, length-normalized feature set for cross-platform stress detection.

--------------------------------------------------------------------------------
1. CORE METHODOLOGY: "REALITY-PRESERVING PREPARATION"
--------------------------------------------------------------------------------
This script adheres to a strict "Methodological Integrity" philosophy:
   - NO DEFAULT UNDERSAMPLING: We intentionally preserve the natural class imbalance
     (e.g., Reddit Combi: 88% Non-Stressed vs 12% Stressed).
   - RATIONALE: Artificially balancing data via deletion distorts the "epidemiological
     reality" of stress prevalence.
   - ADAPTATION: We shift the burden of balance from data deletion to algorithmic
     correction (using Class Weights during training).

--------------------------------------------------------------------------------
2. TECHNICAL PIPELINE STEPS
--------------------------------------------------------------------------------
This script executes the following automated workflow:

A. SMART INGESTION (`smart_read_csv`):
   - Automatically detects CSV delimiters (, or ;) and file encodings (utf-8, cp1252).
   - Handles bad lines and header inconsistencies to prevent data loss.

B. SANITIZATION & ANONYMIZATION (`clean_text`):
   - Removes PII (User mentions, URLs, Emails).
   - Normalizes whitespace and artifacts.
   - [Configurable] Removes Mental Health specific keywords (e.g., "anxiety", "depressed")
     to force the model to learn stylistic patterns rather than relying on keywords.

C. RIGOROUS QUALITY CONTROL:
   - Label Validation: Drops rows with NaN labels; coerces labels to binary {0,1}.
   - Intra-Dataset Deduplication: Removes duplicate texts within a single dataset.
   - Cross-Dataset Decontamination (`remove_cross_duplicates`):
     Ensures NO data leakage between Training sets and Test/Gold sets.
     (e.g., if a Tweet exists in Train and Gold, it is removed from Train).

D. ADVANCED FEATURE ENGINEERING (`extract_features`):
   Extracts a rich set of psycholinguistic features, strictly LENGTH-NORMALIZED:
   1. Morphological: POS ratios (Verbs, Pronouns, Adjectives per token).
   2. Stylistic: Caps ratio, Punctuation density (?!), Elongation (e.g., "soooo").
   3. Affective:
      - NRC-VAD (Valence, Arousal, Dominance).
      - VADER Sentiment (Compound, Pos, Neg).
      - Emotion Density (Fear, Anger, Sadness, Joy per token).
   4. Complexity: Readability scores (Flesch-Kincaid, Gunning Fog) & Type-Token Ratio.
   5. Cognitive: Absolutist words, First-person pronouns, Stress-verbs.

--------------------------------------------------------------------------------
3. DATASET HANDLING
--------------------------------------------------------------------------------
- Dreaddit (Train/Test): Processed as the primary domain anchor.
- Twitter (Full): Auto-labeled dataset.
- Twitter Gold: HUMAN-LABELED validation set (Critical for Ground Truth evaluation).
- Reddit Combi: Large-scale dataset (Optional balancing via --balance-reddit-combi).

USAGE:
    python scripts/01_prepare_data.py --all
    python scripts/01_prepare_data.py --dataset twitter_gold
    python scripts/01_prepare_data.py --all --balance-reddit-combi  # FORCE balance
================================================================================
"""

import argparse
import re
import sys
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tss.config import (
    ALLOWED_POS_TAGS,
    NEGATION_TOKENS,
    ABSOLUTIST_TERMS,
    FEAR_WORDS,
    ANGER_WORDS,
    SADNESS_WORDS,
    JOY_WORDS,
    ANTICIPATION_WORDS,
    STRESS_VERBS,
    FIRST_PERSON,
    DOMAIN_MAP,
    MENTAL_HEALTH_KEYWORDS,
)

from tss.open_lexicons import (
    load_open_lexicons,
    extract_open_lexicon_features,
    get_all_feature_names,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_STATE = 42  # For reproducibility

EMOTION_WORDS = {
    'fear': FEAR_WORDS,
    'anger': ANGER_WORDS,
    'sadness': SADNESS_WORDS,
    'joy': JOY_WORDS,
    'anticipation': ANTICIPATION_WORDS,
}

NEGATION_WORDS = NEGATION_TOKENS

# Valid label column names - NOTE: 'sentiment' REMOVED as it's not stress!
VALID_LABEL_COLUMNS = ['label', 'labels', 'is_stressed', 'stress', 'stressed']


# =============================================================================
# [NEW ] BALANCED SAMPLING FUNCTION
# =============================================================================

def balance_dataset(
    df: pd.DataFrame,
    label_col: str = 'label',
    target_ratio: float = 0.5,
    seed: int = RANDOM_STATE,
    dataset_name: str = "dataset"
) -> pd.DataFrame:
    """
    Balance dataset by downsampling majority class.

    ⚠️ CRITICAL for reddit_combi which has 88% positive class!

    Without balancing:
    - Majority baseline F1 = 0.936
    - Model improvement is nearly invisible (+0.003)

    With balancing (50/50):
    - Majority baseline F1 = 0.667
    - Model improvement becomes meaningful

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with label column
    label_col : str
        Name of label column (default: 'label')
    target_ratio : float
        Target positive class ratio (default: 0.5 for balanced)
    seed : int
        Random seed for reproducibility
    dataset_name : str
        Name for logging

    Returns:
    --------
    pd.DataFrame
        Balanced dataset with ~50% positive class
    """
    if label_col not in df.columns:
        logger.warning(f"[{dataset_name}] No '{label_col}' column found, skipping balance")
        return df

    n_positive = (df[label_col] == 1).sum()
    n_negative = (df[label_col] == 0).sum()
    n_total = len(df)

    if n_total == 0:
        return df

    current_ratio = n_positive / n_total

    logger.info(f"[{dataset_name}] BEFORE balancing:")
    logger.info(f"   Total samples: {n_total:,}")
    logger.info(f"   Positive (stress=1): {n_positive:,} ({current_ratio*100:.1f}%)")
    logger.info(f"   Negative (stress=0): {n_negative:,} ({(1-current_ratio)*100:.1f}%)")

    # Compute majority baseline F1 before balancing
    if n_positive >= n_negative:
        # Predict all positive: TP=n_pos, FP=n_neg, FN=0
        precision_before = n_positive / n_total
        recall_before = 1.0
        baseline_f1_before = 2 * precision_before * recall_before / (precision_before + recall_before)
    else:
        baseline_f1_before = 0.0
    logger.info(f"   Majority baseline F1: {baseline_f1_before:.3f}")

    # If already balanced (within 10%), return as-is
    if abs(current_ratio - target_ratio) < 0.1:
        logger.info(f"[{dataset_name}] Already balanced, no changes needed")
        return df

    # Set random seed
    np.random.seed(seed)

    positive_df = df[df[label_col] == 1].copy()
    negative_df = df[df[label_col] == 0].copy()

    # Downsample majority class
    if n_positive > n_negative:
        # Downsample positives to match negatives
        target_n = n_negative
        sampled_positive = positive_df.sample(n=target_n, random_state=seed)
        balanced_df = pd.concat([sampled_positive, negative_df], ignore_index=True)
        logger.info(f"[{dataset_name}] → Downsampled POSITIVE class: {n_positive:,} → {target_n:,}")
    else:
        # Downsample negatives to match positives
        target_n = n_positive
        sampled_negative = negative_df.sample(n=target_n, random_state=seed)
        balanced_df = pd.concat([positive_df, sampled_negative], ignore_index=True)
        logger.info(f"[{dataset_name}] → Downsampled NEGATIVE class: {n_negative:,} → {target_n:,}")

    # Shuffle to avoid ordering bias
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Log results
    new_n_positive = (balanced_df[label_col] == 1).sum()
    new_n_negative = (balanced_df[label_col] == 0).sum()
    new_ratio = new_n_positive / len(balanced_df) if len(balanced_df) > 0 else 0.5

    # Compute new majority baseline F1
    baseline_f1_after = 2 * 0.5 * 1.0 / (0.5 + 1.0)  # = 0.667 for balanced

    logger.info(f"[{dataset_name}] AFTER balancing:")
    logger.info(f"   Total samples: {len(balanced_df):,}")
    logger.info(f"   Positive (stress=1): {new_n_positive:,} ({new_ratio*100:.1f}%)")
    logger.info(f"   Negative (stress=0): {new_n_negative:,} ({(1-new_ratio)*100:.1f}%)")
    logger.info(f"   Majority baseline F1: {baseline_f1_after:.3f}")

    return balanced_df


# =============================================================================
# NRC-VAD LEXICON
# =============================================================================

_VAD_CACHE: Optional[Dict[str, tuple]] = None

def load_nrc_vad(path: Optional[str] = None) -> Dict[str, tuple]:
    """Load NRC-VAD lexicon."""
    global _VAD_CACHE
    if _VAD_CACHE is not None:
        return _VAD_CACHE

    if path is None:
        path = Path(__file__).parent.parent / "data" / "lexicons" / "NRC-VAD.txt"

    vad = {}
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    term = parts[0].strip().lower()
                    try:
                        v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                        vad[term] = (v, a, d)
                    except ValueError:
                        continue

    _VAD_CACHE = vad
    return vad

# =============================================================================
# VADER SENTIMENT
# =============================================================================

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    _VADER = None
    logger.warning("VADER not available. Using fallback sentiment scores.")

def get_vader_scores(text: str) -> Dict[str, float]:
    """Get VADER sentiment scores."""
    if HAS_VADER and _VADER is not None:
        scores = _VADER.polarity_scores(text)
        return {
            'vader_pos': scores['pos'],
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_compound': scores['compound'],
        }
    else:
        return {
            'vader_pos': 0.0,
            'vader_neg': 0.0,
            'vader_neu': 1.0,
            'vader_compound': 0.0,
        }

# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str, remove_mh_keywords: bool = False) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove markdown links
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    # Remove subreddit/user mentions
    text = re.sub(r'/?r/\w+', ' ', text)
    text = re.sub(r'/?u/\w+', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\w+', ' ', text)

    if remove_mh_keywords:
        pattern = r'\b(' + '|'.join(MENTAL_HEALTH_KEYWORDS) + r')\b'
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =============================================================================
# ROBUST LABEL PARSING
# =============================================================================

def parse_label_column(
    df: pd.DataFrame,
    dataset_name: str
) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    """
    Robustly parse label column with validation.

    Returns:
        (df_with_valid_labels, label_column_name, stats_dict)

    Note: NaN labels are DROPPED (not filled), so the returned df may be shorter.
    """
    # Find label column
    label_col = None
    for col in VALID_LABEL_COLUMNS:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        available = list(df.columns)
        logger.error(f"[{dataset_name}] No valid label column found! Available: {available}")
        raise ValueError(
            f"[{dataset_name}] No valid label column found in columns: {available}. "
            f"Expected one of: {VALID_LABEL_COLUMNS}"
        )

    # Parse labels safely
    raw_labels = df[label_col].copy()

    # Convert to numeric, coercing errors
    labels = pd.to_numeric(raw_labels, errors='coerce')

    # Count issues
    nan_count = labels.isna().sum()
    if nan_count > 0:
        logger.warning(f"[{dataset_name}] {nan_count} NaN labels found in '{label_col}' — DROPPING these rows")

    # DROP NaN labels (do NOT fill with 0 — that biases class distribution)
    valid_mask = labels.notna()
    labels = labels[valid_mask]
    df = df[valid_mask].reset_index(drop=True)

    # Convert to int
    labels = labels.astype(int)

    # Validate binary (0/1)
    unique_vals = set(labels.unique())
    if not unique_vals.issubset({0, 1}):
        logger.warning(f"[{dataset_name}] Non-binary labels found: {unique_vals}")
        # Binarize: anything > 0 is 1
        labels = (labels > 0).astype(int)

    # Distribution stats
    dist = labels.value_counts().to_dict()
    stats = {
        'column': label_col,
        'nan_count': nan_count,
        'distribution': dist,
    }

    logger.info(f"[{dataset_name}] Label column: '{label_col}' | Distribution: {dist}")

    # Assign cleaned labels back
    df = df.copy()
    df['label'] = labels.values

    return df, label_col, stats

# =============================================================================
# SMART CSV READING WITH LOGGING
# =============================================================================

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

# [Arabic comment removed - see English translation above]
logger = logging.getLogger(__name__)

def smart_read_csv(path: Path, dataset_name: str = "unknown") -> Tuple[pd.DataFrame, int]:
    """
    Read CSV with smart delimiter detection, bad line logging, AND auto-encoding detection.
    """

    # 1. Helper function to try multiple encodings
    def read_file_with_encodings(filepath, delim):
        encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                df_temp = pd.read_csv(
                    filepath,
                    delimiter=delim,
                    on_bad_lines='skip',
                    encoding=encoding
                )
                return df_temp, encoding
            except UnicodeDecodeError:
                continue # If this encoding fails, try the next one
            except Exception as e:
                raise e # If it's another error (not encoding), raise it to be handled by outer logic

        # If all encodings fail
        raise ValueError(f"Failed to decode file with encodings: {encodings_to_try}")

    # ---------------------------------------------------------

    # Read first line (header) to detect delimiter more accurately
    # Using errors='replace' here ensures we don't crash just checking the header
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline()

    # Count delimiters in header only
    header_semicolons = header_line.count(';')
    header_commas = header_line.count(',')

    # Use header-based detection
    if header_semicolons >= 3:
        delimiter = ';'
    elif header_commas >= 3:
        delimiter = ','
    else:
        # Fallback to full sample analysis
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(10000)
        semicolons = sample.count(';')
        commas = sample.count(',')
        delimiter = ';' if semicolons > commas else ','

    logger.info(f"[{dataset_name}] Detected delimiter: '{delimiter}'")

    # Count total lines first (for skipping statistics)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header

    # Main Reading Logic
    try:
        df, used_encoding = read_file_with_encodings(path, delimiter)

        # Check if columns look wrong (all in one column)
        if len(df.columns) == 1 and ';' in str(df.columns[0]):
            logger.warning(f"[{dataset_name}] Single column detected, switching to semicolon...")
            df, used_encoding = read_file_with_encodings(path, delimiter=';')

        n_skipped = total_lines - len(df)
        if n_skipped > 0:
            logger.warning(f"[{dataset_name}] Skipped {n_skipped} lines (Encoding: {used_encoding})")
        else:
            logger.info(f"[{dataset_name}] Successfully loaded with encoding: {used_encoding}")

        return df, n_skipped

    except Exception as e:
        logger.error(f"[{dataset_name}] Initial read failed: {e}. Trying alternate delimiter...")

        alt_delimiter = ',' if delimiter == ';' else ';'
        try:
            df, used_encoding = read_file_with_encodings(path, alt_delimiter)
            n_skipped = total_lines - len(df)
            logger.info(f"[{dataset_name}] Recovered with delimiter '{alt_delimiter}' and encoding '{used_encoding}'")
            return df, n_skipped
        except Exception as final_e:
            logger.critical(f"[{dataset_name}] Fatal error reading CSV: {final_e}")
            return pd.DataFrame(), 0
# =============================================================================
# DUPLICATE REMOVAL
# =============================================================================

def remove_duplicates(
    df: pd.DataFrame,
    text_col: str,
    dataset_name: str
) -> pd.DataFrame:
    """Remove duplicate texts within dataset."""
    initial_len = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    removed = initial_len - len(df)

    if removed > 0:
        logger.info(f"[{dataset_name}] Removed {removed} internal duplicates")

    return df

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(text: str, nlp) -> Dict[str, Any]:
    """
    Extract all features needed for TSS channels.
    All count features are LENGTH NORMALIZED.
    """
    # Initialize with defaults
    features = {
        'cleaned_text': text,
        'token_pos_pairs': [],
        'pos_sequence': '',

        # Basic POS ratios
        'pronoun_ratio': 0.0,
        'verb_ratio': 0.0,
        'adjective_ratio': 0.0,
        'adverb_ratio': 0.0,
        'modal_ratio': 0.0,
        'negation_ratio': 0.0,
        'avg_sentence_length': 0.0,
        'fragmentation_index': 0.0,

        # LENGTH NORMALIZED features
        'q_mark_per_token': 0.0,
        'exc_mark_per_token': 0.0,
        'caps_word_ratio': 0.0,
        'elongated_per_token': 0.0,
        'absolutist_per_token': 0.0,
        'negation_per_token': 0.0,

        # Emotion ratios
        'fear_per_token': 0.0,
        'anger_per_token': 0.0,
        'sadness_per_token': 0.0,
        'joy_per_token': 0.0,

        # VAD features
        'avg_valence': 0.5,
        'avg_arousal': 0.5,

        # VADER
        'vader_compound': 0.0,

        # Readability
        'flesch_kincaid_grade': 0.0,
        'gunning_fog_index': 0.0,

        # Lexical diversity
        'type_token_ratio': 0.0,
    }

    if not text or not text.strip():
        return features

    # Process with spaCy
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return features

    # Token/POS pairs for Channel B
    token_pos_pairs = []
    for token in tokens:
        if token.tag_ in ALLOWED_POS_TAGS:  # FIXED: Use tag_ (Penn Treebank) not pos_ (Universal)
            token_pos_pairs.append((token.text, token.tag_, token.lemma_))  # FIXED

    features['token_pos_pairs'] = str(token_pos_pairs)
    features['pos_sequence'] = ' '.join([t.pos_ for t in tokens])

    # POS counts
    pos_counts = Counter([token.pos_ for token in tokens])

    features['pronoun_ratio'] = pos_counts.get('PRON', 0) / n_tokens
    features['verb_ratio'] = pos_counts.get('VERB', 0) / n_tokens
    features['adjective_ratio'] = pos_counts.get('ADJ', 0) / n_tokens
    features['adverb_ratio'] = pos_counts.get('ADV', 0) / n_tokens

    # Modal verbs
    modal_count = sum(1 for t in tokens if t.tag_ == 'MD')
    features['modal_ratio'] = modal_count / n_tokens

    # Negation
    negation_count = sum(1 for t in tokens if t.text.lower() in NEGATION_WORDS)
    features['negation_ratio'] = negation_count / n_tokens
    features['negation_per_token'] = negation_count / n_tokens

    # Sentences
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    if n_sentences > 0:
        features['avg_sentence_length'] = n_tokens / n_sentences
        features['fragmentation_index'] = n_sentences / n_tokens if n_tokens > 0 else 0

    # Punctuation
    features['q_mark_per_token'] = text.count('?') / n_tokens
    features['exc_mark_per_token'] = text.count('!') / n_tokens

    # Caps words
    caps_count = sum(1 for t in tokens if t.text.isupper() and len(t.text) > 1)
    features['caps_word_ratio'] = caps_count / n_tokens

    # Elongated words (e.g., "sooooo")
    elongated_pattern = re.compile(r'(.)\1{2,}')
    elongated_count = sum(1 for t in tokens if elongated_pattern.search(t.text))
    features['elongated_per_token'] = elongated_count / n_tokens

    # Absolutist terms
    absolutist_count = sum(1 for t in tokens if t.lemma_.lower() in ABSOLUTIST_TERMS)
    features['absolutist_per_token'] = absolutist_count / n_tokens

    # Emotion words
    lemmas = [t.lemma_.lower() for t in tokens]
    for emotion, words in EMOTION_WORDS.items():
        count = sum(1 for lemma in lemmas if lemma in words)
        features[f'{emotion}_per_token'] = count / n_tokens

    # VAD features
    vad_lexicon = load_nrc_vad()
    valence_scores = []
    arousal_scores = []

    for token in tokens:
        word = token.lemma_.lower()
        if word in vad_lexicon:
            v, a, d = vad_lexicon[word]
            valence_scores.append(v)
            arousal_scores.append(a)

    if valence_scores:
        features['avg_valence'] = np.mean(valence_scores)
        features['avg_arousal'] = np.mean(arousal_scores)

    # VADER
    vader_scores = get_vader_scores(text)
    features['vader_compound'] = vader_scores['vader_compound']
    features.update(vader_scores)

    # Type-token ratio
    word_tokens = [t.text.lower() for t in tokens if t.is_alpha]
    if word_tokens:
        features['type_token_ratio'] = len(set(word_tokens)) / len(word_tokens)

    # First person pronouns
    first_person_count = sum(1 for t in tokens if t.text.lower() in FIRST_PERSON)
    features['first_person_ratio'] = first_person_count / n_tokens

    # Stress verbs
    stress_verb_count = sum(1 for t in tokens if t.lemma_.lower() in STRESS_VERBS)
    features['has_stress_verb'] = 1 if stress_verb_count > 0 else 0

    # Open lexicon features
    try:
        open_feats = extract_open_lexicon_features(text, prefix='open_cat')
        features.update(open_feats)
    except Exception:
        pass

    return features

# =============================================================================
# DATASET PROCESSING FUNCTIONS
# =============================================================================

def process_dreaddit(input_path: Path, output_dir: Path, nlp,
                     is_test: bool = False, remove_mh_keywords: bool = True) -> pd.DataFrame:
    """Process Dreaddit dataset."""
    dataset_name = "dreaddit_test" if is_test else "dreaddit_train"
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {input_path.name} (as {dataset_name})")
    logger.info('='*60)

    df, n_skipped = smart_read_csv(input_path, dataset_name)
    logger.info(f"Loaded {len(df):,} rows")

    # Identify text column
    text_col = None
    for col in ['text', 'post_text', 'body', 'content', 'selftext']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"No text column found in {input_path}")

    # Remove duplicates
    df = remove_duplicates(df, text_col, dataset_name)

    # Parse labels
    df, label_col, label_stats = parse_label_column(df, dataset_name)

    # Create post_id
    if 'post_id' not in df.columns and 'id' in df.columns:
        df['post_id'] = df['id']
    elif 'post_id' not in df.columns:
        df['post_id'] = range(len(df))

    # Get domain from subreddit
    if 'subreddit' in df.columns:
        df['domain'] = df['subreddit'].apply(
            lambda x: DOMAIN_MAP.get(str(x).lower(), 'unknown') if pd.notna(x) else 'unknown'
        )
    else:
        df['domain'] = 'unknown'
        df['subreddit'] = 'unknown'

    # Extract features
    logger.info("Extracting features...")
    all_features = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"   Processing {idx}/{len(df)}...")

        text = str(row.get(text_col, ''))
        cleaned = clean_text(text, remove_mh_keywords=remove_mh_keywords)
        features = extract_features(cleaned, nlp)
        features['label'] = row['label']
        features['post_id'] = row['post_id']
        features['domain'] = row['domain']
        features['subreddit'] = row.get('subreddit', 'unknown')
        all_features.append(features)

    result_df = pd.DataFrame(all_features)

    # Save
    output_filename = "dreaddit_test.csv" if is_test else "dreaddit_train.csv"
    output_path = output_dir / output_filename
    result_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"Total features: {len(result_df.columns)}")

    return result_df


def process_twitter(input_path: Path, output_dir: Path, nlp,
                    remove_mh_keywords: bool = True) -> pd.DataFrame:
    """Process Twitter dataset."""
    dataset_name = "twitter"
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {input_path.name}")
    logger.info('='*60)

    df, n_skipped = smart_read_csv(input_path, dataset_name)
    logger.info(f"Loaded {len(df):,} rows")

    # Identify text column
    text_col = None
    for col in ['text', 'tweet', 'content', 'body', 'Text', 'post_text']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"No text column found in {input_path}")

    # Remove duplicates
    df = remove_duplicates(df, text_col, dataset_name)

    # Parse labels
    df, label_col, label_stats = parse_label_column(df, dataset_name)

    # Create post_id
    if 'post_id' not in df.columns:
        df['post_id'] = range(len(df))

    # Extract features
    logger.info("Extracting features...")
    all_features = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"   Processing {idx}/{len(df)}...")

        text = str(row.get(text_col, ''))
        cleaned = clean_text(text, remove_mh_keywords=remove_mh_keywords)
        features = extract_features(cleaned, nlp)
        features['label'] = row['label']
        features['post_id'] = row['post_id']
        features['domain'] = 'twitter'
        features['subreddit'] = 'twitter'
        all_features.append(features)

    result_df = pd.DataFrame(all_features)

    # Save
    output_path = output_dir / "twitter_processed.csv"
    result_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"Total features: {len(result_df.columns)}")

    return result_df


# =============================================================================
# [NEW ] TWITTER_SAMPLE_GOLD - HUMAN-LABELED CROSS-PLATFORM EVALUATION
# =============================================================================

def process_twitter_gold(input_path: Path, output_dir: Path, nlp,
                         remove_mh_keywords: bool = True) -> pd.DataFrame:
    """
    Process Twitter_sample_gold dataset (HUMAN-labeled by researcher).

    This dataset is critical for the  hypothesis:
    - Human labels should align better with style/structure channels (B, C)
    - Auto labels tend to align with lexical channel (A)

    The dataset contains 2,671 manually labeled tweets for stress detection.

    Parameters:
    -----------
    input_path : Path
        Path to Twitter_sample_gold.csv
    output_dir : Path
        Output directory for processed file
    nlp : spacy model
        spaCy language model for feature extraction
    remove_mh_keywords : bool
        Whether to remove mental health keywords (default: True)

    Returns:
    --------
    pd.DataFrame
        Processed dataset with all features
    """
    dataset_name = "twitter_gold"
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {input_path.name} (HUMAN-LABELED)")
    logger.info(f"⭐ Critical dataset for cross-platform human-label evaluation")
    logger.info('='*60)

    df, n_skipped = smart_read_csv(input_path, dataset_name)
    logger.info(f"Loaded {len(df):,} rows")

    # Identify text column
    text_col = None
    for col in ['text', 'tweet', 'content', 'body', 'Text', 'post_text']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"No text column found in {input_path}")

    logger.info(f"   Text column: {text_col}")

    # Remove duplicates
    df = remove_duplicates(df, text_col, dataset_name)

    # Parse labels
    df, label_col, label_stats = parse_label_column(df, dataset_name)

    # Log label distribution (important for human-labeled data)
    n_stressed = (df['label'] == 1).sum()
    n_not_stressed = (df['label'] == 0).sum()
    logger.info(f"\n📊 HUMAN LABEL DISTRIBUTION:")
    logger.info(f"   Stressed (1):     {n_stressed:,} ({n_stressed/len(df)*100:.1f}%)")
    logger.info(f"   Not Stressed (0): {n_not_stressed:,} ({n_not_stressed/len(df)*100:.1f}%)")

    # Create post_id
    if 'post_id' not in df.columns:
        df['post_id'] = range(len(df))

    # Extract features
    logger.info("\nExtracting features...")
    all_features = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"   Processing {idx}/{len(df)}...")

        text = str(row.get(text_col, ''))
        cleaned = clean_text(text, remove_mh_keywords=remove_mh_keywords)
        features = extract_features(cleaned, nlp)
        features['label'] = row['label']
        features['post_id'] = row['post_id']
        features['domain'] = 'twitter_gold'  # Unique domain identifier
        features['subreddit'] = 'twitter_gold'
        all_features.append(features)

    result_df = pd.DataFrame(all_features)

    # Compute text length statistics (important for cross-platform analysis)
    text_lengths = result_df['cleaned_text'].str.split().str.len()
    logger.info(f"\n📏 TEXT LENGTH STATISTICS:")
    logger.info(f"   Mean:   {text_lengths.mean():.1f} tokens")
    logger.info(f"   Median: {text_lengths.median():.1f} tokens")
    logger.info(f"   Std:    {text_lengths.std():.1f} tokens")
    logger.info(f"   Min:    {text_lengths.min():.0f} tokens")
    logger.info(f"   Max:    {text_lengths.max():.0f} tokens")

    # Save
    output_path = output_dir / "twitter_gold_processed.csv"
    result_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved to {output_path}")
    logger.info(f"Total features: {len(result_df.columns)}")
    logger.info(f"Total samples:  {len(result_df):,}")

    return result_df


def process_reddit_combi(
    input_path: Path,
    output_dir: Path,
    nlp,
    remove_mh_keywords: bool = True,
    balance: bool = True,  # [NEW ] Balance flag
    seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Process Reddit Combi dataset.

    [NEW ] Added balanced sampling option to address 88% positive class.
    """
    dataset_name = "reddit_combi"
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {input_path.name}")
    logger.info('='*60)

    df, n_skipped = smart_read_csv(input_path, dataset_name)
    logger.info(f"Loaded {len(df):,} rows")

    # Identify text column
    text_col = None
    for col in ['text', 'post_text', 'body', 'Body_Title', 'content', 'selftext']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"No text column found in {input_path}")

    # Remove duplicates
    df = remove_duplicates(df, text_col, dataset_name)

    # Parse labels
    df, label_col, label_stats = parse_label_column(df, dataset_name)

    # =========================================================================
    # [NEW ] BALANCED SAMPLING
    # =========================================================================
    if balance:
        logger.info("\n" + "="*60)
        logger.info("⚖️  APPLYING BALANCED SAMPLING")
        logger.info("="*60)
        df = balance_dataset(
            df,
            label_col='label',
            target_ratio=0.5,
            seed=seed,
            dataset_name=dataset_name
        )
        logger.info("="*60 + "\n")
    else:
        logger.warning("[reddit_combi] Balanced sampling DISABLED (--no-balance-reddit-combi)")
        logger.warning("   ⚠️ Results will be affected by 88% positive class imbalance!")

    # Create post_id
    if 'post_id' not in df.columns:
        df['post_id'] = range(len(df))

    # Get domain
    if 'subreddit' in df.columns:
        df['domain'] = df['subreddit'].apply(
            lambda x: DOMAIN_MAP.get(str(x).lower(), 'reddit') if pd.notna(x) else 'reddit'
        )
    else:
        df['domain'] = 'reddit'
        df['subreddit'] = 'reddit'

    # Extract features
    logger.info("Extracting features...")
    all_features = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"   Processing {idx}/{len(df)}...")

        text = str(row.get(text_col, ''))
        cleaned = clean_text(text, remove_mh_keywords=remove_mh_keywords)
        features = extract_features(cleaned, nlp)
        features['label'] = row['label']
        features['post_id'] = row['post_id']
        features['domain'] = row['domain']
        features['subreddit'] = row.get('subreddit', 'reddit')
        all_features.append(features)

    result_df = pd.DataFrame(all_features)

    # Save
    output_path = output_dir / "reddit_combi_processed.csv"
    result_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"Total features: {len(result_df.columns)}")

    # Final summary
    final_pos = (result_df['label'] == 1).sum()
    final_neg = (result_df['label'] == 0).sum()
    logger.info(f"\n📊 FINAL reddit_combi: {len(result_df):,} samples "
                f"(positive={final_pos:,}, negative={final_neg:,}, "
                f"ratio={final_pos/len(result_df)*100:.1f}%)")

    return result_df

# =============================================================================
# CROSS-DATASET DUPLICATE REMOVAL
# =============================================================================

def remove_cross_duplicates(output_dir: Path):
    """
    Remove duplicates ACROSS all datasets.

    Checks:
    - dreaddit_train ↔ dreaddit_test
    - dreaddit_train ↔ twitter
    - dreaddit_train ↔ twitter_gold  [NEW ]
    - dreaddit_train ↔ reddit_combi
    - twitter ↔ reddit_combi
    - twitter ↔ twitter_gold  [NEW ]

    Priority: test/OOD sets are kept intact; duplicates removed from training.
    """
    files = {
        'dreaddit_train': output_dir / 'dreaddit_train.csv',
        'dreaddit_test': output_dir / 'dreaddit_test.csv',
        'twitter': output_dir / 'twitter_processed.csv',
        'twitter_gold': output_dir / 'twitter_gold_processed.csv',  # NEW 'reddit_combi': output_dir / 'reddit_combi_processed.csv',
    }

    loaded = {}
    for name, path in files.items():
        if path.exists():
            loaded[name] = pd.read_csv(path)

    if not loaded:
        return

    def get_texts(df):
        return set(df['cleaned_text'].dropna().astype(str))

    total_removed = 0

    # 1. train ↔ test (remove from train)
    if 'dreaddit_train' in loaded and 'dreaddit_test' in loaded:
        dupes = get_texts(loaded['dreaddit_train']) & get_texts(loaded['dreaddit_test'])
        if dupes:
            n = len(dupes)
            loaded['dreaddit_train'] = loaded['dreaddit_train'][
                ~loaded['dreaddit_train']['cleaned_text'].astype(str).isin(dupes)
            ].reset_index(drop=True)
            logger.warning(f"Cross-dedup: train↔test: removed {n} from train")
            total_removed += n

    # 2. train ↔ twitter (remove from train)
    if 'dreaddit_train' in loaded and 'twitter' in loaded:
        dupes = get_texts(loaded['dreaddit_train']) & get_texts(loaded['twitter'])
        if dupes:
            n = len(dupes)
            loaded['dreaddit_train'] = loaded['dreaddit_train'][
                ~loaded['dreaddit_train']['cleaned_text'].astype(str).isin(dupes)
            ].reset_index(drop=True)
            logger.warning(f"Cross-dedup: train↔twitter: removed {n} from train")
            total_removed += n

    # 2b. [NEW ] train ↔ twitter_gold (remove from train, preserve gold!)
    if 'dreaddit_train' in loaded and 'twitter_gold' in loaded:
        dupes = get_texts(loaded['dreaddit_train']) & get_texts(loaded['twitter_gold'])
        if dupes:
            n = len(dupes)
            loaded['dreaddit_train'] = loaded['dreaddit_train'][
                ~loaded['dreaddit_train']['cleaned_text'].astype(str).isin(dupes)
            ].reset_index(drop=True)
            logger.warning(f"Cross-dedup: train↔twitter_gold: removed {n} from train")
            total_removed += n

    # 3. train ↔ reddit_combi (remove from train)
    if 'dreaddit_train' in loaded and 'reddit_combi' in loaded:
        dupes = get_texts(loaded['dreaddit_train']) & get_texts(loaded['reddit_combi'])
        if dupes:
            n = len(dupes)
            loaded['dreaddit_train'] = loaded['dreaddit_train'][
                ~loaded['dreaddit_train']['cleaned_text'].astype(str).isin(dupes)
            ].reset_index(drop=True)
            logger.warning(f"Cross-dedup: train↔reddit_combi: removed {n} from train")
            total_removed += n

    # 4. twitter ↔ reddit_combi (remove from the larger one)
    if 'twitter' in loaded and 'reddit_combi' in loaded:
        dupes = get_texts(loaded['twitter']) & get_texts(loaded['reddit_combi'])
        if dupes:
            n = len(dupes)
            if len(loaded['twitter']) >= len(loaded['reddit_combi']):
                loaded['twitter'] = loaded['twitter'][
                    ~loaded['twitter']['cleaned_text'].astype(str).isin(dupes)
                ].reset_index(drop=True)
                logger.warning(f"Cross-dedup: twitter↔reddit_combi: removed {n} from twitter")
            else:
                loaded['reddit_combi'] = loaded['reddit_combi'][
                    ~loaded['reddit_combi']['cleaned_text'].astype(str).isin(dupes)
                ].reset_index(drop=True)
                logger.warning(f"Cross-dedup: twitter↔reddit_combi: removed {n} from reddit_combi")
            total_removed += n

    # 5. [NEW ] twitter ↔ twitter_gold (remove from auto-labeled twitter)
    if 'twitter' in loaded and 'twitter_gold' in loaded:
        dupes = get_texts(loaded['twitter']) & get_texts(loaded['twitter_gold'])
        if dupes:
            n = len(dupes)
            # Always preserve gold labels - remove from auto-labeled
            loaded['twitter'] = loaded['twitter'][
                ~loaded['twitter']['cleaned_text'].astype(str).isin(dupes)
            ].reset_index(drop=True)
            logger.warning(f"Cross-dedup: twitter↔twitter_gold: removed {n} from twitter (preserving gold)")
            total_removed += n

    # Save updated files
    for name, df in loaded.items():
        df.to_csv(files[name], index=False)

    if total_removed > 0:
        logger.info(f"Total cross-dataset duplicates removed: {total_removed}")
    else:
        logger.info("No cross-dataset duplicates found")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TSS Data Preparation")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'dreaddit', 'dreaddit_test', 'twitter', 'twitter_gold', 'reddit_combi'],
                        help='Dataset to process')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--keep-mh-keywords', action='store_true',
                        help='Keep mental health keywords (default: remove)')

    # [UPDATED ] Balance control for reddit_combi
    # PHILOSOPHY: "Let the data express its nature; let the algorithm adapt to scarcity."
    # DEFAULT: False - We use class_weight='balanced' in the model instead of undersampling
    parser.add_argument('--balance-reddit-combi', action='store_true', default=False,
                        help='Balance reddit_combi by downsampling (DEFAULT: False - use class_weight instead)')
    parser.add_argument('--no-balance-reddit-combi', dest='balance_reddit_combi', action='store_false',
                        help='Disable balancing for reddit_combi (DEFAULT behavior - keeps full data)')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help=f'Random seed for reproducibility (default: {RANDOM_STATE})')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    remove_mh = not args.keep_mh_keywords

    # Load spaCy
    logger.info("Loading spaCy model...")
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        logger.info(f"✓ spaCy version: {spacy.__version__}")
        logger.info(f"✓ Model: en_core_web_sm")
    except Exception as e:
        logger.error(f"ERROR: spaCy model not found: {e}")
        logger.error("Please run: python -m spacy download en_core_web_sm")
        sys.exit(1)

    # Log configuration
    logger.info(f"\n📋 CONFIGURATION:")
    logger.info(f"   MH keyword removal: {'ENABLED' if remove_mh else 'DISABLED'}")
    logger.info(f"   Reddit_Combi balancing: {'ENABLED ⚖️' if args.balance_reddit_combi else 'DISABLED ⚠️'}")
    logger.info(f"   Random seed: {args.seed}")

    # Process datasets
    process_all = args.all or args.dataset == 'all'

    if process_all or args.dataset == 'dreaddit':
        dreaddit_path = data_dir / 'dreaddit.csv'
        if dreaddit_path.exists():
            process_dreaddit(dreaddit_path, output_dir, nlp, is_test=False,
                           remove_mh_keywords=remove_mh)

    if process_all or args.dataset == 'dreaddit_test':
        for test_name in ['dreaddit_test.csv', 'dreaddit-test.csv']:
            test_path = data_dir / test_name
            if test_path.exists():
                process_dreaddit(test_path, output_dir, nlp, is_test=True,
                               remove_mh_keywords=remove_mh)
                break

    if process_all or args.dataset == 'twitter':
        twitter_path = data_dir / 'Twitter_Full.csv'
        if twitter_path.exists():
            process_twitter(twitter_path, output_dir, nlp, remove_mh_keywords=remove_mh)

    # ═══════════════════════════════════════════════════════════════════════════
    # [NEW ] Twitter_sample_gold - Human-labeled Twitter dataset
    # ═══════════════════════════════════════════════════════════════════════════
    if process_all or args.dataset == 'twitter_gold':
        twitter_gold_path = data_dir / 'Twitter_sample_gold.csv'
        if twitter_gold_path.exists():
            process_twitter_gold(twitter_gold_path, output_dir, nlp, remove_mh_keywords=remove_mh)

    if process_all or args.dataset == 'reddit_combi':
        reddit_path = data_dir / 'Reddit_Combi.csv'
        if reddit_path.exists():
            process_reddit_combi(
                reddit_path, output_dir, nlp,
                remove_mh_keywords=remove_mh,
                balance=args.balance_reddit_combi,  # [NEW ]
                seed=args.seed
            )

    # Remove cross-dataset duplicates
    if process_all:
        logger.info("\nChecking for cross-dataset duplicates...")
        remove_cross_duplicates(output_dir)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("📊 DATA PREPARATION COMPLETE")
    logger.info("="*60)

    logger.info(f"\nProcessed files saved to: {output_dir}")

    # Print final dataset info
    logger.info("\n📈 FINAL DATASET SUMMARY:")
    for name in ['dreaddit_train', 'dreaddit_test', 'twitter_processed', 'twitter_gold_processed', 'reddit_combi_processed']:
        path = output_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            n_pos = (df['label'] == 1).sum()
            n_total = len(df)
            ratio = n_pos / n_total * 100

            # Compute majority baseline F1 (predict all as the majority class)
            n_neg = n_total - n_pos
            maj_count = max(n_pos, n_neg)
            maj_prop = maj_count / n_total
            # Majority-class F1: precision=maj_prop, recall=1 → F1=2p/(p+1)
            # Minority-class F1: 0 (no predictions for that class)
            # Macro-F1 = average of both = majority_F1 / 2
            maj_class_f1 = 2 * maj_prop / (maj_prop + 1.0)
            maj_f1 = maj_class_f1 / 2  # Macro-F1 for majority baseline

            maj_label = "Positive" if n_pos >= n_neg else "Negative"
            logger.info(f"   {name}: {n_total:,} samples | "
                       f"pos={n_pos:,} ({ratio:.1f}%) | "
                       f"Majority Class: {maj_label} | Baseline Macro-F1: {maj_f1:.3f}")

    logger.info("\n✅ Next step: Train TSS")
    logger.info("   python scripts/02_train_evaluate.py --channels all")

if __name__ == "__main__":
    main()
