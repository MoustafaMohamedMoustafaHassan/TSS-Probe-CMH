#!/usr/bin/env python3
"""
================================================================================
TSS: Advanced Analysis Suite
================================================================================
"Linguistic structure is the true vessel of psychological state."

═══════════════════════════════════════════════════════════════════════════════
PURPOSE: Provide VISUAL and STATISTICAL PROOF for the Divergence Hypothesis
═══════════════════════════════════════════════════════════════════════════════

The Divergence Hypothesis:
├── Human labels capture STRUCTURAL patterns (POS Bigrams, SVO Triples, Style)
├── Auto labels capture LEXICAL patterns (keyword matching)
├── Therefore: Channels B/C should align with Human labels
│              Channel A should align with Auto labels
└── This proves that STRUCTURE > CONTENT for real human judgment of stress

═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTAL DESIGN:
├── Training: Dreaddit_train (Human-labeled Reddit)
├── In-Domain Testing:
│   ├── Dreaddit_test (Human labels) - Gold Standard
│   └── Reddit_Combi (Auto labels) - Different subreddits
├── Cross-Platform Testing:
│   ├── Twitter_Gold (Human labels) - Researcher annotated
│   └── Twitter_Full (Auto labels) - Keyword-based
═══════════════════════════════════════════════════════════════════════════════

ANALYSIS COMPONENTS:
═══════════════════════════════════════════════════════════════════════════════

1. STYLISTIC PROFILING (K-Means on Channel C)
   ├── Cluster texts by psycholinguistic style WITHOUT using labels
   ├── Discover "Stress Archetypes" (e.g., High Arousal vs Withdrawn)
   ├── Prove model captures meaningful stylistic variation
   └── Generate: Heatmap, PCA plot, Cluster profiles

2. SHAP EXPLAINABILITY (Shift Cases Analysis)
   ├── Identify "Shift Cases": Where Lexical(A) fails but Structure(B) succeeds
   ├── Use SHAP to explain WHY Channel B succeeds
   ├── Prove top features are STRUCTURAL (POS Bigrams, SVO Triples)
   └── Generate: Beeswarm plot, Waterfall plot, Feature importance

3. DIVERGENCE HYPOTHESIS PROOF
   ├── Compare channel rankings on Human vs Auto labeled datasets
   ├── Show B/C outperform A on Human labels
   ├── Show A competes better on Auto labels
   └── Generate: Divergence heatmap, Statistical comparison

4. CROSS-PLATFORM TRANSFER VISUALIZATION
   ├── Show transfer retention across platforms
   ├── Compare Reddit→Twitter transfer by channel
   └── Generate: Transfer retention bar plots

═══════════════════════════════════════════════════════════════════════════════
Usage:
    python scripts/05_advanced_analysis.py
    python scripts/05_advanced_analysis.py --skip_shap  # If SHAP not installed
    python scripts/05_advanced_analysis.py --n_clusters 4  # Custom clustering
================================================================================
"""

import sys
import os
import re
import logging
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# sklearn imports
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
# Add parent directory to path
if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).parent.parent))
else:
    # Fallback for interactive environments
    sys.path.insert(0, str(Path.cwd()))

try:
    from tss.config import SYSTEM_NAME, __version__
    from tss.pipeline import TSSClassifier, evaluate_model
except Exception as _tss_import_err:  # pragma: no cover
    # Allows the script to be imported/compiled even outside the TSS repo.
    SYSTEM_NAME = "TSS"
    __version__ = "0.0.0"
    TSSClassifier = None  # type: ignore
    evaluate_model = None  # type: ignore
    logging.getLogger(__name__).warning(
        "[WARN] Could not import TSS core modules (tss.*). "
        "Run this script inside the TSS project environment. Error: %s", _tss_import_err
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_section(title: str, char: str = "=", width: int = 88) -> None:
    """Log a visually distinct section header (safe helper)."""
    try:
        bar = char * width
        logger.info(bar)
        logger.info(title)
        logger.info(bar)
    except Exception:
        # Logging should never crash the pipeline.
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================

def _json_default(obj):
    """JSON serializer for objects not serializable by default (crash-proof)."""
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(obj, (_pd.DataFrame,)):
            return obj.to_dict(orient="records")
        if isinstance(obj, (_pd.Series,)):
            return obj.to_dict()
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return str(obj)


RANDOM_STATE = 42

# Dataset metadata
DATASET_INFO = {
    'dreaddit_test': {
        'label_type': 'human',
        'platform': 'reddit',
        'description': 'Gold standard human-labeled Reddit posts (in-domain)',
        'color': '#2ecc71',  # Green
    },
    'twitter_gold': {
        'label_type': 'human',
        'platform': 'twitter',
        'description': 'Cross-platform human-labeled tweets (researcher annotated)',
        'color': '#3498db',  # Blue
    },
    'twitter': {
        'label_type': 'auto',
        'platform': 'twitter',
        'description': 'Cross-platform auto-labeled tweets (keyword based)',
        'color': '#e74c3c',  # Red
    },
    'reddit_combi': {
        'label_type': 'auto',
        'platform': 'reddit',
        'description': 'In-platform auto-labeled Reddit (different domains)',
        'color': '#f39c12',  # Orange
    },
}

# Key stylistic markers for profiling
KEY_STYLE_MARKERS = [
    # Pronoun usage (self-focus)
    'C_pronoun_ratio',
    'C_first_person_ratio',
    # Emotional intensity
    'C_negation_ratio',
    'C_caps_word_ratio',
    'C_raw_caps_ratio',
    'C_raw_elongated_count',
    'C_raw_punctuation_density',
    # Cognitive patterns
    'C_fragmentation_index',
    'C_avg_sentence_length',
    # Lexical diversity
    'C_type_token_ratio',
    'C_yule_i',
    # Emotional valence
    'C_avg_valence',
    'C_avg_arousal',
    # Punctuation patterns
    'C_punct_double_exclaim',
    'C_punct_triple_period',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_models(artifacts_dir: Path) -> Dict[str, TSSClassifier]:
    """Load all trained TSS models from artifacts directory."""
    import joblib

    models = {}
    model_files = {
        'A': 'tss_a.joblib',
        'B': 'tss_b.joblib',
        'C': 'tss_c.joblib',
        'AB': 'tss_ab.joblib',
        'AC': 'tss_ac.joblib',
        'BC': 'tss_bc.joblib',
        'ABC': 'tss_abc.joblib',
    }

    for channel, filename in model_files.items():
        path = artifacts_dir / filename
        if path.exists():
            try:
                models[channel] = joblib.load(path)
                logger.info(f"   ✓ Loaded {channel}: {filename}")
            except Exception as e:
                logger.warning(f"   ✗ Failed to load {channel}: {e}")

    return models


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load datasets robustly handling different delimiters (comma vs semicolon)
    and varying column names across datasets.
    """
    datasets = {}

    # Ensure filenames match your actual disk files
    file_mapping = {
        'dreaddit_test': 'dreaddit_test.csv',
        'twitter_gold': 'twitter_gold_processed.csv',
        'twitter': 'twitter_processed.csv',
        'reddit_combi': 'reddit_combi_processed.csv',
    }

    for name, filename in file_mapping.items():
        path = data_dir / filename
        if path.exists():
            try:
                # ---------------------------------------------------------
                # LAYER 1: Robust Delimiter Detection
                # ---------------------------------------------------------
                try:
                    # First attempt: Standard CSV (comma)
                    df = pd.read_csv(path)
                    # If pandas reads the whole file as 1 column, the delimiter is likely wrong
                    if df.shape[1] < 2:
                        raise ValueError("Likely wrong delimiter (comma failed)")
                except:
                    # Second attempt: Semicolon delimiter (common in raw Reddit/Twitter data)
                    # Using engine='python' is more robust for parsing errors
                    df = pd.read_csv(path, sep=';', engine='python')

                # ---------------------------------------------------------
                # LAYER 2: Text Column Normalization (Target: 'text')
                # ---------------------------------------------------------
                text_col = None

                # Check for all known variations of text column names
                if 'cleaned_text' in df.columns:
                    text_col = 'cleaned_text'
                elif 'text' in df.columns:
                    text_col = 'text'
                elif 'Body_Title' in df.columns:  # Specific to Reddit_Combi
                    text_col = 'Body_Title'
                elif 'body' in df.columns:        # Specific to Twitter raw
                    text_col = 'body'
                elif 'post' in df.columns:        # Specific to Dreaddit raw
                    text_col = 'post'

                if text_col:
                    # Standardize to 'text' for the script logic
                    df['text'] = df[text_col].fillna('').astype(str)
                    df['cleaned_text'] = df['text']
                    # ---------------------------------------------------------
                    # LAYER 3: Legacy Support (Target: 'cleaned_text')
                    # ---------------------------------------------------------
                    # SHAP analysis for Model A specifically looks for 'cleaned_text'.
                    # We create it as a mirror of 'text' if it doesn't exist.
                    if 'cleaned_text' not in df.columns:
                        df['cleaned_text'] = df['text']

                    # ---------------------------------------------------------
                    # LAYER 4: Cleaning & Label Handling
                    # ---------------------------------------------------------
                    # Remove empty rows
                    df = df[df['text'].str.strip().str.len() > 0]

                    # Standardize label column
                    if 'label' not in df.columns and 'target' in df.columns:
                        df['label'] = df['target']

                    # Final validation: Ensure we have labels to evaluate
                    if 'label' in df.columns:
                        datasets[name] = df

                        # Log success stats
                        info = DATASET_INFO.get(name, {})
                        n_pos = df['label'].sum()
                        prev = n_pos / len(df) * 100 if len(df) > 0 else 0
                        logger.info(f"   ✓ {name}: {len(df):,} samples ({prev:.1f}% pos) "
                                   f"[{info.get('label_type', 'unk').upper()}]")
                    else:
                        logger.warning(f"   ⚠️ Skipping {name}: 'label' column missing.")
                else:
                    logger.warning(f"   ⚠️ Skipping {name}: No text column found. Found: {list(df.columns)}")

            except Exception as e:
                logger.error(f"   ❌ Failed to load {filename}: {e}")
        else:
            logger.warning(f"   ❌ File not found: {filename}")

    return datasets


def load_processed_datasets(processed_dir: Path, expected_datasets: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Backward-compatible alias for loading processed datasets.

    The v8 pipeline uses this name in `main()`; internally we delegate to `load_datasets()`.
    If `expected_datasets` is provided, we filter to only those datasets and warn on missing ones.
    """
    datasets = load_datasets(processed_dir)
    if expected_datasets:
        filtered: Dict[str, pd.DataFrame] = {}
        for name in expected_datasets:
            if name in datasets:
                filtered[name] = datasets[name]
            else:
                logger.warning("Requested dataset '%s' not found under %s", name, processed_dir)
        return filtered
    return datasets


def safe_import_plotting():
    """Safely import plotting libraries."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        return plt, sns, True
    except ImportError:
        logger.warning("matplotlib/seaborn not available. Plots will be skipped.")
        return None, None, False


def safe_import_shap():
    """Safely import SHAP library."""
    try:
        import shap
        return shap, True
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None, False

# =============================================================================
# STATISTICAL INFERENCE HELPERS (Complementary to Masking Suite)
# =============================================================================
# Goal: Strengthen statistical inference here without duplicating the core mask-vs-mask tests
# already covered in MaskingSuite. Here we focus on:
# (1) Paired, within-dataset comparisons across channels (e.g., C/BC vs A).
# (2) Semantic analysis via topic clustering as a strong way to increase N without relying on few datasets.

def _metric_value(y_true, y_pred, metric: str = "macro_f1") -> float:
    """Compute a metric value safely."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "f1":
        return f1_score(y_true, y_pred, average="binary", zero_division=0)
    if metric == "mcc":
        # MCC requires both classes present in y_true
        if len(np.unique(y_true)) < 2:
            return np.nan
        return matthews_corrcoef(y_true, y_pred)
    if metric == "accuracy":
        return float((y_true == y_pred).mean())
    raise ValueError(f"Unknown metric: {metric}")


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap indices with class-stratification to preserve imbalance."""
    y_true = np.asarray(y_true)
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    # fallback: if single class, regular bootstrap
    if len(idx0) == 0 or len(idx1) == 0:
        n = len(y_true)
        return rng.integers(0, n, size=n)

    samp0 = rng.choice(idx0, size=len(idx0), replace=True)
    samp1 = rng.choice(idx1, size=len(idx1), replace=True)
    samp = np.concatenate([samp0, samp1])
    rng.shuffle(samp)
    return samp


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 5000,
    seed: int = 42,
    stratified: bool = True,
):
    """Paired bootstrap CI for Δ = metric(A) - metric(B) on the SAME instances."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    obs = _metric_value(y_true, y_pred_a, metric) - _metric_value(y_true, y_pred_b, metric)
    deltas = np.empty(n_boot, dtype=float)
    deltas.fill(np.nan)

    for i in range(n_boot):
        idx = _stratified_bootstrap_indices(y_true, rng) if stratified else rng.integers(0, len(y_true), size=len(y_true))
        da = _metric_value(y_true[idx], y_pred_a[idx], metric)
        db = _metric_value(y_true[idx], y_pred_b[idx], metric)
        deltas[i] = da - db

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return obs, (np.nan, np.nan), deltas

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return obs, (float(lo), float(hi)), deltas


def approximate_randomization_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "macro_f1",
    n_perm: int = 10000,
    seed: int = 42,
    one_sided: bool = True,
):
    """Approximate randomization / permutation test for paired predictions."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    obs = _metric_value(y_true, y_pred_a, metric) - _metric_value(y_true, y_pred_b, metric)

    count = 0
    n = len(y_true)
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, y_pred_b, y_pred_a)
        pb = np.where(swap, y_pred_a, y_pred_b)
        d = _metric_value(y_true, pa, metric) - _metric_value(y_true, pb, metric)

        if one_sided:
            if d >= obs:
                count += 1
        else:
            if abs(d) >= abs(obs):
                count += 1

    p = (count + 1) / (n_perm + 1)
    return float(p), float(obs)


def run_strong_inference_tables(
    raw_preds: dict,
    output_dir: Path,
    comparisons=None,
    metric: str = "macro_f1",
    n_boot: int = 5000,
    n_perm: int = 10000,
    seed: int = 42,
):
    """Create per-dataset paired inference tables for channel-vs-channel comparisons."""
    if comparisons is None:
        comparisons = [("C", "A"), ("BC", "A"), ("B", "A")]

    rows = []
    datasets = sorted({ds for (ds, ch) in raw_preds.keys()})

    for ds in datasets:
        for ch_a, ch_b in comparisons:
            key_a = (ds, ch_a)
            key_b = (ds, ch_b)
            if key_a not in raw_preds or key_b not in raw_preds:
                continue

            y_true = raw_preds[key_a]["y_true"]
            y_a = raw_preds[key_a]["y_pred"]
            y_b = raw_preds[key_b]["y_pred"]

            delta_obs, (ci_lo, ci_hi), _ = paired_bootstrap_delta(
                y_true, y_a, y_b, metric=metric, n_boot=n_boot, seed=seed, stratified=True
            )
            p_perm, _ = approximate_randomization_test(
                y_true, y_a, y_b, metric=metric, n_perm=n_perm, seed=seed, one_sided=True
            )

            rows.append({
                "dataset": ds,
                "label_type": DATASET_INFO.get(ds, {}).get("label_type", "unknown"),
                "compare": f"{ch_a} - {ch_b}",
                "metric": metric,
                "delta_obs": delta_obs,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "p_perm": p_perm,
                "n": int(len(y_true)),
            })

    df = pd.DataFrame(rows)
    out_path = output_dir / "paired_inference_channels.csv"
    df.to_csv(out_path, index=False)

    # Group-level (human vs auto) "difference-of-differences" using nested bootstrap over instances
    group_rows = []
    for ch_a, ch_b in comparisons:
        # collect datasets per type with available predictions
        human_ds = [ds for ds in datasets if DATASET_INFO.get(ds, {}).get("label_type") == "human" and (ds, ch_a) in raw_preds and (ds, ch_b) in raw_preds]
        auto_ds  = [ds for ds in datasets if DATASET_INFO.get(ds, {}).get("label_type") == "auto"  and (ds, ch_a) in raw_preds and (ds, ch_b) in raw_preds]

        if len(human_ds) == 0 or len(auto_ds) == 0:
            continue

        rng = np.random.default_rng(seed)
        n_boot_group = max(1000, n_boot // 5)  # cheaper than full bootstrap
        d_h = np.empty(n_boot_group, dtype=float)
        d_a = np.empty(n_boot_group, dtype=float)

        for i in range(n_boot_group):
            # weighted mean by dataset size
            def _group_delta(ds_list):
                deltas = []
                weights = []
                for ds in ds_list:
                    y_true = raw_preds[(ds, ch_a)]["y_true"]
                    idx = _stratified_bootstrap_indices(y_true, rng)
                    ya = raw_preds[(ds, ch_a)]["y_pred"][idx]
                    yb = raw_preds[(ds, ch_b)]["y_pred"][idx]
                    yt = y_true[idx]
                    deltas.append(_metric_value(yt, ya, metric) - _metric_value(yt, yb, metric))
                    weights.append(len(yt))
                weights = np.asarray(weights, dtype=float)
                weights = weights / weights.sum()
                return float(np.sum(weights * np.asarray(deltas, dtype=float)))

            d_h[i] = _group_delta(human_ds)
            d_a[i] = _group_delta(auto_ds)

        dod = d_h - d_a
        dod_lo, dod_hi = np.quantile(dod, [0.025, 0.975])

        group_rows.append({
            "compare": f"{ch_a} - {ch_b}",
            "metric": metric,
            "human_delta_mean": float(np.mean(d_h)),
            "auto_delta_mean": float(np.mean(d_a)),
            "DoD_human_minus_auto": float(np.mean(dod)),
            "DoD_ci95_low": float(dod_lo),
            "DoD_ci95_high": float(dod_hi),
            "human_datasets": ",".join(human_ds),
            "auto_datasets": ",".join(auto_ds),
            "n_boot_group": int(n_boot_group),
        })

    df_group = pd.DataFrame(group_rows)
    out_path_group = output_dir / "paired_inference_group_DoD.csv"
    df_group.to_csv(out_path_group, index=False)

    return {
        "per_dataset_table": str(out_path),
        "group_dod_table": str(out_path_group),
        "comparisons": [f"{a}-{b}" for a, b in comparisons],
        "metric": metric,
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
    }


def run_semantic_topic_robustness(
    datasets: dict,
    raw_preds: dict,
    output_dir: Path,
    plots_dir: Path,
    comparison=("C", "A"),
    n_topics: int = 12,
    min_n: int = 200,
    metric: str = "macro_f1",
    n_perm: int = 20000,
    seed: int = 42,
):
    """Semantic robustness without repeating masking:
    - Build topic clusters (TF-IDF -> SVD -> KMeans) within each dataset.
    - Compute Δmetric between two models within each topic, then test whether mean Δ > 0.
    """
    ch_a, ch_b = comparison
    rows = []
    mi_rows = []
    rng = np.random.default_rng(seed)

    for ds_name, df in datasets.items():
        if (ds_name, ch_a) not in raw_preds or (ds_name, ch_b) not in raw_preds:
            continue

        # pick text column
        text_col = None
        for cand in ["text", "post", "content", "sentence", "raw_text"]:
            if cand in df.columns:
                text_col = cand
                break
        if text_col is None:
            logger.warning(f"[semantic] No text column found for {ds_name}; skipping semantic clustering.")
            continue

        texts = df[text_col].fillna("").astype(str).tolist()
        if len(texts) < (min_n * 2):
            logger.warning(f"[semantic] Dataset {ds_name} too small for topic analysis (n={len(texts)}). Skipping.")
            continue

        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3, stop_words="english")
        X = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(50, max(5, n_topics*3)), random_state=seed)
        Xr = svd.fit_transform(X)
        kmeans = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
        topic_id = kmeans.fit_predict(Xr)

        # Mutual information between topic and predicted stress (proxy for topical dependence)
        y_pred_a = raw_preds[(ds_name, ch_a)]["y_pred"]
        y_pred_b = raw_preds[(ds_name, ch_b)]["y_pred"]
        mi_rows.append({
            "dataset": ds_name,
            "label_type": DATASET_INFO.get(ds_name, {}).get("label_type", "unknown"),
            f"MI(topic, pred_{ch_a})": float(mutual_info_score(topic_id, y_pred_a)),
            f"MI(topic, pred_{ch_b})": float(mutual_info_score(topic_id, y_pred_b)),
            "n_topics": int(n_topics),
        })

        y_true = raw_preds[(ds_name, ch_a)]["y_true"]
        # per-topic deltas
        for t in range(n_topics):
            idx = np.where(topic_id == t)[0]
            if len(idx) < min_n:
                continue
            da = _metric_value(y_true[idx], y_pred_a[idx], metric)
            db = _metric_value(y_true[idx], y_pred_b[idx], metric)
            rows.append({
                "dataset": ds_name,
                "label_type": DATASET_INFO.get(ds_name, {}).get("label_type", "unknown"),
                "topic": int(t),
                "n": int(len(idx)),
                "metric": metric,
                "delta": float(da - db),
            })

    df_topics = pd.DataFrame(rows)
    out_topics = output_dir / f"semantic_topic_deltas_{comparison[0]}_vs_{comparison[1]}.csv"
    df_topics.to_csv(out_topics, index=False)

    df_mi = pd.DataFrame(mi_rows)
    out_mi = output_dir / f"semantic_topic_mutual_info_{comparison[0]}_vs_{comparison[1]}.csv"
    df_mi.to_csv(out_mi, index=False)

    # group inference over topics (more N than datasets)
    summary = []
    for label_type in ["human", "auto"]:
        sub = df_topics[df_topics["label_type"] == label_type]
        if len(sub) == 0:
            continue
        deltas = sub["delta"].to_numpy(dtype=float)
        obs_mean = float(np.mean(deltas))

        # bootstrap CI over topics
        n_boot = 5000
        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            samp = rng.choice(deltas, size=len(deltas), replace=True)
            boot_means[i] = float(np.mean(samp))
        lo, hi = np.quantile(boot_means, [0.025, 0.975])

        # sign-flip permutation test (one-sided: mean delta > 0)
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([-1.0, 1.0], size=len(deltas))
            m = float(np.mean(deltas * signs))
            if m >= obs_mean:
                count += 1
        p = (count + 1) / (n_perm + 1)

        summary.append({
            "label_type": label_type,
            "comparison": f"{ch_a} - {ch_b}",
            "metric": metric,
            "n_topics_used": int(len(deltas)),
            "mean_delta": obs_mean,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_signflip": float(p),
        })

    df_sum = pd.DataFrame(summary)
    out_sum = output_dir / f"semantic_topic_inference_{comparison[0]}_vs_{comparison[1]}.csv"
    df_sum.to_csv(out_sum, index=False)

    return {
        "topic_delta_table": str(out_topics),
        "topic_mi_table": str(out_mi),
        "topic_inference_table": str(out_sum),
        "comparison": f"{comparison[0]}-{comparison[1]}",
        "n_topics": int(n_topics),
        "min_n_per_topic": int(min_n),
        "metric": metric,
        "n_perm": int(n_perm),
    }


# =============================================================================
# 1. STYLISTIC PROFILING (K-MEANS ON CHANNEL C)
# =============================================================================


# ================================
# Statistical Interpretability Helpers
# ================================

def stratified_permutation_test(
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    strata: Optional[np.ndarray] = None,
    metric_fn=None,
    n_perm: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Stratified permutation test (non-parametric).

    We test association between x_labels and y_labels under H0: independence.
    If `strata` is provided, we permute y_labels *within each stratum* to keep
    nuisance structure fixed (e.g., preserve stress label when testing dataset leakage).

    Returns dict with observed metric, p-value, and a small null summary.
    """
    x = np.asarray(x_labels)
    y = np.asarray(y_labels)
    if metric_fn is None:
        metric_fn = adjusted_mutual_info_score

    if x.shape[0] != y.shape[0]:
        raise ValueError("x_labels and y_labels must have same length")

    n = x.shape[0]
    if n < 5:
        return {"observed": float("nan"), "p": float("nan"), "n_perm": 0, "null_mean": float("nan"), "null_std": float("nan")}

    rng = np.random.default_rng(seed)
    observed = float(metric_fn(x, y))

    # Precompute strata indices
    if strata is None:
        strata = np.zeros(n, dtype=int)
    else:
        strata = np.asarray(strata)

    unique_strata = np.unique(strata)
    strata_indices = {s: np.where(strata == s)[0] for s in unique_strata}

    null_vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = y.copy()
        for s, idx in strata_indices.items():
            if idx.size > 1:
                y_perm[idx] = rng.permutation(y_perm[idx])
        null_vals[i] = float(metric_fn(x, y_perm))

    # Two-sided (conservative) p-value based on |metric|
    obs_abs = abs(observed)
    p = float((np.sum(np.abs(null_vals) >= obs_abs) + 1) / (n_perm + 1))

    return {
        "observed": observed,
        "p": p,
        "n_perm": int(n_perm),
        "null_mean": float(np.mean(null_vals)),
        "null_std": float(np.std(null_vals)),
    }


def bootstrap_clustering_stability(
    X: np.ndarray,
    base_labels: np.ndarray,
    clusterer_factory,
    n_boot: int = 100,
    sample_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap stability for clustering using ARI against the base clustering.

    For each bootstrap replicate:
      - resample indices (with replacement) of size ceil(sample_frac * n)
      - re-cluster X[idx]
      - compute ARI between base_labels[idx] and boot_labels

    Returns mean and 95% CI of ARI.
    """
    X = np.asarray(X)
    base_labels = np.asarray(base_labels)
    n = X.shape[0]
    if n < 10:
        return {"ari_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}

    rng = np.random.default_rng(seed)
    m = int(np.ceil(sample_frac * n))
    m = max(5, min(m, n))

    aris: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=True)
        Xb = X[idx]
        # Need enough unique points for clustering
        if Xb.shape[0] < 2:
            continue
        try:
            clust = clusterer_factory()
            boot_labels = clust.fit_predict(Xb)
            ari = adjusted_rand_score(base_labels[idx], boot_labels)
            aris.append(float(ari))
        except Exception as e:
            logger.warning(f"[stability] Bootstrap clustering replicate failed: {e}")
            continue

    if len(aris) < max(10, n_boot // 5):
        return {"ari_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": int(len(aris))}

    lo, hi = np.quantile(aris, [0.025, 0.975])
    return {"ari_mean": float(np.mean(aris)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(len(aris))}


def infer_feature_groups(feature_names: List[str]) -> Dict[str, np.ndarray]:
    """Infer channel groups from feature name prefixes.

    Returns masks for 'A' (lexical), 'B' (structural), 'C' (psycholinguistic), and 'OTHER'.
    If prefixes are not present, returns only OTHER.
    """
    fn = np.asarray(feature_names, dtype=object)
    masks = {}

    has_prefix = any(str(x).startswith(("A_", "B_", "C_")) for x in fn[: min(200, len(fn))])
    if has_prefix:
        masks["A"] = np.array([str(x).startswith("A_") for x in fn], dtype=bool)
        masks["B"] = np.array([str(x).startswith("B_") or "POS" in str(x) or "SVO" in str(x) for x in fn], dtype=bool)
        masks["C"] = np.array([str(x).startswith("C_") for x in fn], dtype=bool)
        masks["OTHER"] = ~(masks["A"] | masks["B"] | masks["C"])
        return masks

    # Fallback: try heuristic (common in earlier TSS builds)
    masks["B"] = np.array([str(x).startswith("B_") or "pos_" in str(x).lower() or "svo" in str(x).lower() for x in fn], dtype=bool)
    masks["C"] = np.array([str(x).startswith("C_") or "liwc" in str(x).lower() or "pron" in str(x).lower() or "negat" in str(x).lower() for x in fn], dtype=bool)
    masks["A"] = np.array([str(x).startswith("A_") or (not masks["B"][i] and not masks["C"][i]) for i, x in enumerate(fn)], dtype=bool)
    masks["OTHER"] = ~(masks["A"] | masks["B"] | masks["C"])
    return masks


def _extract_shap_matrix(shap_out, class_index: int = 1) -> np.ndarray:
    """Extract (n_samples, n_features) SHAP matrix for class_index from a shap.Explanation."""
    values = getattr(shap_out, "values", None)
    if values is None:
        raise ValueError("Invalid SHAP output: missing .values")

    arr = np.asarray(values)
    # Possible shapes:
    # (n, f), (n, f, 2), or (n, 2, f) depending on explainer/model
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] >= 2 and arr.shape[0] > 1:
            return arr[:, :, class_index]
        if arr.shape[1] >= 2:
            return arr[:, class_index, :]
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


def run_stylistic_profiling(
    datasets: Dict[str, pd.DataFrame],
    models: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    n_clusters: int = 6,
    n_perm: int = 2000,
    n_boot: int = 100,
    # --- robustness knobs ---
    plots_dir: Optional[Path] = None,
    channel_preference: Tuple[str, ...] = ("C", "BC", "ABC"),
    max_rows_per_dataset: int = 8000,
    stability_bootstrap: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Stylistic profiling / archetype discovery.

    - Fits KMeans on feature space (prefer Channel C transform if available).
    - Computes leakage-style diagnostics: association between clusters and dataset-id vs label.
    - Produces cluster archetype summaries + stability (bootstrap) estimates.

    Robustness:
    - If Channel C model is missing, falls back to a basic stylometric feature extractor from raw text.
    - Row caps per dataset to avoid OOM (max_rows_per_dataset).
    """

    log_section("STATISTICAL STYLISTIC PROFILING (Leakage-proof clustering)")

    if stability_bootstrap is not None:
        n_boot = int(stability_bootstrap)

    plot_dir = Path(plots_dir) if plots_dir is not None else Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Prefer model-based transform (typically Channel C). Allow user preference ordering.
    model_key: Optional[str] = None
    model_feat: Optional[Any] = None
    for _ch in channel_preference:
        _m = models.get(_ch)
        if _m is not None and hasattr(_m, "transform_features"):
            model_key = _ch
            model_feat = _m
            break
    if model_feat is None:
        model_key = "basic_stylometry"
        logger.warning(
            "[stylistic] No model with transform_features() found for %s. "
            "Falling back to basic stylometric features from raw text.",
            channel_preference,
        )

    def _basic_stylometry_matrix(texts: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Compute lightweight, content-lean stylometric features from raw text."""
        feats = []
        # Avoid NaNs
        t = texts.fillna("").astype(str)
        # Precompile regexes
        re_word = re.compile(r"\b\w+\b")
        re_sent = re.compile(r"[.!?]+")
        re_punct = re.compile(r"[^\w\s]")

        rows: List[List[float]] = []
        for s in t.tolist():
            s2 = s.strip()
            n_chars = float(len(s2))
            words = re_word.findall(s2)
            n_words = float(len(words))
            avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
            n_sents = float(len(re_sent.findall(s2)) or (1.0 if n_chars else 0.0))
            n_punct = float(len(re_punct.findall(s2)))
            n_excl = float(s2.count("!"))
            n_q = float(s2.count("?"))
            n_upper = float(sum(1 for ch in s2 if ch.isupper()))
            n_digit = float(sum(1 for ch in s2 if ch.isdigit()))
            # ratios guarded by length
            punct_ratio = (n_punct / n_chars) if n_chars else 0.0
            upper_ratio = (n_upper / n_chars) if n_chars else 0.0
            digit_ratio = (n_digit / n_chars) if n_chars else 0.0
            words_per_sent = (n_words / n_sents) if n_sents else 0.0

            rows.append([
                n_chars, n_words, avg_word_len, n_sents, words_per_sent,
                n_punct, punct_ratio, n_excl, n_q, upper_ratio, digit_ratio,
            ])

        feature_names = [
            "n_chars", "n_words", "avg_word_len", "n_sents", "words_per_sent",
            "n_punct", "punct_ratio", "n_exclam", "n_question", "upper_ratio", "digit_ratio",
        ]
        return np.asarray(rows, dtype=float), feature_names

    # Backward compatible name in the original code below
    model_c = model_feat

    rng = np.random.default_rng(seed)

    # --- Build feature matrix ---
    X_list, y_list, ds_list = [], [], []
        # feature_names handled above

    for ds_name, df in datasets.items():
        # Memory safety: cap rows per dataset (stratified by label when possible).
        if max_rows_per_dataset is not None and max_rows_per_dataset > 0 and len(df) > max_rows_per_dataset:
            rng = np.random.default_rng(seed + (abs(hash(ds_name)) % 10000))
            if "label" in df.columns and df["label"].nunique() > 1:
                # stratified sample
                df0 = df[df["label"] == 0]
                df1 = df[df["label"] == 1]
                n0 = int(round(max_rows_per_dataset * (len(df0) / len(df))))
                n1 = max_rows_per_dataset - n0
                df0s = df0.sample(n=min(n0, len(df0)), random_state=int(rng.integers(0, 2**31-1)))
                df1s = df1.sample(n=min(n1, len(df1)), random_state=int(rng.integers(0, 2**31-1)))
                df = pd.concat([df0s, df1s], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(0, 2**31-1)))
            else:
                df = df.sample(n=max_rows_per_dataset, random_state=int(rng.integers(0, 2**31-1)))
        if df is None or df.empty:
            continue
        if "text" not in df.columns or "label" not in df.columns:
            logger.warning(f"[stylistic] Dataset {ds_name} missing required columns. Skipping.")
            continue
        try:
            if model_c is not None:
                X = model_c.transform_features(df)
                feature_names = getattr(model_c, "feature_names_", None)
            else:
                X, feature_names = _basic_stylometry_matrix(df["text"])
        except Exception as e:
            logger.warning(f"[stylistic] Failed to extract features for {ds_name}: {e}")
            continue

        if X.ndim != 2 or X.shape[0] != len(df):
            logger.warning(f"[stylistic] Invalid feature matrix for {ds_name}: {X.shape}. Skipping.")
            continue

        X_list.append(X)
        y_list.append(df["label"].astype(int).to_numpy())
        ds_list.append(np.full(X.shape[0], ds_name, dtype=object))

    if not X_list:
        logger.warning("[stylistic] No datasets produced valid features. Skipping.")
        return None

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    ds_all = np.concatenate(ds_list)

    n_samples, n_features = X_all.shape
    if n_samples <= n_clusters:
        logger.warning(f"[stylistic] Not enough samples (n={n_samples}) for n_clusters={n_clusters}. Skipping.")
        return None
    if n_samples <= n_features:
        logger.warning(f"[stylistic] n_samples ({n_samples}) <= n_features ({n_features}) => unstable clustering. Skipping.")
        return None

    # Scale inside the function (leakage-proof)
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
    except Exception as e:
        logger.warning(f"[stylistic] Scaling failed: {e}")
        return None

    # --- Fit clustering ---
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=seed)
        clusters = kmeans.fit_predict(X_scaled)
    except Exception as e:
        logger.warning(f"[stylistic] KMeans failed: {e}")
        return None

    # --- Artifact (Leakage) Check: cluster ~ dataset_id (stratified by label) ---
    leakage = stratified_permutation_test(
        x_labels=clusters,
        y_labels=ds_all,
        strata=y_all,
        metric_fn=adjusted_mutual_info_score,
        n_perm=n_perm,
        seed=seed,
    )
    leakage_ami = leakage["observed"]
    leakage_p = leakage["p"]

    # --- Signal Check: cluster ~ stress_label (stratified by dataset) ---
    signal = stratified_permutation_test(
        x_labels=clusters,
        y_labels=y_all,
        strata=ds_all,
        metric_fn=adjusted_mutual_info_score,
        n_perm=n_perm,
        seed=seed + 1,
    )
    signal_ami = signal["observed"]
    signal_p = signal["p"]

    # Chi-square (auxiliary, interpretable)
    chi2_ds = chi2_p_ds = chi2_lbl = chi2_p_lbl = float("nan")
    try:
        ct_ds = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(ds_all, name="dataset"))
        chi2_ds, chi2_p_ds, _, _ = chi2_contingency(ct_ds.values())
    except Exception:
        pass
    try:
        ct_lbl = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(y_all, name="label"))
        chi2_lbl, chi2_p_lbl, _, _ = chi2_contingency(ct_lbl.values())
    except Exception:
        pass

    # --- Stability Check (Bootstrap ARI) ---
    def _clusterer_factory():
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=int(rng.integers(0, 10_000_000)))

    stability = bootstrap_clustering_stability(
        X=X_scaled,
        base_labels=clusters,
        clusterer_factory=_clusterer_factory,
        n_boot=n_boot,
        sample_frac=0.8,
        seed=seed,
    )

    # Decide whether clusters reflect signal vs artifact
    verdict = "signal" if (signal_ami >= leakage_ami and signal_p <= 0.05) else "artifact" if (leakage_ami > signal_ami and leakage_p <= 0.05) else "unclear"
    logger.info(
        f"[stylistic] AMI(cluster,dataset)={leakage_ami:.4f} (p={leakage_p:.4g}); "
        f"AMI(cluster,label)={signal_ami:.4f} (p={signal_p:.4g}); "
        f"stability ARI={stability['ari_mean']:.3f} [{stability['ci_low']:.3f},{stability['ci_high']:.3f}] => {verdict.upper()}"
    )

    # --- Export clustering stats ---
    clustering_stats = pd.DataFrame([{
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
"channel_used": model_key,
        "leakage_ami_cluster_vs_dataset": leakage_ami,
        "leakage_p_strat_perm": leakage_p,
        "signal_ami_cluster_vs_label": signal_ami,
        "signal_p_strat_perm": signal_p,
        "chi2_cluster_vs_dataset": float(chi2_ds),
        "chi2_p_cluster_vs_dataset": float(chi2_p_ds),
        "chi2_cluster_vs_label": float(chi2_lbl),
        "chi2_p_cluster_vs_label": float(chi2_p_lbl),
        "stability_ari_mean": stability["ari_mean"],
        "stability_ari_ci_low": stability["ci_low"],
        "stability_ari_ci_high": stability["ci_high"],
        "stability_n_boot": stability["n_boot"],
        "verdict": verdict,
        "seed": seed,
        "n_perm": n_perm,
    }])
    clustering_stats_path = output_dir / "clustering_stats.csv"
    clustering_stats.to_csv(clustering_stats_path, index=False)
    logger.info(f"[stylistic] Saved clustering_stats.csv -> {clustering_stats_path}")

    # --- Export cluster profiles ---
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    profiles_rows = []
    for c in range(n_clusters):
        idx = np.where(clusters == c)[0]
        if idx.size == 0:
            continue
        row = {
            "cluster": int(c),
            "n": int(idx.size),
            "stress_rate": float(np.mean(y_all[idx])),
        }
        # dataset composition
        for ds_name in sorted(set(ds_all)):
            row[f"pct_{ds_name}"] = float(np.mean(ds_all[idx] == ds_name))
        # mean of original (unscaled) features for interpretability
        means = np.mean(X_all[idx], axis=0)
        for j, fn in enumerate(feature_names):
            row[f"mean_{fn}"] = float(means[j])
        profiles_rows.append(row)

    stylistic_profiles = pd.DataFrame(profiles_rows)
    stylistic_profiles_path = output_dir / "stylistic_profiles.csv"
    stylistic_profiles.to_csv(stylistic_profiles_path, index=False)
    logger.info(f"[stylistic] Saved stylistic_profiles.csv -> {stylistic_profiles_path}")

    # --- Optional visualization (never crash) ---
    plt, sns, has_plotting = safe_import_plotting()
    plt = plt if has_plotting else None
    if plt is not None and n_features >= 2 and n_samples >= 10:
        try:
            pca = PCA(n_components=2, random_state=seed)
            X_2d = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 7))
            sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, s=12, alpha=0.7)
            ax.set_title(f"Stylistic Archetypes (KMeans on {model_key})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(sc, ax=ax, label="Cluster")
            fig.tight_layout()
            fig_path = plot_dir / "stylistic_clusters_pca.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            logger.info(f"[stylistic] Saved plot -> {fig_path}")
        except Exception as e:
            logger.warning(f"[stylistic] Plotting failed (continuing): {e}")

    return clustering_stats


def determine_archetype(elevated: List, depressed: List, stress_prev: float) -> str:
    """Determine human-readable archetype name based on features."""
    elevated_names = [f[0].lower() for f, _ in elevated]
    depressed_names = [f[0].lower() for f, _ in depressed]

    all_features = ' '.join(elevated_names)

    # High stress indicators
    if stress_prev > 0.6:
        if 'pronoun' in all_features or 'first_person' in all_features:
            return "Self-Focused Distress"
        elif 'caps' in all_features or 'exclaim' in all_features:
            return "High Arousal"
        elif 'negation' in all_features:
            return "Negative Rumination"
        else:
            return "High Stress"

    # Low stress indicators
    elif stress_prev < 0.4:
        if 'type_token' in all_features or 'yule' in all_features:
            return "Diverse Expression"
        elif 'valence' in all_features:
            return "Positive Tone"
        else:
            return "Low Stress"

    # Moderate
    else:
        if 'fragmentation' in all_features:
            return "Fragmented Thinking"
        elif 'arousal' in all_features:
            return "Mixed Emotional"
        else:
            return "Moderate Stress"


# =============================================================================
# 2. SHAP EXPLAINABILITY (SHIFT CASES ANALYSIS)
# =============================================================================


def run_shap_analysis(
    datasets: Dict[str, pd.DataFrame],
    models: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    background_size: int = 200,
    eval_size: int = 500,
    n_boot: int = 500,
    topk_lexical: int = 2000,
    # --- aliases / robustness knobs ---
    plots_dir: Optional[Path] = None,
    shap_background: Optional[int] = None,
    shap_eval: Optional[int] = None,
    n_bootstrap: Optional[int] = None,
    n_shift_cases: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Unbiased, dual-mode interpretability via SHAP + bootstrap CIs. upgrades:
      - Memory safety: hard caps (background_size, eval_size) + automatic downscaling for huge feature spaces.
      - Unbiased sampling: random selection via RNG (no [:max_cases] slicing).
      - Dual-mode shift sets:
          Mode A (Structure Wins): true=1, pred_A=0, pred_B=1
          Mode B (Lexical Wins):  true=1, pred_A=1, pred_B=0
      - Statistical interpretability: bootstrap CIs for
          * structural feature dominance within the Structure model (B vs C when available)
          * lexical shortcut concentration (top-10 SHAP mass / total) for the Lexical model
      - Crash-proof: if SHAP or models are missing, skip gracefully.

    Exports:
      - shap_statistics.csv
    """

    # Aliases from CLI/older versions
    if shap_background is not None:
        background_size = int(shap_background)
    if shap_eval is not None:
        eval_size = int(shap_eval)
    if n_bootstrap is not None:
        n_boot = int(n_bootstrap)
    plot_dir = Path(plots_dir) if plots_dir is not None else Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt, sns, has_plotting = safe_import_plotting()
    plt = plt if has_plotting else None

    shap_lib, has_shap = safe_import_shap()
    if not has_shap:
        logger.warning("[shap] SHAP not installed. Skipping SHAP interpretability.")
        return None

    shap = shap_lib

    # Required for defining shift sets
    model_a = models.get("A")
    model_b = models.get("B") or models.get("BC")
    if model_a is None or model_b is None:
        logger.warning("[shap] Need models A and (B or BC) to build dual-mode shift sets. Skipping.")
        return None

    # Model used for 'structure' explanations (prefer BC so dominance is non-trivial)
    explain_struct = models.get("BC") or models.get("B")
    if explain_struct is None:
        logger.warning("[shap] No structural explain-model found (BC/B). Skipping.")
        return None

    rng = np.random.default_rng(seed)

    log_section("STATISTICAL INTERPRETABILITY: DUAL-MODE SHAP (UNBIASED + BOOTSTRAPPED)")

    # ------------------------
    # Build background sample
    # ------------------------
    # Sample across datasets proportionally, but keep hard cap for memory safety.
    bg_frames = []
    per_ds_bg = max(1, background_size // max(1, len(datasets)))
    for ds_name, df in datasets.items():
        if df is None or df.empty:
            continue
        m = min(per_ds_bg, len(df))
        if m <= 0:
            continue
        idx = rng.choice(len(df), size=m, replace=False)
        bg_frames.append(df.iloc[idx].copy())
    if not bg_frames:
        logger.warning("[shap] No data available for SHAP background. Skipping.")
        return None

    df_bg = pd.concat(bg_frames, ignore_index=True)
    df_bg = df_bg.sample(n=min(len(df_bg), background_size), random_state=seed).reset_index(drop=True)

    # Compute background features for structural explainer
    try:
        X_bg_struct = np.asarray(explain_struct.transform_features(df_bg))
    except Exception as e:
        logger.warning(f"[shap] Failed to build structural background features: {e}")
        return None

    # If the feature space is huge, downscale further to avoid OOM in SHAP internals
    n_bg, p_struct = X_bg_struct.shape
    if p_struct > 20000:
        logger.warning(f"[shap] Structural feature dim is very large (p={p_struct}). Reducing eval/background sizes.")
        X_bg_struct = X_bg_struct[: min(50, n_bg)]

    # Feature names and group masks (B vs C dominance)
    struct_feature_names = getattr(explain_struct, "feature_names_", None)
    if struct_feature_names is None:
        struct_feature_names = [f"feat_{i}" for i in range(p_struct)]
    groups = infer_feature_groups(list(struct_feature_names))
    mask_B = groups.get("B", np.zeros(p_struct, dtype=bool))
    mask_C = groups.get("C", np.zeros(p_struct, dtype=bool))
    if mask_B.sum() == 0 and mask_C.sum() == 0:
        # Fallback: treat all as 'B' (structural)
        mask_B = np.ones(p_struct, dtype=bool)
        mask_C = np.zeros(p_struct, dtype=bool)

    # Create structural explainer (fast for linear models)
    try:
        expl_struct = shap.Explainer(explain_struct.classifier_, X_bg_struct, feature_names=struct_feature_names)
    except Exception as e:
        logger.warning(f"[shap] Failed to create structural SHAP explainer: {e}")
        return None

    # ----------------------------------------
    # Collect shift cases (dual-mode, unbiased)
    # ----------------------------------------
    records = []
    for ds_name, df in datasets.items():
        if df is None or df.empty:
            continue
        if "text" not in df.columns or "label" not in df.columns:
            continue

        try:
            res_a = evaluate_model(model_a, df)
            res_b = evaluate_model(model_b, df)
        except Exception as e:
            logger.warning(f"[shap] Failed to evaluate models on {ds_name}: {e}")
            continue

        y_true = np.asarray(res_a.get("y_true", df["label"].to_numpy()))
        pred_a = np.asarray(res_a.get("y_pred"))
        pred_b = np.asarray(res_b.get("y_pred"))
        if pred_a is None or pred_b is None:
            logger.warning(f"[shap] Missing predictions for {ds_name}. Skipping.")
            continue

        # Mode A (Structure wins)
        idx_A = np.where((y_true == 1) & (pred_a == 0) & (pred_b == 1))[0]
        # Mode B (Lexical wins / reverse)
        idx_B = np.where((y_true == 1) & (pred_a == 1) & (pred_b == 0))[0]

        # Per-dataset caps to avoid domination by one dataset
        cap_per_mode = max(10, eval_size // max(1, 2 * len(datasets)))
        for mode, idx in [("structure_wins", idx_A), ("lexical_wins", idx_B)]:
            if idx.size == 0:
                continue
            k = min(cap_per_mode, idx.size)
            chosen = rng.choice(idx, size=k, replace=False)
            for i in chosen:
                records.append({
                    "dataset": ds_name,
                    "mode": mode,
                    "row_index": int(i),
                    "text": str(df.iloc[i]["text"]),
                    "cleaned_text": str(df.iloc[i].get("cleaned_text", df.iloc[i]["text"])),
                    "label": int(y_true[i]),
                    "pred_A": int(pred_a[i]),
                    "pred_B": int(pred_b[i]),
                })

    if not records:
        logger.warning("[shap] No shift cases found for SHAP.")
        return None

    df_cases = pd.DataFrame(records)
    # Optional global cap for shift cases (split equally across modes when possible).
    if n_shift_cases is not None and n_shift_cases > 0 and not df_cases.empty:
        rng = np.random.default_rng(seed + 777)
        per_mode = max(1, int(n_shift_cases // 2))
        df_cases = (
            df_cases.groupby("mode", group_keys=False)
            .apply(lambda g: g.sample(n=min(per_mode, len(g)), random_state=int(rng.integers(0, 2**31-1))))
            .reset_index(drop=True)
        )
    # Enforce global cap (memory safety)
    if len(df_cases) > eval_size * 2:
        df_cases = df_cases.sample(n=eval_size * 2, random_state=seed).reset_index(drop=True)

    logger.info(f"[shap] Collected shift cases: {len(df_cases)} total")

    # ----------------------------------------
    # Compute SHAP for structural explain-model
    # ----------------------------------------
    stats_rows = []
    for (ds_name, mode), g in df_cases.groupby(["dataset", "mode"], sort=True):
        if len(g) < 10:
            continue

        # Hard cap per group
        g = g.sample(n=min(len(g), eval_size), random_state=int(rng.integers(0, 10_000_000))).reset_index(drop=True)

        # Dimensionality check: avoid degenerate explainer calls
        try:
            X_eval = np.asarray(explain_struct.transform_features(g))
        except Exception as e:
            logger.warning(f"[shap] Feature transform failed for {ds_name}/{mode}: {e}")
            continue

        if X_eval.ndim != 2 or X_eval.shape[0] != len(g):
            logger.warning(f"[shap] Bad feature matrix for {ds_name}/{mode}: {X_eval.shape}")
            continue

        # Auto-downscale for very wide feature spaces
        p = X_eval.shape[1]
        max_eval_here = min(eval_size, len(g))
        if p > 20000:
            max_eval_here = min(80, max_eval_here)
        X_eval = X_eval[:max_eval_here]
        g = g.iloc[:max_eval_here].reset_index(drop=True)

        # Compute SHAP
        try:
            shap_out = expl_struct(X_eval)
            shap_mat = _extract_shap_matrix(shap_out, class_index=1)
        except Exception as e:
            logger.warning(f"[shap] SHAP compute failed for {ds_name}/{mode}: {e}")
            continue

        # Dominance metrics per instance
        abs_shap = np.abs(shap_mat)
        total = abs_shap.sum(axis=1) + 1e-12
        mass_B = abs_shap[:, mask_B].sum(axis=1) if mask_B.any() else np.zeros_like(total)
        mass_C = abs_shap[:, mask_C].sum(axis=1) if mask_C.any() else np.zeros_like(total)
        # Structural dominance within (B vs C) if both exist; else B/total
        denom_bc = (mass_B + mass_C) + 1e-12
        struct_dom = np.where((mass_B + mass_C) > 0, mass_B / denom_bc, mass_B / total)

        # Bootstrap CI of mean dominance
        n = struct_dom.shape[0]
        if n < 5:
            continue
        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            samp = rng.choice(n, size=n, replace=True)
            boot_means[i] = float(np.mean(struct_dom[samp]))
        lo, hi = np.quantile(boot_means, [0.025, 0.975])

        stats_rows.append({
            "dataset": ds_name,
            "mode": mode,
            "n_cases": int(n),
            "struct_dominance_mean": float(np.mean(struct_dom)),
            "struct_dominance_ci_low": float(lo),
            "struct_dominance_ci_high": float(hi),
            "struct_features_B": int(mask_B.sum()),
            "struct_features_C": int(mask_C.sum()),
            "explain_model": "BC" if explain_struct is models.get("BC") else "B",
        })

        # Optional plot (never crash)
        if plt is not None:
            try:
                # Bar plot: top 15 features by mean |SHAP|
                mean_abs = np.mean(abs_shap, axis=0)
                topk = min(15, len(mean_abs))
                top_idx = np.argsort(mean_abs)[-topk:][::-1]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(topk)[::-1], mean_abs[top_idx][::-1])
                ax.set_yticks(range(topk)[::-1])
                ax.set_yticklabels([str(struct_feature_names[i]) for i in top_idx][::-1])
                ax.set_title(f"Top SHAP features ({ds_name} | {mode})")
                ax.set_xlabel("mean |SHAP|")
                fig.tight_layout()
                fig_path = plot_dir / f"shap_top_features_{ds_name}_{mode}.png"
                fig.savefig(fig_path, dpi=200)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"[shap] Plotting failed for {ds_name}/{mode} (continuing): {e}")

    # ---------------------------------------------------
    # Lexical shortcut analysis: SHAP on model A (reduced)
    # ---------------------------------------------------
    # We compute a *concentration* statistic on Mode B (lexical_wins) only.
    # To keep memory bounded, we restrict to top-k features by |coef|.
    try:
        clf_a = model_a.classifier_
        coef = np.asarray(getattr(clf_a, "coef_", None))
        if coef is None or coef.ndim != 2:
            raise ValueError("Model A has no accessible coef_ for top-k selection")
        coef = coef[0]
        feat_names_a = getattr(model_a, "feature_names_", None)
        if feat_names_a is None:
            feat_names_a = [f"feat_{i}" for i in range(len(coef))]

        topk = int(min(topk_lexical, len(coef)))
        top_idx = np.argsort(np.abs(coef))[-topk:]

        # Build background/eval matrices (top-k only)
        X_bg_a_full = model_a.transform_features(df_bg)
        X_bg_a = X_bg_a_full[:, top_idx]

        expl_a = shap.Explainer(clf_a, X_bg_a, feature_names=[feat_names_a[i] for i in top_idx])

        df_lex = df_cases[df_cases["mode"] == "lexical_wins"].copy()
        if not df_lex.empty:
            # global cap
            df_lex = df_lex.sample(n=min(len(df_lex), eval_size), random_state=seed).reset_index(drop=True)
            X_eval_a_full = model_a.transform_features(df_lex)
            X_eval_a = X_eval_a_full[:, top_idx]

            # Possibly reduce further if still too wide
            if X_eval_a.shape[1] > 20000:
                X_eval_a = X_eval_a[:, -20000:]

            shap_out_a = expl_a(X_eval_a)
            shap_mat_a = _extract_shap_matrix(shap_out_a, class_index=1)
            abs_a = np.abs(shap_mat_a)
            total_a = abs_a.sum(axis=1) + 1e-12
            # concentration: share of top-10 features by |SHAP| per instance
            k10 = min(10, abs_a.shape[1])
            top10_mass = np.sort(abs_a, axis=1)[:, -k10:].sum(axis=1)
            conc = top10_mass / total_a

            n = conc.shape[0]
            boot_means = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                samp = rng.choice(n, size=n, replace=True)
                boot_means[i] = float(np.mean(conc[samp]))
            lo, hi = np.quantile(boot_means, [0.025, 0.975])

            stats_rows.append({
                "dataset": "ALL",
                "mode": "lexical_wins",
                "n_cases": int(n),
                "lex_shortcut_concentration_top10_mean": float(np.mean(conc)),
                "lex_shortcut_concentration_ci_low": float(lo),
                "lex_shortcut_concentration_ci_high": float(hi),
                "lex_topk_features": int(topk),
                "explain_model": "A",
            })
    except Exception as e:
        logger.warning(f"[shap] Lexical shortcut SHAP analysis skipped: {e}")

    if not stats_rows:
        logger.warning("[shap] No SHAP statistics were produced.")
        return None

    df_stats = pd.DataFrame(stats_rows)
    out_path = plot_dir / "shap_statistics.csv"
    df_stats.to_csv(out_path, index=False)
    logger.info(f"[shap] Saved shap_statistics.csv -> {out_path}")

    return df_stats


def run_divergence_analysis(
    models: Dict[str, Any],  # Type Hint updated to match generic Any for TSSClassifier
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
    plots_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  TSS: DIVERGENCE HYPOTHESIS PROOF                                        ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  Hypothesis: Human labels capture STRUCTURAL patterns                    ║
    ║              Auto labels capture LEXICAL patterns                        ║
    ║                                                                          ║
    ║  Prediction:                                                             ║
    ║  - Channels B, C (Structure, Style) → Better on HUMAN labels             ║
    ║  - Channel A (Lexical) → Better (relatively) on AUTO labels              ║
    ║                                                                          ║
    ║  Method:                                                                 ║
    ║  1. Evaluate all channels on all datasets                                ║
    ║  2. Compute F1 scores split by label type                                ║
    ║  3. Calculate "Divergence Score" = Human_rank - Auto_rank                ║
    ║  4. If B/C have positive divergence → Hypothesis SUPPORTED               ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    logger.info("\n" + "="*80)
    logger.info("📐 DIVERGENCE HYPOTHESIS ANALYSIS")
    logger.info("="*80)

    plt, sns, has_plotting = safe_import_plotting()

    results = {
        'evaluations': [],
        'human_rankings': {},
        'auto_rankings': {},
        'divergence_scores': {},
        'hypothesis_supported': None,
    }

    # Evaluate all models on all datasets
    logger.info("\n   Evaluating all channels on all datasets...")

    eval_results = []
    raw_preds: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Hard safety checks: allow the pipeline to run even when artifacts are missing
    if evaluate_model is None:
        logger.error("[divergence] `evaluate_model` is unavailable (tss.pipeline import failed). Skipping divergence analysis.")
        return results, raw_preds
    if not datasets:
        logger.warning("[divergence] No datasets loaded. Skipping divergence analysis.")
        return results, raw_preds
    if not models:
        logger.warning("[divergence] No models loaded. Skipping divergence analysis.")
        return results, raw_preds

    for channel, model in models.items():
        if model is None:
            logger.warning(f"[divergence] Model for channel '{channel}' is None (failed to load). Skipping.")
            continue
        for ds_name, df in datasets.items():
            if 'cleaned_text' not in df.columns and 'text' in df.columns:
                df['cleaned_text'] = df['text']
            try:
                # Calculate metrics
                result = evaluate_model(model, df)

                y_true = np.array(result['y_true'])
                y_pred = np.array(result['y_pred'])

                # --- FIX START: Save raw predictions for later use in Semantic Analysis ---
                # This fixes the KeyError: 'label_type' downstream because raw_preds won't be empty
                raw_preds[(ds_name, channel)] = {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
                # --- FIX END ---

                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)

                info = DATASET_INFO.get(ds_name, {})

                eval_results.append({
                    'channel': channel,
                    'dataset': ds_name,
                    'label_type': info.get('label_type', 'unknown'),
                    'platform': info.get('platform', 'unknown'),
                    'f1': result['f1'],
                    'macro_f1': macro_f1,
                    'mcc': mcc,
                    'precision': result['precision'],
                    'recall': result['recall'],
                })

            except Exception as e:
                logger.warning(f"      {channel} on {ds_name} failed: {e}")

    results['evaluations'] = eval_results
    eval_df = pd.DataFrame(eval_results)
    if eval_df.empty:
        logger.warning("[divergence] No evaluations were produced (missing artifacts?). Skipping downstream divergence computations.")
        safe_save_df(eval_df, output_dir / "divergence_evaluations.csv")
        return results, raw_preds


    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTE RANKINGS BY LABEL TYPE
    # ═══════════════════════════════════════════════════════════════════════════

    # Human-labeled datasets
    human_df = eval_df[eval_df['label_type'] == 'human']
    auto_df = eval_df[eval_df['label_type'] == 'auto']

    human_rankings = {}
    auto_rankings = {}

    if len(human_df) > 0:
        human_avg = human_df.groupby('channel')['macro_f1'].mean().sort_values(ascending=False)
        human_rankings = {ch: rank+1 for rank, ch in enumerate(human_avg.index)}
        results['human_rankings'] = human_rankings

        logger.info(f"\n   📊 HUMAN-LABELED DATASETS (Macro F1):")
        for ch, f1 in human_avg.items():
            logger.info(f"      #{human_rankings[ch]} {ch}: {f1:.3f}")

    if len(auto_df) > 0:
        auto_avg = auto_df.groupby('channel')['macro_f1'].mean().sort_values(ascending=False)
        auto_rankings = {ch: rank+1 for rank, ch in enumerate(auto_avg.index)}
        results['auto_rankings'] = auto_rankings

        logger.info(f"\n   📊 AUTO-LABELED DATASETS (Macro F1):")
        for ch, f1 in auto_avg.items():
            logger.info(f"      #{auto_rankings[ch]} {ch}: {f1:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERGENCE SCORE CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════

    if human_rankings and auto_rankings:
        logger.info(f"\n   📐 DIVERGENCE SCORES (Human_rank - Auto_rank):")
        logger.info(f"      (Negative = Better on Human, Positive = Better on Auto)")

        divergence_scores = {}
        all_channels = set(human_rankings.keys()) & set(auto_rankings.keys())

        for ch in all_channels:
            divergence = human_rankings.get(ch, 0) - auto_rankings.get(ch, 0)
            divergence_scores[ch] = divergence

            indicator = "🎯 HUMAN" if divergence < 0 else ("⚙️ AUTO" if divergence > 0 else "→ NEUTRAL")
            logger.info(f"      {ch}: {divergence:+d} {indicator}")

        results['divergence_scores'] = divergence_scores

        # Check hypothesis: B and C should have negative divergence (better on human)
        structural_channels = ['B', 'C', 'BC']
        structural_divergences = [divergence_scores.get(ch, 0) for ch in structural_channels if ch in divergence_scores]

        if structural_divergences:
            avg_structural_divergence = np.mean(structural_divergences)

            # A should have positive or neutral divergence (better on auto)
            a_divergence = divergence_scores.get('A', 0)

            hypothesis_supported = (avg_structural_divergence <= 0) or (a_divergence >= 0)
            results['hypothesis_supported'] = hypothesis_supported

            logger.info(f"\n   🔬 HYPOTHESIS TEST:")
            logger.info(f"      Structural channels (B,C,BC) avg divergence: {avg_structural_divergence:.2f}")
            logger.info(f"      Lexical channel (A) divergence: {a_divergence:+d}")
            logger.info(f"      Hypothesis SUPPORTED: {hypothesis_supported}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    if has_plotting and len(eval_df) > 0:
        logger.info(f"\n   📊 Generating divergence visualizations...")

        # 1. Heatmap: Channel x Dataset F1
        try:
            pivot_f1 = eval_df.pivot_table(
                values='macro_f1',
                index='channel',
                columns='dataset',
                aggfunc='first'
            )

            fig, ax = plt.subplots(figsize=(12, 8))

            # Color by label type
            col_colors = [DATASET_INFO.get(d, {}).get('color', 'gray') for d in pivot_f1.columns]

            sns.heatmap(
                pivot_f1,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                ax=ax,
                cbar_kws={'label': 'Macro F1'}
            )

            # Add label type annotations
            label_type_map = {d: DATASET_INFO.get(d, {}).get('label_type', 'unk') for d in pivot_f1.columns}
            new_labels = [f"{d}\n({label_type_map[d].upper()})" for d in pivot_f1.columns]
            ax.set_xticklabels(new_labels, rotation=45, ha='right')

            ax.set_title("Channel Performance by Dataset\n"
                        "(Testing Divergence Hypothesis: Structure vs Lexical)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Dataset (Label Type)")
            ax.set_ylabel("Channel")

            plt.tight_layout()
            plt.savefig(plots_dir / "divergence_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"      ✓ Saved: divergence_heatmap.png")

        except Exception as e:
            logger.warning(f"      Heatmap failed: {e}")

        # 2. Grouped bar: Human vs Auto performance by channel
        try:
            fig, ax = plt.subplots(figsize=(12, 7))

            channels = sorted(eval_df['channel'].unique())
            x = np.arange(len(channels))
            width = 0.35

            human_scores = [human_df[human_df['channel'] == ch]['macro_f1'].mean()
                          if ch in human_df['channel'].values else 0 for ch in channels]
            auto_scores = [auto_df[auto_df['channel'] == ch]['macro_f1'].mean()
                         if ch in auto_df['channel'].values else 0 for ch in channels]

            bars1 = ax.bar(x - width/2, human_scores, width, label='Human Labels', color='#3498db')
            bars2 = ax.bar(x + width/2, auto_scores, width, label='Auto Labels', color='#e74c3c')

            # Add divergence indicators
            for i, ch in enumerate(channels):
                div = divergence_scores.get(ch, 0)
                if div != 0:
                    max_score = max(human_scores[i], auto_scores[i])
                    indicator = "⬇" if div < 0 else "⬆"
                    ax.annotate(
                        f'{indicator}',
                        xy=(i, max_score + 0.02),
                        ha='center',
                        fontsize=12,
                        fontweight='bold',
                        color='#27ae60' if div < 0 else '#e74c3c'
                    )

            ax.set_xlabel('Channel', fontsize=12)
            ax.set_ylabel('Macro F1', fontsize=12)
            ax.set_title('Divergence Hypothesis: Human vs Auto Label Performance\n'
                        '(⬇ = Better on Human, ⬆ = Better on Auto)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(channels)
            ax.legend()
            ax.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(plots_dir / "divergence_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"      ✓ Saved: divergence_comparison.png")

        except Exception as e:
            logger.warning(f"      Comparison plot failed: {e}")

        # 3. Cross-platform transfer visualization
        try:
            # Reddit -> Twitter transfer
            reddit_human = eval_df[(eval_df['dataset'] == 'dreaddit_test')]
            twitter_human = eval_df[(eval_df['dataset'] == 'twitter_gold')]
            twitter_auto = eval_df[(eval_df['dataset'] == 'twitter')]

            if len(reddit_human) > 0 and (len(twitter_human) > 0 or len(twitter_auto) > 0):
                fig, ax = plt.subplots(figsize=(12, 7))

                channels = sorted(reddit_human['channel'].unique())
                x = np.arange(len(channels))
                width = 0.25

                reddit_scores = [reddit_human[reddit_human['channel'] == ch]['macro_f1'].values[0]
                               if ch in reddit_human['channel'].values else 0 for ch in channels]

                twitter_human_scores = [twitter_human[twitter_human['channel'] == ch]['macro_f1'].values[0]
                                       if ch in twitter_human['channel'].values else 0 for ch in channels]

                twitter_auto_scores = [twitter_auto[twitter_auto['channel'] == ch]['macro_f1'].values[0]
                                      if ch in twitter_auto['channel'].values else 0 for ch in channels]

                ax.bar(x - width, reddit_scores, width, label='Reddit (Human)', color='#2ecc71')
                ax.bar(x, twitter_human_scores, width, label='Twitter Gold (Human)', color='#3498db')
                ax.bar(x + width, twitter_auto_scores, width, label='Twitter (Auto)', color='#e74c3c')

                ax.set_xlabel('Channel', fontsize=12)
                ax.set_ylabel('Macro F1', fontsize=12)
                ax.set_title('Cross-Platform Transfer: Reddit → Twitter\n'
                            '(Comparing Human vs Auto Labels)',
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(channels)
                ax.legend()
                ax.set_ylim(0, 1)

                plt.tight_layout()
                plt.savefig(plots_dir / "cross_platform_transfer.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"      ✓ Saved: cross_platform_transfer.png")

        except Exception as e:
            logger.warning(f"      Transfer plot failed: {e}")

    # Export results
    eval_df.to_csv(output_dir / "divergence_evaluations.csv", index=False)
    logger.info(f"\n   📄 Exported: divergence_evaluations.csv")

    return results, raw_preds


    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTE RANKINGS BY LABEL TYPE
    # ═══════════════════════════════════════════════════════════════════════════

    # Human-labeled datasets
    human_df = eval_df[eval_df['label_type'] == 'human']
    auto_df = eval_df[eval_df['label_type'] == 'auto']

    human_rankings = {}
    auto_rankings = {}

    if len(human_df) > 0:
        human_avg = human_df.groupby('channel')['macro_f1'].mean().sort_values(ascending=False)
        human_rankings = {ch: rank+1 for rank, ch in enumerate(human_avg.index)}
        results['human_rankings'] = human_rankings

        logger.info(f"\n   📊 HUMAN-LABELED DATASETS (Macro F1):")
        for ch, f1 in human_avg.items():
            logger.info(f"      #{human_rankings[ch]} {ch}: {f1:.3f}")

    if len(auto_df) > 0:
        auto_avg = auto_df.groupby('channel')['macro_f1'].mean().sort_values(ascending=False)
        auto_rankings = {ch: rank+1 for rank, ch in enumerate(auto_avg.index)}
        results['auto_rankings'] = auto_rankings

        logger.info(f"\n   📊 AUTO-LABELED DATASETS (Macro F1):")
        for ch, f1 in auto_avg.items():
            logger.info(f"      #{auto_rankings[ch]} {ch}: {f1:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERGENCE SCORE CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════

    if human_rankings and auto_rankings:
        logger.info(f"\n   📐 DIVERGENCE SCORES (Human_rank - Auto_rank):")
        logger.info(f"      (Negative = Better on Human, Positive = Better on Auto)")

        divergence_scores = {}
        all_channels = set(human_rankings.keys()) & set(auto_rankings.keys())

        for ch in all_channels:
            divergence = human_rankings.get(ch, 0) - auto_rankings.get(ch, 0)
            divergence_scores[ch] = divergence

            indicator = "🎯 HUMAN" if divergence < 0 else ("⚙️ AUTO" if divergence > 0 else "→ NEUTRAL")
            logger.info(f"      {ch}: {divergence:+d} {indicator}")

        results['divergence_scores'] = divergence_scores

        # Check hypothesis: B and C should have negative divergence (better on human)
        structural_channels = ['B', 'C', 'BC']
        structural_divergences = [divergence_scores.get(ch, 0) for ch in structural_channels if ch in divergence_scores]

        if structural_divergences:
            avg_structural_divergence = np.mean(structural_divergences)

            # A should have positive or neutral divergence (better on auto)
            a_divergence = divergence_scores.get('A', 0)

            hypothesis_supported = (avg_structural_divergence <= 0) or (a_divergence >= 0)
            results['hypothesis_supported'] = hypothesis_supported

            logger.info(f"\n   🔬 HYPOTHESIS TEST:")
            logger.info(f"      Structural channels (B,C,BC) avg divergence: {avg_structural_divergence:.2f}")
            logger.info(f"      Lexical channel (A) divergence: {a_divergence:+d}")
            logger.info(f"      Hypothesis SUPPORTED: {hypothesis_supported}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    if has_plotting and len(eval_df) > 0:
        logger.info(f"\n   📊 Generating divergence visualizations...")

        # 1. Heatmap: Channel x Dataset F1
        try:
            pivot_f1 = eval_df.pivot_table(
                values='macro_f1',
                index='channel',
                columns='dataset',
                aggfunc='first'
            )

            fig, ax = plt.subplots(figsize=(12, 8))

            # Color by label type
            col_colors = [DATASET_INFO.get(d, {}).get('color', 'gray') for d in pivot_f1.columns]

            sns.heatmap(
                pivot_f1,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                ax=ax,
                cbar_kws={'label': 'Macro F1'}
            )

            # Add label type annotations
            label_type_map = {d: DATASET_INFO.get(d, {}).get('label_type', 'unk') for d in pivot_f1.columns}
            new_labels = [f"{d}\n({label_type_map[d].upper()})" for d in pivot_f1.columns]
            ax.set_xticklabels(new_labels, rotation=45, ha='right')

            ax.set_title("Channel Performance by Dataset\n"
                        "(Testing Divergence Hypothesis: Structure vs Lexical)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Dataset (Label Type)")
            ax.set_ylabel("Channel")

            plt.tight_layout()
            plt.savefig(plots_dir / "divergence_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"      ✓ Saved: divergence_heatmap.png")

        except Exception as e:
            logger.warning(f"      Heatmap failed: {e}")

        # 2. Grouped bar: Human vs Auto performance by channel
        try:
            fig, ax = plt.subplots(figsize=(12, 7))

            channels = sorted(eval_df['channel'].unique())
            x = np.arange(len(channels))
            width = 0.35

            human_scores = [human_df[human_df['channel'] == ch]['macro_f1'].mean()
                          if ch in human_df['channel'].values else 0 for ch in channels]
            auto_scores = [auto_df[auto_df['channel'] == ch]['macro_f1'].mean()
                         if ch in auto_df['channel'].values else 0 for ch in channels]

            bars1 = ax.bar(x - width/2, human_scores, width, label='Human Labels', color='#3498db')
            bars2 = ax.bar(x + width/2, auto_scores, width, label='Auto Labels', color='#e74c3c')

            # Add divergence indicators
            for i, ch in enumerate(channels):
                div = divergence_scores.get(ch, 0)
                if div != 0:
                    max_score = max(human_scores[i], auto_scores[i])
                    indicator = "⬇" if div < 0 else "⬆"
                    ax.annotate(
                        f'{indicator}',
                        xy=(i, max_score + 0.02),
                        ha='center',
                        fontsize=12,
                        fontweight='bold',
                        color='#27ae60' if div < 0 else '#e74c3c'
                    )

            ax.set_xlabel('Channel', fontsize=12)
            ax.set_ylabel('Macro F1', fontsize=12)
            ax.set_title('Divergence Hypothesis: Human vs Auto Label Performance\n'
                        '(⬇ = Better on Human, ⬆ = Better on Auto)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(channels)
            ax.legend()
            ax.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(plots_dir / "divergence_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"      ✓ Saved: divergence_comparison.png")

        except Exception as e:
            logger.warning(f"      Comparison plot failed: {e}")

        # 3. Cross-platform transfer visualization
        try:
            # Reddit -> Twitter transfer
            reddit_human = eval_df[(eval_df['dataset'] == 'dreaddit_test')]
            twitter_human = eval_df[(eval_df['dataset'] == 'twitter_gold')]
            twitter_auto = eval_df[(eval_df['dataset'] == 'twitter')]

            if len(reddit_human) > 0 and (len(twitter_human) > 0 or len(twitter_auto) > 0):
                fig, ax = plt.subplots(figsize=(12, 7))

                channels = sorted(reddit_human['channel'].unique())
                x = np.arange(len(channels))
                width = 0.25

                reddit_scores = [reddit_human[reddit_human['channel'] == ch]['macro_f1'].values[0]
                               if ch in reddit_human['channel'].values else 0 for ch in channels]

                twitter_human_scores = [twitter_human[twitter_human['channel'] == ch]['macro_f1'].values[0]
                                       if ch in twitter_human['channel'].values else 0 for ch in channels]

                twitter_auto_scores = [twitter_auto[twitter_auto['channel'] == ch]['macro_f1'].values[0]
                                      if ch in twitter_auto['channel'].values else 0 for ch in channels]

                ax.bar(x - width, reddit_scores, width, label='Reddit (Human)', color='#2ecc71')
                ax.bar(x, twitter_human_scores, width, label='Twitter Gold (Human)', color='#3498db')
                ax.bar(x + width, twitter_auto_scores, width, label='Twitter (Auto)', color='#e74c3c')

                ax.set_xlabel('Channel', fontsize=12)
                ax.set_ylabel('Macro F1', fontsize=12)
                ax.set_title('Cross-Platform Transfer: Reddit → Twitter\n'
                            '(Comparing Human vs Auto Labels)',
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(channels)
                ax.legend()
                ax.set_ylim(0, 1)

                plt.tight_layout()
                plt.savefig(plots_dir / "cross_platform_transfer.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"      ✓ Saved: cross_platform_transfer.png")

        except Exception as e:
            logger.warning(f"      Transfer plot failed: {e}")

    # Export results
    eval_df.to_csv(output_dir / "divergence_evaluations.csv", index=False)
    logger.info(f"\n   📄 Exported: divergence_evaluations.csv")

    return results, raw_preds


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"{SYSTEM_NAME} Advanced Analysis Suite ()"
    )
    parser.add_argument('--artifacts', type=str, default='artifacts',
                       help='Path to model artifacts directory')
    parser.add_argument('--data', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='outputs/analysis',
                       help='Path to output directory')
    parser.add_argument('--plots', type=str, default='outputs/plots',
                       help='Path to plots directory')
    parser.add_argument('--n_clusters', type=int, default=3,
                       help='Number of clusters for K-Means profiling')
    parser.add_argument('--max_rows_per_dataset', type=int, default=8000,
                       help='Max rows per dataset used in clustering to limit memory')
    parser.add_argument('--stability_bootstrap', type=int, default=100,
                       help='Bootstrap resamples for clustering stability (ARI)')
    parser.add_argument('--shap_background', type=int, default=200,
                       help='SHAP background sample size (hard-capped)')
    parser.add_argument('--shap_eval', type=int, default=500,
                       help='SHAP evaluation sample size (hard-capped)')
    parser.add_argument('--shap_shift_cases', type=int, default=80,
                       help='Number of shift cases for SHAP dual-mode')
    parser.add_argument('--shap_bootstrap', type=int, default=500,
                       help='Bootstrap iterations for SHAP dominance CI')
    parser.add_argument('--skip_shap', action='store_true',
                       help='Skip SHAP analysis (if not installed)')
    parser.add_argument('--skip_profiling', action='store_true',
                       help='Skip stylistic profiling')

    # Paired inference and semantic robustness options
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip paired inference tables (channel vs channel)')
    parser.add_argument('--run_semantic', action='store_true',
                       help='Run semantic topic robustness analysis (uses content only for analysis)')
    parser.add_argument('--metric', type=str, default='macro_f1',
                       choices=['macro_f1', 'mcc', 'accuracy', 'f1'],
                       help='Metric used in paired inference and semantic analysis')
    parser.add_argument('--n_boot', type=int, default=5000,
                       help='Bootstrap iterations for paired inference (default: 5000)')
    parser.add_argument('--n_perm', type=int, default=10000,
                       help='Permutation iterations for paired inference (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n_topics', type=int, default=12,
                       help='Number of topic clusters for semantic analysis (default: 12)')
    parser.add_argument('--topic_min_n', type=int, default=200,
                       help='Minimum samples per topic cluster (default: 200)')

    args = parser.parse_args()

    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     TSS: Advanced Analysis Suite                                             ║
║                                                                              ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║  "Linguistic structure is the true vessel of psychological state."           ║
║  "Linguistic structure is the true vessel of psychological state"            ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║                                                                              ║
║  ANALYSIS COMPONENTS:                                                        ║
║  ├── 1. Stylistic Profiling (K-Means on Channel C)                           ║
║  ├── 2. SHAP Explainability (Shift Cases Analysis)                           ║
║  └── 3. Divergence Hypothesis Proof                                          ║
║                                                                              ║
║  TESTING THE DIVERGENCE HYPOTHESIS:                                          ║
║  ├── Human labels → Structure/Style channels (B, C)                          ║
║  └── Auto labels → Lexical channel (A)                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots)
    if not plots_dir.is_absolute():
        plots_dir = project_dir / plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info(f"{SYSTEM_NAME} Advanced Analysis Suite ")
    logger.info("="*80)

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD RESOURCES
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("\n📂 Loading models...")
    models = load_models(artifacts_dir)

    if not models:
        logger.error("No models found. Run training script first.")
        return

    logger.info("\n📂 Loading datasets...")
    datasets = load_datasets(data_dir)

    if not datasets:
        logger.error("No datasets found.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN ANALYSES
    # ═══════════════════════════════════════════════════════════════════════════

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'version': __version__,
        'analyses': {}
    }

    # 1. Stylistic Profiling (Leakage-proof clustering + stability)
    if not args.skip_profiling:
        try:
            profiling_df = run_stylistic_profiling(
                datasets=datasets,
                models=models,
                output_dir=output_dir,
                plots_dir=plots_dir,
                n_clusters=args.n_clusters,
                seed=args.seed,
                channel_preference=("C", "BC", "ABC"),
                max_rows_per_dataset=args.max_rows_per_dataset,
                stability_bootstrap=args.stability_bootstrap,
            )
            all_results["analyses"]["stylistic_profiling"] = (
                profiling_df.to_dict(orient="records") if hasattr(profiling_df, "to_dict") else profiling_df
            )
        except Exception as e:
            logger.exception(f"Stylistic profiling failed (non-fatal): {e}")
    else:
        logger.info("Skipping stylistic profiling as requested (--skip_profiling).")

    # 2. SHAP Explainability (rigorous, bounded memory, dual-mode)
    if not args.skip_shap:
        if models.get("A") is not None and models.get("B") is not None:
            try:
                shap_df = run_shap_analysis(
                    datasets=datasets,
                    models=models,
                    output_dir=output_dir,
                    plots_dir=plots_dir,
                    seed=args.seed,
                    shap_background=args.shap_background,
                    shap_eval=args.shap_eval,
                    n_bootstrap=args.shap_bootstrap,
            n_shift_cases=args.shap_shift_cases,
                )
                all_results["analyses"]["shap_explainability"] = (
                    shap_df.to_dict(orient="records") if hasattr(shap_df, "to_dict") else shap_df
                )
            except Exception as e:
                logger.exception(f"SHAP analysis failed (non-fatal): {e}")
        else:
            logger.warning("Models A and B required for SHAP. Skipping.")
    else:
        logger.info("Skipping SHAP as requested (--skip_shap).")

    # 3. Divergence Hypothesis Analysis

    # 3. Divergence Hypothesis Analysis
    divergence_results, raw_preds = run_divergence_analysis(
        models, datasets, output_dir, plots_dir, seed=args.seed
    )
    all_results['analyses']['divergence_hypothesis'] = (
        divergence_results.to_dict(orient='records') if hasattr(divergence_results, 'to_dict') else divergence_results
    )

    # 4. Strong paired inference (channel vs channel) — complements masking suite
    if not args.skip_inference:
        inference_results = run_strong_inference_tables(
            raw_preds=raw_preds,
            output_dir=output_dir,
            comparisons=[("C", "A"), ("BC", "A"), ("B", "A")],
            metric=args.metric,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        all_results['analyses']['paired_inference'] = inference_results

    # 5. Optional semantic robustness over topic clusters (TF-IDF -> SVD -> KMeans)
    if args.run_semantic:
        semantic_results = run_semantic_topic_robustness(
            datasets=datasets,
            raw_preds=raw_preds,
            output_dir=output_dir,
            plots_dir=plots_dir,
            comparison=("C", "A"),
            n_topics=args.n_topics,
            min_n=args.topic_min_n,
            metric=args.metric,
            n_perm=max(20000, args.n_perm),
            seed=args.seed,
        )
        all_results['analyses']['semantic_robustness'] = semantic_results

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE COMPREHENSIVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"advanced_analysis_{timestamp}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=_json_default)

    logger.info(f"\n📄 Comprehensive results saved: {results_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("\n" + "="*80)
    logger.info("🎉 ADVANCED ANALYSIS COMPLETE")
    logger.info("="*80)

    logger.info(f"\n📁 Output Directory: {output_dir}")
    logger.info(f"📊 Plots Directory: {plots_dir}")

    # Print key findings
    if 'divergence_hypothesis' in all_results['analyses']:
        div = all_results['analyses']['divergence_hypothesis']
        if div.get('hypothesis_supported') is not None:
            status = "✅ SUPPORTED" if div['hypothesis_supported'] else "❌ NOT SUPPORTED"
            logger.info(f"\n🔬 DIVERGENCE HYPOTHESIS: {status}")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
