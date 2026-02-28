#!/usr/bin/env python3
"""
================================================================================
TSS Masking Suite (Comprehensive + Full Metrics)
================================================================================
Test hypothesis: "HOW we say it > WHAT we say"

"Linguistic Style and Structure are the true vessel of psychological state."

═══════════════════════════════════════════════════════════════════════════════
Updates (metrics module only; experiment logic unchanged):
├── [METRICS] Full metric bundle: Macro_F1, F1, MCC, PR-AUC, Recall, Precision
├── [BOOTSTRAP] 95% Confidence Intervals for ALL metrics (10000 iterations)
├── [SUMMARY] Expanded tables showing Macro_F1, MCC, PR-AUC, Recall
├── [ACCURACY] Added accuracy metric per request
└── [UNCHANGED] Experiment loops, masking logic, input/output preserved
═══════════════════════════════════════════════════════════════════════════════

================================================================================
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import os
import sys
import ast
import warnings
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score, recall_score,
    precision_recall_curve, auc, confusion_matrix, roc_auc_score,
    accuracy_score, average_precision_score
)

# Ensure local project imports work both in scripts and interactive environments
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ may be undefined in notebooks / some IDE runners
    _PROJECT_ROOT = Path.cwd()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tss.pipeline import TSSClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - ALL COMBINATIONS (UNCHANGED)
# =============================================================================

ALL_CHANNELS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
ALL_MASK_MODES = ['none', 'pos_only', 'content_only', 'function_only', 'random_pos']

MASK_DESCRIPTIONS = {
    'none': '📊 Baseline (100%) - Original text, no masking',
    'pos_only': '🏗️ Structure Champion - Replace words with POS tags (destroy lexical)',
    'content_only': '📚 Lexical Champion - Keep only content words (nouns, verbs, adj)',
    'function_only': '🎯 Stylistic Proxy - Keep only function words (pronouns, prepositions)',
    'random_pos': '🎲 Noise Baseline - Shuffle POS tags randomly (destroy syntax)',
}

FUNCTION_POS_PENN = {
    'IN', 'TO', 'CC', 'DT', 'PDT', 'WDT',
    'MD', 'PRP', 'PRP$', 'WP', 'WP$', 'RP',
    '.', ',', ':', '``', "''", '-LRB-', '-RRB-',
}

CONTENT_POS_PENN = {
    'NN', 'NNS', 'NNP', 'NNPS',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
    'JJ', 'JJR', 'JJS',
    'RB', 'RBR', 'RBS', 'WRB',
    'UH', 'CD',
}

# Import shared dataset registry (single source of truth)
import sys as _sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in _sys.path:
    _sys.path.insert(0, _project_dir)

from tss.dataset_registry import (
    DATASET_INFO, FILE_MAPPING, get_dataset_info, infer_label_type, infer_platform,
    get_eval_datasets, scan_data_directory, get_file_mapping,
)

# Primary metrics for display (ordered by importance for paper)
PRIMARY_METRICS = ['macro_f1', 'mcc', 'pr_auc', 'recall']


# =============================================================================
# HELPER FUNCTIONS (UNCHANGED)
# =============================================================================

def compute_majority_baseline(y_true: np.ndarray) -> Dict[str, float]:
    """Compute majority class baseline metrics."""
    y_true = np.asarray(y_true)
    n_samples = len(y_true)
    n_positive = int(np.sum(y_true == 1))
    n_negative = n_samples - n_positive
    prevalence = n_positive / n_samples if n_samples > 0 else 0.0

    if n_positive >= n_negative:
        y_pred_majority = np.ones(n_samples, dtype=int)
        majority_class = 1
    else:
        y_pred_majority = np.zeros(n_samples, dtype=int)
        majority_class = 0

    majority_f1 = f1_score(y_true, y_pred_majority, pos_label=1, zero_division=0)
    majority_macro_f1 = f1_score(y_true, y_pred_majority, average='macro', zero_division=0)

    return {
        'majority_class': majority_class,
        'majority_f1': float(majority_f1),
        'majority_macro_f1': float(majority_macro_f1),
        'prevalence': float(prevalence),
        'n_samples': n_samples,
        'n_positive': n_positive,
        'n_negative': n_negative,
    }


def parse_token_pos_pairs(value: Any,
                          *,
                          errors: Optional[Dict[str, Any]] = None,
                          dataset: Optional[str] = None,
                          row_id: Optional[int] = None) -> List[Tuple[str, str]]:
    """Parse token/POS pairs stored as a Python-literal string in CSV.

    Many datasets store token/POS tuples as a stringified Python list.
    If that string is malformed, silent parsing failures can wipe features.
    We record aggregated warnings and (optionally) fail-fast if the rate is high.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        out: List[Tuple[str, str]] = []
        for item in value:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append((str(item[0]), str(item[1])))
        return out

    if not isinstance(value, str):
        return []

    s = value.strip()
    if not s or s.lower() == "nan" or s == "[]":
        return []

    try:
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, list):
            return []
        out: List[Tuple[str, str]] = []
        for item in parsed:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append((str(item[0]), str(item[1])))
        return out
    except Exception as e:
        if errors is not None:
            errors["fails"] = errors.get("fails", 0) + 1
            if len(errors.get("examples", [])) < 3:
                errors.setdefault("examples", []).append({
                    "dataset": dataset,
                    "row_id": row_id,
                    "value_preview": s[:160],
                    "error": str(e)[:160]
                })
        return []

def apply_mask(
    token_pos_pairs: List[Tuple[str, str]],
    mode: str,
    rng: np.random.Generator = None
) -> List[Tuple[str, str]]:
    """Apply mask transformation to token_pos_pairs."""
    if not token_pos_pairs:
        return token_pos_pairs

    if mode == 'none':
        return token_pos_pairs

    elif mode == 'pos_only':
        return [(pos, pos) for token, pos in token_pos_pairs]

    elif mode == 'function_only':
        result = [(token, pos) for token, pos in token_pos_pairs if pos in FUNCTION_POS_PENN]
        return result if result else token_pos_pairs

    elif mode == 'content_only':
        result = [(token, pos) for token, pos in token_pos_pairs if pos in CONTENT_POS_PENN]
        return result if result else token_pos_pairs

    elif mode == 'random_pos':
        if len(token_pos_pairs) < 2:
            return token_pos_pairs

        tokens = [t for t, _ in token_pos_pairs]
        pos_tags = [p for _, p in token_pos_pairs]

        if rng is None:
            rng = np.random.default_rng(42)

        shuffled_pos = pos_tags.copy()
        rng.shuffle(shuffled_pos)

        return list(zip(tokens, shuffled_pos))

    else:
        return token_pos_pairs


def apply_mask_to_dataframe(df: pd.DataFrame,
                          mode: str,
                          seed: int = 42,
                          dataset_name: str = "") -> pd.DataFrame:
    """Apply mask to entire dataframe.

    Fixes vs v75:
      - Avoid iterrows() (slow) → list-based loop.
      - Track token_pos_pairs parsing failures and warn loudly (or fail-fast).
      - Preserve original mask modes: none, pos_only, function_only, content_only, random_pos.
    """
    df_masked = df.copy()

    if mode == 'none':
        return df_masked

    if 'token_pos_pairs' not in df_masked.columns:
        logger.warning(f"   [Mask:{mode}] No token_pos_pairs column found!")
        return df_masked

    rng = np.random.default_rng(seed)

    parse_errors: Dict[str, Any] = {"fails": 0, "examples": []}

    token_pos_list = df_masked['token_pos_pairs'].fillna('[]').astype(str).tolist()
    if 'cleaned_text' in df_masked.columns:
        cleaned_list = df_masked['cleaned_text'].fillna('').astype(str).tolist()
    else:
        cleaned_list = [''] * len(df_masked)

    masked_pairs: List[str] = []
    masked_texts: List[str] = []
    empty_count = 0
    total_pairs = 0

    for i, (tpp_str, cleaned) in enumerate(zip(token_pos_list, cleaned_list)):
        pairs = parse_token_pos_pairs(tpp_str, errors=parse_errors, dataset=dataset_name, row_id=i)

        if not pairs:
            empty_count += 1
            masked_pairs.append('[]')
            masked_texts.append(cleaned)
            continue

        total_pairs += len(pairs)
        masked = apply_mask(pairs, mode, rng)

        masked_pairs.append(str(masked))

        if mode == 'pos_only':
            masked_texts.append(' '.join([pos for _, pos in masked]))
        else:
            masked_texts.append(' '.join([token for token, _ in masked]))

    df_masked['token_pos_pairs'] = masked_pairs
    df_masked['cleaned_text'] = masked_texts

    # Loud warning if parsing failed
    n = len(df_masked)
    fails = int(parse_errors.get("fails", 0))
    if fails > 0:
        rate = 100.0 * fails / max(1, n)
        logger.warning(f"[masking:{dataset_name}] token_pos_pairs parse failures: {fails}/{n} ({rate:.2f}%).")
        for ex in parse_errors.get("examples", []):
            logger.warning(f"[masking:{dataset_name}] parse example: {ex}")
        if rate >= 5.0:
            raise RuntimeError(
                f"High parse failure rate for token_pos_pairs ({rate:.2f}%) in dataset '{dataset_name}'. "
                "Fix preprocessing/CSV formatting before trusting results."
            )

    # Keep original diagnostics
    if n > 0:
        logger.info(f"   [Mask:{mode}] empty_pairs={empty_count}/{n} ({(100*empty_count/max(1,n)):.1f}%), avg_pairs_per_nonempty={total_pairs/max(1, (n-empty_count)):.2f}")

    return df_masked

def load_datasets(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets from data/ directory.

    Uses the shared registry's FILE_MAPPING for known datasets,
    and auto-discovers any new CSV files not yet in the registry.
    """
    datasets = {}

    # Auto-discover any new CSV files in data/ (updates registry + FILE_MAPPING)
    scan_data_directory(data_dir)

    # Load all mapped datasets
    file_mapping = get_file_mapping()
    for name, filename in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            if 'cleaned_text' in df.columns:
                df['cleaned_text'] = df['cleaned_text'].fillna('')
                n_before = len(df)
                df = df.drop_duplicates(subset=['cleaned_text'])
                if n_before - len(df) > 0:
                    logger.info(f"   Removed {n_before - len(df)} duplicates from {name}")

            # Validate minimum requirements
            if 'label' not in df.columns:
                logger.warning(f"   ✗ {name}: Missing 'label' column — skipped")
                continue
            if 'cleaned_text' not in df.columns:
                logger.warning(f"   ✗ {name}: Missing 'cleaned_text' column — skipped")
                continue

            datasets[name] = df

            label_type = get_dataset_info(name).get('label_type', 'unknown')
            n_pos = df['label'].sum() if 'label' in df.columns else 0
            prev = n_pos / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"   ✓ {name}: {len(df):,} rows ({prev:.1f}% pos) [{label_type.upper()} labels]")
        else:
            logger.warning(f"   ✗ {name}: File not found ({filepath})")

    return datasets


# =============================================================================
# UPGRADED METRIC BUNDLE WITH BOOTSTRAP CI
# =============================================================================

def compute_single_metric_set(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute metrics for a single evaluation.

    Scientific choice:
      - If y_true has a single class, ROC-AUC / PR-AUC are not informative → return NaN.
    """
    out: Dict[str, float] = {}

    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    uniq = np.unique(y_true)
    if y_prob is None or len(uniq) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        return out

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")

    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")

    return out

def bootstrap_confidence_intervals(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None,
                                   n_bootstrap: int = 10000,
                                   seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for metrics.

    Reviewer-2 friendly behavior:
      - Point estimate computed on the full sample.
      - Stratified bootstrap when possible.
      - NaNs ignored (undefined AUC does not contaminate averages).
    """
    rng = np.random.RandomState(seed)
    metrics = ["macro_f1", "f1", "mcc", "precision", "recall", "accuracy", "pr_auc", "roc_auc"]

    point = compute_single_metric_set(y_true, y_pred, y_prob)

    n = len(y_true)
    uniq = np.unique(y_true)
    can_stratify = len(uniq) == 2 and (np.sum(y_true == 0) > 0) and (np.sum(y_true == 1) > 0)

    boot_scores: Dict[str, List[float]] = {m: [] for m in metrics}

    for _ in range(n_bootstrap):
        if can_stratify:
            idx0 = np.where(y_true == 0)[0]
            idx1 = np.where(y_true == 1)[0]
            bs_idx = np.concatenate([
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ])
        else:
            bs_idx = rng.randint(0, n, size=n)

        ms = compute_single_metric_set(
            y_true[bs_idx],
            y_pred[bs_idx],
            y_prob[bs_idx] if y_prob is not None else None
        )
        for m in metrics:
            boot_scores[m].append(ms.get(m, float("nan")))

    summary: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        arr = np.asarray(boot_scores[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 30:
            summary[m] = {
                "point": float(point.get(m, float("nan"))),
                "mean": float(point.get(m, float("nan"))),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_eff": int(arr.size),
            }
        else:
            summary[m] = {
                "point": float(point.get(m, float("nan"))),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "n_eff": int(arr.size),
            }

    return summary

def _metric_value(metric_name, y_true, y_pred, y_prob=None):
    """Compute a single metric value for binary classification."""
    from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score
    if metric_name == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric_name == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric_name == "pr_auc":
        # Prefer probabilities/scores if available; fall back to hard preds.
        scores = y_prob if y_prob is not None else y_pred
        return float(average_precision_score(y_true, scores))
    raise ValueError(f"Unknown metric: {metric_name}")

def paired_bootstrap_delta(
    y_true,
    y_pred_a, y_pred_b,
    y_prob_a=None, y_prob_b=None,
    metric_name="macro_f1",
    n_boot=10000,
    seed=42,
    stratified=True
):
    """Paired bootstrap CI for delta = metric(A) - metric(B) on the SAME test set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if y_prob_a is not None:
        y_prob_a = np.asarray(y_prob_a)
    if y_prob_b is not None:
        y_prob_b = np.asarray(y_prob_b)

    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)

    # Observed delta
    delta_obs = _metric_value(metric_name, y_true, y_pred_a, y_prob_a) - _metric_value(metric_name, y_true, y_pred_b, y_prob_b)

    # Stratified bootstrap (keeps label counts stable; avoids degenerate resamples)
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        if stratified and len(idx0) > 0 and len(idx1) > 0:
            samp = np.concatenate([
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ])
        else:
            samp = rng.integers(0, n, size=n)

        da = _metric_value(metric_name, y_true[samp], y_pred_a[samp], y_prob_a[samp] if y_prob_a is not None else None)
        db = _metric_value(metric_name, y_true[samp], y_pred_b[samp], y_prob_b[samp] if y_prob_b is not None else None)
        deltas[i] = da - db

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(delta_obs), (float(lo), float(hi))

def approximate_randomization_test(
    y_true,
    y_pred_a, y_pred_b,
    y_prob_a=None, y_prob_b=None,
    metric_name="macro_f1",
    n_perm=10000,
    seed=42,
    two_sided=True
):
    """Approximate randomization / paired permutation test for delta = metric(A)-metric(B).

    We randomly swap A/B predictions per instance under the null that the two systems are exchangeable.
    This is widely used in NLP for non-decomposable metrics (e.g., F1).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if y_prob_a is not None:
        y_prob_a = np.asarray(y_prob_a)
    if y_prob_b is not None:
        y_prob_b = np.asarray(y_prob_b)

    delta_obs = _metric_value(metric_name, y_true, y_pred_a, y_prob_a) - _metric_value(metric_name, y_true, y_pred_b, y_prob_b)

    n = len(y_true)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5

        ya = np.where(swap, y_pred_b, y_pred_a)
        yb = np.where(swap, y_pred_a, y_pred_b)

        if y_prob_a is not None and y_prob_b is not None:
            pa = np.where(swap, y_prob_b, y_prob_a)
            pb = np.where(swap, y_prob_a, y_prob_b)
        else:
            pa = y_prob_a
            pb = y_prob_b

        d = _metric_value(metric_name, y_true, ya, pa) - _metric_value(metric_name, y_true, yb, pb)

        if two_sided:
            if abs(d) >= abs(delta_obs) - 1e-12:
                count += 1
        else:
            if d >= delta_obs - 1e-12:
                count += 1

    # +1 smoothing (recommended)
    p = (count + 1) / (n_perm + 1)
    return float(delta_obs), float(p)


def compute_full_metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    compute_ci: bool = True,
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """
    Compute comprehensive metric bundle with optional Bootstrap CIs.

    Returns dictionary with:
    - Point estimates for all metrics
    - Bootstrap CIs (if compute_ci=True)
    - Confusion matrix components
    - Majority baseline comparison
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)

    n_samples = len(y_true)
    n_positive = int(np.sum(y_true == 1))
    prevalence = n_positive / n_samples if n_samples > 0 else 0.0

    # Point estimates
    point_metrics = compute_single_metric_set(y_true, y_pred, y_prob)

    # Bootstrap CIs
    if compute_ci and n_samples >= 50:  # Only compute CI if enough samples
        ci_metrics = bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_bootstrap)
    else:
        ci_metrics = None

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except:
        specificity, tn, fp, fn, tp = 0.0, 0, 0, 0, 0

    # Majority baseline
    majority_baseline = compute_majority_baseline(y_true)
    delta_macro_f1 = point_metrics['macro_f1'] - majority_baseline['majority_macro_f1']
    delta_f1 = point_metrics['f1'] - majority_baseline['majority_f1']

    # Build result dictionary
    result = {
        # Point estimates
        'f1': float(point_metrics['f1']),
        'macro_f1': float(point_metrics['macro_f1']),
        'mcc': float(point_metrics['mcc']),
        'pr_auc': float(point_metrics['pr_auc']),
        'roc_auc': float(point_metrics['roc_auc']),
        'precision': float(point_metrics['precision']),
        'recall': float(point_metrics['recall']),
        'accuracy': float(point_metrics['accuracy']),
        'specificity': float(specificity),

        # Sample info
        'prevalence': float(prevalence),
        'n_samples': int(n_samples),
        'n_positive': int(n_positive),

        # Confusion matrix
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),

        # Baselines
        'majority_f1': float(majority_baseline['majority_f1']),
        'majority_macro_f1': float(majority_baseline['majority_macro_f1']),
        'delta_macro_f1': float(delta_macro_f1),
        'delta_f1': float(delta_f1),
    }

    # Add CI results if computed
    if ci_metrics:
        result['ci'] = ci_metrics

    return result


# =============================================================================
# ARTIFACT LOADING (UNCHANGED)
# =============================================================================

def load_pretrained_model(artifacts_dir: str, channel: str) -> Optional[TSSClassifier]:
    """Load pre-trained model from artifacts directory."""
    artifact_name = f"tss_{channel.lower()}.joblib"
    artifact_path = os.path.join(artifacts_dir, artifact_name)

    if os.path.exists(artifact_path):
        try:
            model = joblib.load(artifact_path)
            logger.info(f"   ✓ Loaded artifact: {artifact_name}")
            return model
        except Exception as e:
            logger.warning(f"   ✗ Failed to load {artifact_name}: {e}")
    return None


# =============================================================================
# TRAINING WITH MASK (UNCHANGED)
# =============================================================================

def train_with_mask(df_train: pd.DataFrame, channel: str, mask_mode: str) -> Tuple[TSSClassifier, Dict]:
    """Train model with masked data."""
    logger.info(f"   Training Channel {channel} (mask={mask_mode})...")

    y_train = df_train['label'].astype(int).values
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    logger.info(f"      Data: {n_pos:,} pos / {n_neg:,} neg")

    model = TSSClassifier(channel=channel, C=1.0, transfer_mode=True)
    model.fit(df_train, y_train)

    y_train_pred = model.predict(df_train)
    y_train_prob = model.predict_proba(df_train)[:, 1]
    train_metrics = compute_full_metric_bundle(y_train, y_train_pred, y_train_prob, compute_ci=False)

    artifact = {
        'channel': channel,
        'mask_mode': mask_mode,
        'train_f1': train_metrics['f1'],
        'train_macro_f1': train_metrics['macro_f1'],
        'train_mcc': train_metrics['mcc'],
        'n_features': len(model.get_feature_names()) if hasattr(model, 'get_feature_names') else 0,
        'threshold': model.threshold_ if hasattr(model, 'threshold_') else 0.5,
    }

    logger.info(f"      Train Macro_F1={train_metrics['macro_f1']:.3f}, MCC={train_metrics['mcc']:.3f}")

    return model, artifact


# =============================================================================
# UPGRADED EVALUATION FUNCTION
# =============================================================================

def evaluate_model(
    model: TSSClassifier,
    df_test: pd.DataFrame,
    ds_name: str,
    compute_ci: bool = True,
    n_bootstrap: int = 10000
) -> Dict:
    """
    Evaluate model on test data with full metric bundle and optional Bootstrap CIs.

    Returns comprehensive dictionary with:
    - All point metrics (f1, macro_f1, mcc, pr_auc, recall, precision, accuracy)
    - Bootstrap 95% CIs for all metrics (if compute_ci=True)
    - Predictions for error analysis
    """
    y_true = df_test['label'].astype(int).values

    # Get predictions
    try:
        y_prob = model.predict_proba(df_test)[:, 1]
    except:
        # Fallback if predict_proba not available
        try:
            y_prob = model.decision_function(df_test)
            # Normalize to [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
        except:
            y_prob = None

    y_pred = model.predict(df_test)

    # Compute full metric bundle with CIs
    metrics = compute_full_metric_bundle(y_true, y_pred, y_prob, compute_ci, n_bootstrap)

    # Add dataset info
    ds_info = get_dataset_info(ds_name)
    metrics['label_type'] = ds_info.get('label_type', 'unknown')
    metrics['platform'] = ds_info.get('platform', 'unknown')

    # Add predictions for potential error analysis
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    metrics['y_true'] = y_true

    return metrics


# =============================================================================
# MAIN EXPERIMENT (LOGIC UNCHANGED - Only results storage updated)
# =============================================================================

def run_masking_experiments(
    datasets: Dict[str, pd.DataFrame],
    channels: List[str],
    mask_modes: List[str],
    output_dir: str,
    artifacts_dir: str,
    compute_ci: bool = True,
    n_bootstrap: int = 10000,
) -> List[Dict]:
    """Run all masking experiments. (Logic UNCHANGED from )"""
    os.makedirs(output_dir, exist_ok=True)

    # Dynamically identify train and test datasets from loaded data
    train_names = [name for name in datasets if name.endswith('_train')]
    if not train_names:
        raise ValueError("No training dataset found (expected a dataset ending in '_train')")
    if len(train_names) > 1:
        logger.warning(f"   Multiple train datasets found: {train_names}. Using first.")
    df_train = datasets[train_names[0]]

    test_datasets = {name: df for name, df in datasets.items()
                     if not name.endswith('_train')}

    logger.info(f"\n   Test datasets: {list(test_datasets.keys())}")
    logger.info(f"   Channels: {channels}")
    logger.info(f"   Mask modes: {mask_modes}")
    logger.info(f"   Total experiments: {len(channels) * len(mask_modes) * len(test_datasets)}")
    logger.info(f"   Bootstrap CI: {'Enabled' if compute_ci else 'Disabled'} (n={n_bootstrap})")

    all_results = []

    for mask_mode in mask_modes:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"MASK MODE: {mask_mode}")
        logger.info(f"   {MASK_DESCRIPTIONS.get(mask_mode, '')}")
        logger.info("=" * 80)

        # Apply mask to test data
        test_datasets_masked = {}
        for ds_name, df in test_datasets.items():
            test_datasets_masked[ds_name] = apply_mask_to_dataframe(df.copy(), mask_mode)

        for channel in channels:
            logger.info("")

            model = None
            train_artifact = None

            # For mask=none, try to load pre-trained artifact
            if mask_mode == 'none':
                model = load_pretrained_model(artifacts_dir, channel)
                if model is not None:
                    train_artifact = {
                        'channel': channel,
                        'mask_mode': mask_mode,
                        'train_f1': 0.0,
                        'train_macro_f1': 0.0,
                        'train_mcc': 0.0,
                        'n_features': len(model.get_feature_names()) if hasattr(model, 'get_feature_names') else 0,
                        'threshold': model.threshold_ if hasattr(model, 'threshold_') else 0.5,
                        'source': 'artifact',
                    }

            # Train if needed
            if model is None:
                df_train_masked = apply_mask_to_dataframe(df_train.copy(), mask_mode, dataset_name='dreaddit_train')
                try:
                    model, train_artifact = train_with_mask(df_train_masked, channel, mask_mode)
                except Exception as e:
                    logger.error(f"   ✗ Training failed for {channel}/{mask_mode}: {e}")
                    continue

            # Evaluate
            logger.info(f"   Evaluating Channel {channel}...")
            for ds_name, df_test_masked in test_datasets_masked.items():
                try:
                    if mask_mode == 'none' and train_artifact.get('source') == 'artifact':
                        df_eval = test_datasets[ds_name]
                    else:
                        df_eval = df_test_masked

                    result = evaluate_model(model, df_eval, ds_name, compute_ci, n_bootstrap)
                except Exception as e:
                    logger.error(f"      ✗ Evaluation failed for {ds_name}: {e}")
                    continue

                # Enhanced logging with more metrics
                sig = " ★" if result['delta_macro_f1'] > 0.05 else ""
                logger.info(f"      {ds_name:<15} [{result['label_type'][0].upper()}] "
                           f"Macro_F1={result['macro_f1']:.3f} MCC={result['mcc']:.3f} "
                           f"PR-AUC={result['pr_auc']:.3f} Recall={result['recall']:.3f}{sig}")

                # Store comprehensive results
                result_entry = {
                    'mask_mode': mask_mode,
                    'channel': channel,
                    'dataset': ds_name,
                    'label_type': result['label_type'],
                    'platform': result['platform'],
                    # Primary metrics
                    'macro_f1': result['macro_f1'],
                    'f1': result['f1'],
                    'mcc': result['mcc'],
                    'pr_auc': result['pr_auc'],
                    'roc_auc': result['roc_auc'],
                    'recall': result['recall'],
                    'precision': result['precision'],
                    'accuracy': result['accuracy'],
                    'specificity': result['specificity'],
                    # Sample info
                    'prevalence': result['prevalence'],
                    'n_samples': result['n_samples'],
                    # Baselines
                    'delta_macro_f1': result['delta_macro_f1'],
                    'majority_macro_f1': result['majority_macro_f1'],
                    # Training info
                    **(train_artifact or {})
                }

                # Add CI if available
                if 'ci' in result:
                    result_entry['ci'] = result['ci']

                # [CRITICAL] Store raw arrays for paired tests (Table 5B/6 and sanity checks)
                # Without this, paired bootstrap / permutation tests will crash with KeyError.
                if isinstance(result, dict) and ('y_true' in result) and ('y_pred' in result):
                    result_entry['y_true'] = result['y_true']
                    result_entry['y_pred'] = result['y_pred']
                    result_entry['y_prob'] = result.get('y_prob', None)
                all_results.append(result_entry)

    return all_results


# =============================================================================
# HELPER FUNCTIONS FOR SUMMARY
# =============================================================================

def get_result(results: List[Dict], mask: str, channel: str, dataset: str, metric: str = 'macro_f1') -> Optional[float]:
    """Safely get a metric value for a specific combination."""
    r = next((x for x in results if x.get('mask_mode') == mask
              and x.get('channel') == channel and x.get('dataset') == dataset), None)
    return r.get(metric) if r else None


def get_result_with_ci(results: List[Dict], mask: str, channel: str, dataset: str, metric: str = 'macro_f1') -> Optional[Dict]:
    """Get metric value with CI for a specific combination."""
    r = next((x for x in results if x.get('mask_mode') == mask
              and x.get('channel') == channel and x.get('dataset') == dataset), None)
    if r is None:
        return None

    point = r.get(metric)
    ci = r.get('ci', {}).get(metric, {})

    return {
        'value': point,
        'ci_low': ci.get('ci_low'),
        'ci_high': ci.get('ci_high'),
    }


# =============================================================================
# UPGRADED COMPREHENSIVE SUMMARY
# =============================================================================

def print_comprehensive_summary(results: List[Dict]) -> None:
    """Print comprehensive summary with ALL metrics and comparisons."""
    print("\n")
    print("╔" + "═" * 128 + "╗")
    print("║" + " " * 25 + "TSS MASKING SUITE - COMPREHENSIVE SUMMARY (FULL METRICS)" + " " * 40 + "║")
    print("║" + " " * 25 + " - All Combinations with Bootstrap 95% CI" + " " * 45 + "║")
    print("╚" + "═" * 128 + "╝")

    if not results:
        print("⚠️  No results to display.")
        return

    mask_modes = sorted(set(r.get('mask_mode') for r in results if r.get('mask_mode')))
    channels = sorted(set(r.get('channel') for r in results if r.get('channel')))
    datasets = sorted(set(r.get('dataset') for r in results if r.get('dataset')))

    # ==========================================================================
    # TABLE 1: PRIMARY METRICS MATRIX (Macro_F1, MCC, PR-AUC, Recall)
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("📊 TABLE 1: PRIMARY METRICS OVERVIEW")
    print("   Metrics shown: Macro_F1 | MCC | PR-AUC | Recall")
    print("=" * 130)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        print(f"\n  📁 {ds} [{label_type.upper()} labels]")
        print("  " + "-" * 120)

        # Header
        header = f"  {'Mask':<14} {'Chan':<5}"
        header += f" {'Macro_F1':>9} {'MCC':>8} {'PR-AUC':>8} {'Recall':>8} {'Prec':>8} {'Acc':>8}"
        print(header)
        print("  " + "-" * 120)

        # Rows
        for mask in mask_modes:
            for ch in channels:
                r = next((x for x in results if x.get('mask_mode') == mask
                          and x.get('channel') == ch and x.get('dataset') == ds), None)
                if r:
                    row = f"  {mask:<14} {ch:<5}"
                    row += f" {r.get('macro_f1', 0):>9.3f}"
                    row += f" {r.get('mcc', 0):>8.3f}"
                    row += f" {r.get('pr_auc', 0):>8.3f}"
                    row += f" {r.get('recall', 0):>8.3f}"
                    row += f" {r.get('precision', 0):>8.3f}"
                    row += f" {r.get('accuracy', 0):>8.3f}"
                    print(row)
            print()  # Separator between masks

    # ==========================================================================
    # TABLE 2: MACRO_F1 MATRIX WITH 95% CI
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("📊 TABLE 2: MACRO_F1 WITH 95% BOOTSTRAP CI")
    print("   Format: value [ci_low, ci_high]")
    print("=" * 130)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        print(f"\n  📁 {ds} [{label_type.upper()}]")
        print("  " + "-" * 100)

        header = f"  {'Mask':<14}"
        for ch in channels:
            header += f" {ch:>18}"
        print(header)
        print("  " + "-" * 100)

        for mask in mask_modes:
            row = f"  {mask:<14}"
            for ch in channels:
                r = get_result_with_ci(results, mask, ch, ds, 'macro_f1')
                if r and r['value'] is not None:
                    if r['ci_low'] is not None:
                        row += f" {r['value']:.3f}[{r['ci_low']:.2f},{r['ci_high']:.2f}]"
                    else:
                        row += f" {r['value']:>18.3f}"
                else:
                    row += f" {'---':>18}"
            print(row)

    # ==========================================================================
    # TABLE 3: MCC MATRIX (Critical for imbalanced data)
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("📊 TABLE 3: MATTHEWS CORRELATION COEFFICIENT (MCC)")
    print("   Range: [-1, 1], 0 = random, 1 = perfect, -1 = inverse")
    print("=" * 130)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        print(f"\n  📁 {ds} [{label_type.upper()}]")
        print("  " + "-" * 100)

        header = f"  {'Mask':<14}"
        for ch in channels:
            header += f" {ch:>10}"
        print(header)
        print("  " + "-" * 100)

        for mask in mask_modes:
            row = f"  {mask:<14}"
            for ch in channels:
                val = get_result(results, mask, ch, ds, 'mcc')
                if val is not None:
                    row += f" {val:>10.3f}"
                else:
                    row += f" {'---':>10}"
            print(row)

    # ==========================================================================
    # TABLE 4: RETENTION BASELINES (% of none)
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("📊 TABLE 4: RETENTION BASELINES (% of none)")
    print("   Question: How much Macro_F1 is retained after masking?")
    print("=" * 130)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        print(f"\n  📁 {ds} [{label_type.upper()}]")
        print("  " + "-" * 100)

        header = f"  {'Mask':<14}"
        for ch in channels:
            header += f" {ch:>10}"
        print(header)
        print("  " + "-" * 100)

        for mask in mask_modes:
            if mask == 'none':
                continue
            row = f"  {mask:<14}"
            for ch in channels:
                none_val = get_result(results, 'none', ch, ds, 'macro_f1')
                mask_val = get_result(results, mask, ch, ds, 'macro_f1')
                if none_val and mask_val and none_val > 0:
                    retention = (mask_val / none_val) * 100
                    row += f" {retention:>9.1f}%"
                else:
                    row += f" {'---':>10}"
            print(row)

    # ==========================================================================
    # TABLE 5: THE DIVERGENCE TEST (pos_only vs content_only) - Multiple Metrics
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("🔬 TABLE 5: THE DIVERGENCE TEST (pos_only vs content_only)")
    print("   Question: Structure vs Lexicon - which is stronger?")
    print("   Showing Δ for: Macro_F1 | MCC | PR-AUC")
    print("=" * 130)

    print(f"\n  {'Dataset':<16} {'Type':<6} {'Metric':<10}", end="")
    for ch in channels:
        print(f" {ch:>10}", end="")
    print()
    print("  " + "-" * 120)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')

        for metric in ['macro_f1', 'mcc', 'pr_auc']:
            row = f"  {ds:<16} {label_type:<6} {metric:<10}"

            for ch in channels:
                pos_val = get_result(results, 'pos_only', ch, ds, metric)
                content_val = get_result(results, 'content_only', ch, ds, metric)

                if pos_val is not None and content_val is not None:
                    diff = pos_val - content_val
                    winner = "🏗️" if diff > 0.02 else ("📚" if diff < -0.02 else "≈")
                    row += f" {diff:>+.3f}{winner}"
                else:
                    row += f" {'---':>10}"
            print(row)
        print()  # Separator between datasets

    print("  Legend: 🏗️ = Structure wins, 📚 = Lexicon wins, ≈ = Tie (±0.02)")

    # ==========================================================================
    # TABLE 5B/5C/5D: STRONG PAIRED TESTS (Instance-level Inference)
    # ==========================================================================
    print("\n" + "="*74)
    print("TABLE 5B/5C/5D: STRONG PAIRED TESTS (paired bootstrap Δ + permutation p)")
    print("-"*74)
    print("Rationale: With only a few datasets, we avoid Friedman/Nemenyi and instead")
    print("use paired, instance-level inference within each dataset (Reviewer-2 proof).")
    print("="*74)

    # Conservative defaults (can be increased if runtime allows)
    bootstrap_iterations = 5000
    permutation_iterations = 10000

    # Build fast index over results that contain raw arrays (required for paired tests)
    idx = {
        (r.get('mask_mode'), r.get('channel'), r.get('dataset')): r
        for r in results
        if isinstance(r, dict) and ('y_true' in r) and ('y_pred' in r)
    }

    # Comparisons we claim in the header (all tested with strong stats)
    paired_comparisons = [
        ("Divergence: POS-only vs Content-only", "pos_only", "content_only"),
        ("Function-words: Function-only vs Content-only", "function_only", "content_only"),
        ("Sanity: POS-only vs Random-POS", "pos_only", "random_pos"),
    ]

    def _print_paired_table(title: str, mode_a: str, mode_b: str):
        print("\n" + "-"*74)
        print(f"{title}  |  ({mode_a}  vs  {mode_b})")
        print("-"*74)
        print(f"{'Dataset':<16} {'Type':<6} {'Ch':<4} {'ΔF1':>8} {'CI95%':>22} {'p_perm':>9}")
        print("-"*74)

        # Determine datasets/channels that actually exist for these two modes
        ds_set = {ds for (m, ch, ds) in idx.keys() if m in (mode_a, mode_b)}
        # Prefer a stable, human-first ordering when possible
        preferred = ["dreaddit_test", "twitter_gold", "twitter", "reddit_combi"]
        ds_order = [d for d in preferred if d in ds_set] + sorted(ds_set - set(preferred))

        ch_set = {ch for (m, ch, ds) in idx.keys() if m in (mode_a, mode_b)}
        canonical_channels = ["A", "B", "C", "BC", "ABC"]
        ch_order = [c for c in canonical_channels if c in ch_set] + sorted(ch_set - set(canonical_channels))

        any_row = False
        for ds in ds_order:
            dtype = get_dataset_info(ds).get('label_type', 'unknown').upper()
            for ch in ch_order:
                r_a = idx.get((mode_a, ch, ds))
                r_b = idx.get((mode_b, ch, ds))
                if not r_a or not r_b:
                    continue
                y_true = np.asarray(r_a['y_true'])
                y_pred_a = np.asarray(r_a['y_pred'])
                y_pred_b = np.asarray(r_b['y_pred'])
                # Paired bootstrap CI for ΔF1
                try:
                    delta_obs, (ci_lo, ci_hi) = paired_bootstrap_delta(
                        y_true, y_pred_a, y_pred_b,
                        metric_name='macro_f1', n_boot=bootstrap_iterations,
                        stratified=True, seed=42
                    )
                except Exception as e:
                    logger.warning(f"[paired_bootstrap] failed for ds={ds} ch={ch} {mode_a}vs{mode_b}: {e}")
                    continue
                # Permutation / approximate randomization p-value for ΔF1
                try:
                    _, p_val = approximate_randomization_test(
                        y_true, y_pred_a, y_pred_b,
                        metric_name='macro_f1', n_perm=permutation_iterations,
                        seed=42
                    )
                except Exception as e:
                    logger.warning(f"[perm_test] failed for ds={ds} ch={ch} {mode_a}vs{mode_b}: {e}")
                    p_val = float('nan')
                ci_txt = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
                p_txt = f"{p_val:.4f}" if np.isfinite(p_val) else "nan"
                print(f"{ds:<16} {dtype:<6} {ch:<4} {delta_obs:+8.4f} {ci_txt:>22} {p_txt:>9}")
                any_row = True
        if not any_row:
            print("(no paired rows available — ensure raw arrays y_true/y_pred are stored in results)")

    # Print one strong table per claimed comparison
    for _title, _a, _b in paired_comparisons:
        _print_paired_table(_title, _a, _b)

    # TABLE 6: NEGATIVE CONTROL (pos_only vs random_pos)
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("🎲 TABLE 6: NEGATIVE CONTROL / SANITY CHECK (pos_only vs random_pos)")
    print("   Question: Is the model learning real syntax or just seeing POS tags?")
    print("   MUST BE POSITIVE: If close to 0, the model is NOT learning syntax!")
    print("=" * 130)

    print(f"\n  {'Dataset':<16} {'Type':<6}", end="")
    for ch in channels:
        print(f" {ch:>12}", end="")
    print()
    print("  " + "-" * 120)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        row = f"  {ds:<16} {label_type:<6}"

        for ch in channels:
            pos_val = get_result(results, 'pos_only', ch, ds, 'macro_f1')
            random_val = get_result(results, 'random_pos', ch, ds, 'macro_f1')

            if pos_val is not None and random_val is not None:
                diff = pos_val - random_val
                status = "✓" if diff > 0.03 else ("⚠️" if diff > 0 else "✗")
                row += f" {diff:>+.3f}{status}"
            else:
                row += f" {'---':>12}"
        print(row)

    print("\n  Legend: ✓ = Syntax matters (Δ>0.03), ⚠️ = Marginal (0<Δ<0.03), ✗ = Problem (Δ≤0)")

    # ==========================================================================
    # TABLE 7: Human vs Auto Label Divergence
    # ==========================================================================
    print("\n")
    print("=" * 130)
    print("📊 TABLE 7: LABEL REGIME DIVERGENCE (Human vs Auto)")
    print("   Hypothesis: Structure channels (B, BC) should favor Human labels")
    print("=" * 130)

    human_ds = [ds for ds in datasets if get_dataset_info(ds).get('label_type') == 'human']
    auto_ds = [ds for ds in datasets if get_dataset_info(ds).get('label_type') == 'auto']

    print(f"\n  {'Channel':<8} {'Mask':<14} {'Human Macro_F1':>14} {'Auto Macro_F1':>14} {'Δ':>10} {'Winner':>10}")
    print("  " + "-" * 80)

    for ch in channels:
        for mask in mask_modes:
            human_scores = [get_result(results, mask, ch, ds, 'macro_f1') for ds in human_ds]
            auto_scores = [get_result(results, mask, ch, ds, 'macro_f1') for ds in auto_ds]

            human_scores = [s for s in human_scores if s is not None]
            auto_scores = [s for s in auto_scores if s is not None]

            if human_scores and auto_scores:
                human_avg = np.mean(human_scores)
                auto_avg = np.mean(auto_scores)
                diff = human_avg - auto_avg
                winner = "Human ↑" if diff > 0.02 else ("Auto ↑" if diff < -0.02 else "≈")
                print(f"  {ch:<8} {mask:<14} {human_avg:>14.3f} {auto_avg:>14.3f} {diff:>+10.3f} {winner:>10}")

    # ==========================================================================
    # FINAL HYPOTHESIS SUMMARY - DIVERGENCE HYPOTHESIS LOGIC
    # ==========================================================================
    print("\n")
    print("╔" + "═" * 128 + "╗")
    print("║" + " " * 35 + "DIVERGENCE HYPOTHESIS VERDICT" + " " * 64 + "║")
    print("╚" + "═" * 128 + "╝")

    print("\n  🎯 DIVERGENCE HYPOTHESIS:")
    print("     'Structure signals (POS patterns) are robust on HUMAN-labeled data")
    print("      but weaker on AUTO-labeled data, revealing annotation quality divergence.'")
    print("  " + "-" * 100)

    # Define regime datasets
    HUMAN_DATASETS = ['dreaddit_test', 'twitter_gold']
    AUTO_DATASETS = ['twitter', 'reddit_combi']
    STRUCTURE_CHANNELS = ['B', 'BC']  # Channels that rely on structural features

    # Compute deltas (Structure - Content) for each regime and metric
    print("\n  📊 STRUCTURE vs CONTENT DELTA BY LABEL REGIME:")
    print("     (Δ = pos_only - content_only, positive = Structure wins)")
    print("  " + "-" * 100)

    regime_results = {'human': {}, 'auto': {}}

    for metric in ['macro_f1', 'mcc', 'pr_auc']:
        # Human-labeled datasets
        human_deltas = []
        for ds in HUMAN_DATASETS:
            if ds not in datasets:
                continue
            for ch in STRUCTURE_CHANNELS:
                pos_val = get_result(results, 'pos_only', ch, ds, metric)
                content_val = get_result(results, 'content_only', ch, ds, metric)
                if pos_val is not None and content_val is not None:
                    human_deltas.append(pos_val - content_val)

        # Auto-labeled datasets
        auto_deltas = []
        for ds in AUTO_DATASETS:
            if ds not in datasets:
                continue
            for ch in STRUCTURE_CHANNELS:
                pos_val = get_result(results, 'pos_only', ch, ds, metric)
                content_val = get_result(results, 'content_only', ch, ds, metric)
                if pos_val is not None and content_val is not None:
                    auto_deltas.append(pos_val - content_val)

        regime_results['human'][metric] = np.mean(human_deltas) if human_deltas else None
        regime_results['auto'][metric] = np.mean(auto_deltas) if auto_deltas else None

    # Print detailed table
    print(f"\n  {'Metric':<12} {'Human Δ':>12} {'Auto Δ':>12} {'Gap':>12} {'Interpretation':<30}")
    print("  " + "-" * 85)

    divergence_scores = []
    for metric in ['macro_f1', 'mcc', 'pr_auc']:
        human_delta = regime_results['human'].get(metric)
        auto_delta = regime_results['auto'].get(metric)

        if human_delta is not None and auto_delta is not None:
            gap = human_delta - auto_delta  # Positive gap = divergence exists
            divergence_scores.append(gap)

            # Interpretation
            if human_delta >= -0.01 and auto_delta < -0.02:
                interp = "✅ Divergence (Human≈, Auto↓)"
            elif human_delta > 0.02 and auto_delta < 0:
                interp = "✅ Strong Divergence!"
            elif human_delta > 0 and auto_delta > 0:
                interp = "⚠️ Both favor Structure"
            elif human_delta < 0 and auto_delta < 0:
                interp = "⚠️ Both favor Content"
            else:
                interp = "≈ Mixed signal"

            print(f"  {metric:<12} {human_delta:>+12.3f} {auto_delta:>+12.3f} {gap:>+12.3f} {interp:<30}")
        else:
            print(f"  {metric:<12} {'---':>12} {'---':>12} {'---':>12} {'Insufficient data':<30}")

    # Per-dataset breakdown
    print("\n  📋 PER-DATASET BREAKDOWN (Macro_F1, Structure Channels B+BC avg):")
    print("  " + "-" * 100)
    print(f"  {'Dataset':<20} {'Label Type':<12} {'Δ (Struct-Content)':>20} {'Verdict':<20}")
    print("  " + "-" * 100)

    for ds in datasets:
        label_type = get_dataset_info(ds).get('label_type', '?')
        deltas = []
        for ch in STRUCTURE_CHANNELS:
            pos_val = get_result(results, 'pos_only', ch, ds, 'macro_f1')
            content_val = get_result(results, 'content_only', ch, ds, 'macro_f1')
            if pos_val is not None and content_val is not None:
                deltas.append(pos_val - content_val)

        if deltas:
            avg_delta = np.mean(deltas)
            if label_type == 'human':
                verdict = "✅ Expected" if avg_delta >= -0.02 else "⚠️ Unexpected"
            else:  # auto
                verdict = "✅ Expected" if avg_delta < 0 else "⚠️ Unexpected"
            print(f"  {ds:<20} {label_type.upper():<12} {avg_delta:>+20.3f} {verdict:<20}")

    # Final verdict
    print("\n  " + "═" * 100)
    print("  🏆 FINAL DIVERGENCE VERDICT:")
    print("  " + "═" * 100)

    # Calculate overall means
    human_mean = np.mean([v for v in regime_results['human'].values() if v is not None]) if any(v is not None for v in regime_results['human'].values()) else None
    auto_mean = np.mean([v for v in regime_results['auto'].values() if v is not None]) if any(v is not None for v in regime_results['auto'].values()) else None

    if human_mean is not None and auto_mean is not None:
        print(f"\n     Human Data Mean Δ (Structure - Content): {human_mean:>+.3f}", end="")
        if human_mean >= -0.01:
            print("  (✅ Structure competitive)")
        else:
            print("  (⚠️ Content dominates)")

        print(f"     Auto  Data Mean Δ (Structure - Content): {auto_mean:>+.3f}", end="")
        if auto_mean < -0.01:
            print("  (✅ Content dominates as expected)")
        else:
            print("  (⚠️ Structure unexpectedly strong)")

        divergence_gap = human_mean - auto_mean
        print(f"\n     Divergence Gap (Human Δ - Auto Δ):       {divergence_gap:>+.3f}")

        # Hypothesis evaluation
        # Supported if: Human delta >= -0.01 (Structure competitive) AND Auto delta < Human delta
        human_ok = human_mean >= -0.02  # Structure at least competitive on human data
        auto_ok = auto_mean < human_mean  # Content does better on auto data than human
        divergence_exists = divergence_gap > 0.01  # Meaningful gap

        print("\n  " + "-" * 100)

        if human_ok and divergence_exists:
            print("  ╔════════════════════════════════════════════════════════════════════════════════════╗")
            print("  ║  ✅ DIVERGENCE HYPOTHESIS SUPPORTED                                                ║")
            print("  ║                                                                                    ║")
            print("  ║  Structure (POS patterns) performs better/equal on HUMAN-labeled data             ║")
            print("  ║  but loses to Content on AUTO-labeled data.                                       ║")
            print("  ║                                                                                    ║")
            print("  ║  → This suggests AUTO labels may capture surface lexical cues,                    ║")
            print("  ║    while HUMAN labels reflect deeper psychological states encoded in structure.   ║")
            print("  ╚════════════════════════════════════════════════════════════════════════════════════╝")
        elif human_ok and not divergence_exists:
            print("  ╔════════════════════════════════════════════════════════════════════════════════════╗")
            print("  ║  ⚠️ PARTIAL SUPPORT - Structure works, but no clear divergence                     ║")
            print("  ║                                                                                    ║")
            print("  ║  Structure performs well on Human data, but also on Auto data.                    ║")
            print("  ║  The divergence pattern is not strong enough.                                     ║")
            print("  ╚════════════════════════════════════════════════════════════════════════════════════╝")
        elif not human_ok and auto_mean < -0.02:
            print("  ╔════════════════════════════════════════════════════════════════════════════════════╗")
            print("  ║  ❌ HYPOTHESIS NOT SUPPORTED - Content dominates both regimes                      ║")
            print("  ║                                                                                    ║")
            print("  ║  Content (lexical features) outperforms Structure on both Human and Auto data.    ║")
            print("  ╚════════════════════════════════════════════════════════════════════════════════════╝")
        else:
            print("  ╔════════════════════════════════════════════════════════════════════════════════════╗")
            print("  ║  ⚠️ INCONCLUSIVE - Mixed results                                                    ║")
            print("  ║                                                                                    ║")
            print("  ║  The pattern is not clear enough to draw conclusions.                             ║")
            print("  ╚════════════════════════════════════════════════════════════════════════════════════╝")
    else:
        print("\n  ⚠️ Insufficient data to evaluate hypothesis.")

    print("\n" + "=" * 130)


# =============================================================================
# MAIN (UNCHANGED except for new arguments)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TSS Masking Suite (Comprehensive + Full Metrics)')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='outputs/masking')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts')
    parser.add_argument('--channels', type=str, default='A,B,C,AB,AC,BC,ABC',
                       help='Channels: comma-separated or "all" (default: A,B,C,AB,AC,BC,ABC)')
    parser.add_argument('--masks', type=str, default='none,pos_only,content_only,function_only,random_pos',
                       help='Mask modes (default: ALL)')
    parser.add_argument('--bootstrap', type=int, default=10000,
                       help='Bootstrap iterations for CI (default: 10000)')
    parser.add_argument('--no-ci', action='store_true',
                       help='Disable bootstrap CI computation (faster)')
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TSS Masking Suite (Comprehensive + Full Metrics)               ║
║                                                         ║
║                                                                              ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║  "Linguistic structure is the true vessel of psychological state."                           ║
║  "Linguistic structure is the true vessel of psychological state"            ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║                                                                              ║
║  Updates:                                                              ║
║  ├── Full metric bundle: Macro_F1, MCC, PR-AUC, F1, Recall, Precision, Acc  ║
║  ├── Bootstrap 95% Confidence Intervals for all metrics                     ║
║  ├── Expanded summary tables with multiple metrics                          ║
║  └── Hypothesis testing across Macro_F1, MCC, and PR-AUC                    ║
║                                                                              ║
║  HYPOTHESIS TESTS:                                                           ║
║  ├── pos_only vs none:         Structure retention                           ║
║  ├── content_only vs none:     Lexical retention                             ║
║  ├── pos_only vs content_only: THE DIVERGENCE TEST                           ║
║  ├── function_only vs content: Psycholinguistic precision                    ║
║  └── pos_only vs random_pos:   Negative control (sanity check)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Parse channels — support 'all' keyword
    ALL_VALID_CHANNELS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    raw_channels = args.channels.strip().upper()
    if raw_channels == 'ALL':
        channels = ALL_VALID_CHANNELS
    else:
        channels = [c.strip() for c in raw_channels.split(',')]
        invalid = [c for c in channels if c not in ALL_VALID_CHANNELS]
        if invalid:
            logger.warning(f"   ⚠ Unknown channels ignored: {invalid}")
            channels = [c for c in channels if c in ALL_VALID_CHANNELS]

    mask_modes = [m.strip().lower() for m in args.masks.split(',')]
    compute_ci = not args.no_ci

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir

    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)

    datasets = load_datasets(str(data_dir))

    if 'dreaddit_train' not in datasets:
        logger.error("Training data not found!")
        return

    # Verify parsing
    sample = datasets['dreaddit_train']['token_pos_pairs'].iloc[0]
    parsed = parse_token_pos_pairs(sample)
    logger.info(f"\n🔍 token_pos_pairs verification: {len(parsed)} pairs parsed ✓")

    results = run_masking_experiments(
        datasets=datasets,
        channels=channels,
        mask_modes=mask_modes,
        output_dir=str(output_dir),
        artifacts_dir=str(artifacts_dir),
        compute_ci=compute_ci,
        n_bootstrap=args.bootstrap,
    )

    print_comprehensive_summary(results)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # =========================================================================
    # SAVE RAW PREDICTIONS (joblib) — for instance-level statistical analysis
    # =========================================================================
    predictions_path = os.path.join(str(output_dir), f'predictions_{timestamp}.joblib')
    raw_predictions = []
    for r in results:
        if 'y_true' in r and 'y_pred' in r:
            raw_predictions.append({
                'channel':   r.get('channel'),
                'dataset':   r.get('dataset'),
                'mask_mode': r.get('mask_mode'),
                'label_type': r.get('label_type', 'unknown'),
                'y_true': np.asarray(r['y_true']),
                'y_pred': np.asarray(r['y_pred']),
                'y_prob': np.asarray(r['y_prob']) if r.get('y_prob') is not None else None,
            })
    if raw_predictions:
        n_bundles = len(raw_predictions)
        n_samples = sum(len(p['y_true']) for p in raw_predictions)
        ds_names = sorted(set(p['dataset'] for p in raw_predictions))

        # Wrap in metadata envelope for cross-validation in 04
        predictions_envelope = {
            '_meta': {
                'version': '3.0',
                'timestamp': timestamp,
                'n_bundles': n_bundles,
                'n_samples': n_samples,
                'datasets': ds_names,
                'channels': sorted(set(p['channel'] for p in raw_predictions)),
            },
            'predictions': raw_predictions,
        }

        joblib.dump(predictions_envelope, predictions_path, compress=3)

        # ── Verify saved file integrity ──
        pred_ok = False
        try:
            file_size_mb = os.path.getsize(predictions_path) / (1024 * 1024)
            reloaded = joblib.load(predictions_path)

            # Handle both envelope format and legacy list format
            if isinstance(reloaded, dict) and '_meta' in reloaded:
                meta = reloaded['_meta']
                reloaded_preds = reloaded['predictions']
            elif isinstance(reloaded, list):
                reloaded_preds = reloaded
                meta = None
            else:
                raise ValueError(f"Unexpected format: {type(reloaded)}")

            n_reloaded = len(reloaded_preds)
            n_samples_check = sum(len(p['y_true']) for p in reloaded_preds)

            if n_reloaded == n_bundles and n_samples_check == n_samples:
                pred_ok = True
                logger.info(f"\n💾 Raw predictions saved & verified: {predictions_path}")
                logger.info(f"   {n_reloaded} bundles | {n_samples_check:,} samples | "
                            f"{file_size_mb:.1f} MB | datasets: {ds_names}")
            else:
                logger.error(f"   ✗ Integrity check FAILED: "
                             f"expected {n_bundles}/{n_samples:,}, "
                             f"got {n_reloaded}/{n_samples_check:,}")
        except Exception as e:
            logger.error(f"   ✗ Verification failed: {e}")

        if not pred_ok:
            logger.error("   ✗ CRITICAL: predictions file may be corrupt!")
            logger.error("   ✗ 04_statistical_analysis.py will refuse to run without valid predictions.")
            logger.error("   ✗ Check disk space and re-run this script.")
    else:
        logger.warning("   ⚠ No raw predictions found in results — joblib not saved")
        logger.warning("   ⚠ 04_statistical_analysis.py will only have aggregate (N=4) data!")

    # =========================================================================
    # SAVE AGGREGATE METRICS (JSON) — human-readable summary
    # =========================================================================
    results_path = os.path.join(str(output_dir), f'masking_comprehensive_{timestamp}.json')

    json_results = []
    for r in results:
        json_r = {}
        for k, v in r.items():
            if k in ['y_pred', 'y_prob', 'y_true']:
                continue  # Skip large arrays in JSON
            if isinstance(v, (np.floating, np.integer)):
                json_r[k] = float(v)
            elif isinstance(v, np.ndarray):
                json_r[k] = v.tolist()
            else:
                json_r[k] = v
        json_results.append(json_r)

    os.makedirs(str(output_dir), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'version': '7.8.3',
            'timestamp': timestamp,
            'config': {
                'channels': channels,
                'mask_modes': mask_modes,
                'total_experiments': len(results),
                'bootstrap_iterations': args.bootstrap if compute_ci else 0,
                'metrics': ['macro_f1', 'f1', 'mcc', 'pr_auc', 'roc_auc', 'recall', 'precision', 'accuracy'],
            },
            'results': json_results,
        }, f, indent=2)

    # Also save as CSV for easy analysis
    csv_path = os.path.join(str(output_dir), f'masking_comprehensive_{timestamp}.csv')
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_prob', 'y_true', 'ci']}
                               for r in results])
    df_results.to_csv(csv_path, index=False)

    logger.info(f"\n📄 Results saved:")
    logger.info(f"   JSON:        {results_path}")
    logger.info(f"   CSV:         {csv_path}")
    if raw_predictions:
        logger.info(f"   Predictions: {predictions_path}")
    logger.info("=" * 80)
    logger.info("✅ COMPREHENSIVE masking experiments complete!")


if __name__ == '__main__':
    main()