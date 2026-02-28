#!/usr/bin/env python3
"""
================================================================================
TSS: Training, Evaluation & Interpretability Pipeline
================================================================================
The core engine for the "Transferable Stress Signals" (TSS) framework, designed
to validate the Labeling-Regime Divergence Hypothesis.

--------------------------------------------------------------------------------
1. THEORETICAL FRAMEWORK: THE DIVERGENCE HYPOTHESIS
--------------------------------------------------------------------------------
"Linguistic structure is the true vessel of psychological state."
"Linguistic structure is the true vessel of psychological state."

This script tests the hypothesis that Human-labeled data (Gold Standard) aligns
strongly with Structural/Stylistic features (Channels B & C), whereas Auto-labeled
data (Silver Standard) is biased towards Surface Lexical patterns (Channel A).

STRATEGIC DECISION: FULL DATA RETENTION
- We strictly AVOID undersampling. Deleting data = Deleting reality.
- Instead, we utilize `class_weight='balanced'` to enforce algorithmic fairness
  without distorting the natural epidemiological prevalence of stress.

--------------------------------------------------------------------------------
2. EVALUATION REGIME (Rigorous & Nested)
--------------------------------------------------------------------------------
A. METRIC INTEGRITY:
   - Primary: Macro F1 (Treats both classes equally, unbiased by prevalence).
   - Bundle: Precision, Recall, PR-AUC, MCC, and Specificity.
   - Baselines: Every evaluation starts with a "Majority Class Baseline" to
     quantify true model lift.

B. CROSS-DOMAIN VALIDATION (LODO):
   - Implements "Nested Validation" to prevents optimistic bias:
     1. Train (60%) -> Learn weights.
     2. Calibrate (20%) -> Optimize decision threshold.
     3. Validate (20%) -> Measure In-Domain score.
     4. Test (Held-out Domain) -> Measure Out-of-Domain Transferability.

C. STATISTICAL VALIDATION:
   - Channel Ranking by Regime: Separately ranks channels on Human vs. Auto datasets
     to statistically confirm the "Divergence Hypothesis."

--------------------------------------------------------------------------------
3. INTERPRETABILITY SUITE (The "Why")
--------------------------------------------------------------------------------
This pipeline goes beyond metrics to explain *behavior*:

A. ERROR ANALYSIS EXPORT (Excel):
   - "Shift Cases": Samples where Lexicon (A) Fails but Structure (B) Succeeds.
   - "Gold Disagreement": Where Auto-labels say "Stressed" but the Model says
     "Not Stressed" (revealing noise in auto-labeled datasets).

B. SHAP EXPLAINABILITY:
   - Uses Game Theory (Shapley Values) to mathematically prove which features
     drove the decision in "Shift Cases" (proving structure > keywords).

C. STYLISTIC PROFILING (K-Means):
   - Unsupervised clustering of Channel C to prove the model detects distinct
     writing styles (e.g., "Anxious Ranting" vs. "Formal Discussion").

USAGE:
    python scripts/02_train_evaluate.py --channels all
    # Run with full interpretability suite (SHAP + K-Means):
    python scripts/02_train_evaluate.py --channels BC --run_advanced_analysis
================================================================================
"""
import argparse
import json
import sys
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score,
    precision_recall_curve, auc as sklearn_auc, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from tss.pipeline import (
    TSSClassifier,
    train_channel,
    evaluate_model,
    optimize_threshold,
    ThresholdOptimizer,
    TrainingArtifact,
)
from tss.config import __version__, SYSTEM_NAME, ALL_DOMAINS, DOMAIN_MAP

# =============================================================================
# LOGGING — visible, not suppressed
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('sklearn').setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_CHANNELS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
RANDOM_STATE = 42

CHANNEL_DESCRIPTIONS = {
    'A': 'Character TF-IDF (Lexical Surface Patterns)',
    'B': 'Lexical Polarity (Class-Conditional Log-Odds)',
    'C': 'Psycholinguistic Features (Length-Normalised)',
    'AB': 'A + B Combined',
    'AC': 'A + C Combined',
    'BC': 'B + C Combined',
    'ABC': 'All Channels Combined',
}

# Dataset metadata for label type classification
DATASET_INFO = {
    'dreaddit_train': {'label_type': 'human', 'platform': 'reddit'},
    'dreaddit_test': {'label_type': 'human', 'platform': 'reddit'},
    'twitter': {'label_type': 'auto', 'platform': 'twitter'},
    'twitter_gold': {'label_type': 'human', 'platform': 'twitter'},
    'reddit_combi': {'label_type': 'auto', 'platform': 'reddit'},
}


# =============================================================================
# MAJORITY BASELINE COMPUTATION
# =============================================================================

def compute_majority_baseline(y_true: np.ndarray) -> Dict:
    """
    Compute majority class baseline metrics.

    This shows what a naive model (always predicting the majority class)
    would achieve - essential for demonstrating model value on imbalanced data.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {'majority_f1': 0.0, 'majority_macro_f1': 0.0, 'prevalence': 0.0, 'majority_class': 0}

    n_positive = int(np.sum(y_true == 1))
    n_negative = n_samples - n_positive

    # Majority prediction
    if n_positive >= n_negative:
        y_pred_majority = np.ones_like(y_true)
        majority_class = 1
    else:
        y_pred_majority = np.zeros_like(y_true)
        majority_class = 0

    majority_f1 = f1_score(y_true, y_pred_majority, zero_division=0)
    majority_macro_f1 = f1_score(y_true, y_pred_majority, average='macro', zero_division=0)

    return {
        'majority_f1': float(majority_f1),
        'majority_macro_f1': float(majority_macro_f1),
        'prevalence': float(n_positive / n_samples),
        'majority_class': int(majority_class),
    }


# =============================================================================
# BOOTSTRAP CI - ENHANCED FOR MACRO F1
# =============================================================================

def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_iterations: int,
                 metric: str = 'macro_f1') -> Dict:
    """
    Compute bootstrap confidence interval for F1 metrics.

    Args:
        metric: 'f1' for positive class F1, 'macro_f1' for macro-averaged F1
    """
    np.random.seed(RANDOM_STATE)
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue

        if metric == 'macro_f1':
            score = f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0)
        else:
            score = f1_score(y_true[idx], y_pred[idx], zero_division=0)
        scores.append(score)

    if len(scores) < 100:
        logger.warning(f"Only {len(scores)} valid bootstrap samples")
        return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0, 'n_iterations': len(scores)}

    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'ci_lower': float(np.percentile(scores, 2.5)),
        'ci_upper': float(np.percentile(scores, 97.5)),
        'n_iterations': len(scores),
    }


# =============================================================================
# COMPREHENSIVE METRIC BUNDLE
# =============================================================================

def compute_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """
    Compute comprehensive metric bundle .

    PRIMARY METRIC: Macro F1 (treats both classes equally - fair for imbalanced data)

    MANDATORY METRICS:
    - Positive F1 (for stress=1 class)
    - Positive Recall (critical for mental health applications)
    - PR-AUC (better than ROC-AUC for imbalanced data)
    - MCC (Matthews Correlation Coefficient - robust to imbalance)
    - Prevalence (% positive in dataset - for context)
    - Majority Baseline F1 (to demonstrate model value)
    """
    n_samples = len(y_true)
    n_positive = int(np.sum(y_true == 1))
    prevalence = n_positive / n_samples if n_samples > 0 else 0.0

    # PRIMARY METRIC: Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Positive class metrics
    positive_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    positive_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    positive_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # MCC - Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # PR-AUC
    pr_auc = prevalence  # Default to prevalence (random baseline)
    if len(np.unique(y_true)) > 1:
        try:
            pr_auc = average_precision_score(y_true, y_proba)
        except ValueError:
            pass

    # ROC-AUC (for reference)
    roc_auc = 0.5
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass

    # Majority Baseline
    baseline = compute_majority_baseline(y_true)

    # Delta from baseline
    delta_macro_f1 = macro_f1 - baseline['majority_macro_f1']
    delta_f1 = positive_f1 - baseline['majority_f1']

    # Confusion matrix for specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0

    return {
        # Primary
        'macro_f1': float(macro_f1),

        # Positive class metrics
        'f1': float(positive_f1),
        'recall': float(positive_recall),
        'precision': float(positive_precision),

        # Other metrics
        'balanced_accuracy': float(balanced_acc),
        'mcc': float(mcc),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'specificity': float(specificity),

        # Dataset info
        'prevalence': float(prevalence),
        'n_samples': n_samples,
        'n_positive': n_positive,

        # Baselines
        'majority_f1': baseline['majority_f1'],
        'majority_macro_f1': baseline['majority_macro_f1'],
        'delta_macro_f1': float(delta_macro_f1),
        'delta_f1': float(delta_f1),
    }


# =============================================================================
# DATA LOADING - NO UNDERSAMPLING
# =============================================================================

def load_data(data_dir: Path, remove_duplicates: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets with comprehensive duplicate removal.

    CRITICAL: NO undersampling/balancing - use FULL DATA.
    class_weight='balanced' in the model handles imbalance.

    Removes:
    - Internal duplicates within each dataset
    - Cross-dataset duplicates: train↔test, train↔twitter, train↔twitter_gold, train↔reddit_combi : Added twitter_gold (human-labeled cross-platform dataset)
    """
    datasets = {}

    for name, filename in [
        ('dreaddit_train', 'dreaddit_train.csv'),
        ('dreaddit_test', 'dreaddit_test.csv'),
        ('twitter', 'twitter_processed.csv'),
        ('twitter_gold', 'twitter_gold_processed.csv'),  # CRITICAL: Human-labeled!
        ('reddit_combi', 'reddit_combi_processed.csv'),
    ]:
        path = data_dir / filename
        if path.exists():
            df = pd.read_csv(path)

            # Remove duplicates only (NO undersampling!)
            if remove_duplicates:
                n_before = len(df)
                df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
                n_removed = n_before - len(df)
                if n_removed > 0:
                    logger.info(f"   Removed {n_removed} internal duplicates from {name}")

            datasets[name] = df

            # Log with label type
            ds_info = DATASET_INFO.get(name, {})
            label_type = ds_info.get('label_type', 'unknown').upper()
            n_pos = df['label'].sum() if 'label' in df.columns else 0
            prev = n_pos / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"   ✓ {name}: {len(df):,} samples ({prev:.1f}% pos) [{label_type} labels]")

    # Cross-dataset duplicate removal
    if remove_duplicates and 'dreaddit_train' in datasets:
        train_texts = set(datasets['dreaddit_train']['cleaned_text'].dropna().astype(str))

        for target in ['dreaddit_test', 'twitter', 'twitter_gold', 'reddit_combi']:
            if target not in datasets:
                continue
            target_texts = set(datasets[target]['cleaned_text'].dropna().astype(str))
            dupes = train_texts & target_texts
            if dupes:
                mask = ~datasets['dreaddit_train']['cleaned_text'].astype(str).isin(dupes)
                n_before = len(datasets['dreaddit_train'])
                datasets['dreaddit_train'] = datasets['dreaddit_train'][mask].reset_index(drop=True)
                logger.warning(f"   ⚠️ Removed {n_before - len(datasets['dreaddit_train'])} "
                             f"cross-duplicates: train↔{target}")
                train_texts = set(datasets['dreaddit_train']['cleaned_text'].dropna().astype(str))

    return datasets


# =============================================================================
# LODO — FIXED WITH NESTED VALIDATION
# =============================================================================

def run_lodo(
    train_df: pd.DataFrame,
    channel: str,
    n_bootstrap: int = 1000,
) -> List[Dict]:
    """
    Leave-One-Domain-Out evaluation with NESTED validation.

    For each held-out domain:
      1. train_inner (60% of remaining domains) → train model
      2. cal_inner   (20% of remaining domains) → calibrate threshold
      3. val_inner   (20% of remaining domains) → measure in-domain F1
      4. test_fold   (held-out domain)           → measure out-domain F1
    """
    results = []

    if 'domain' not in train_df.columns:
        logger.warning("No domain column found")
        return results

    train_df = train_df.reset_index(drop=True)
    domains = sorted([d for d in train_df['domain'].unique() if d in ALL_DOMAINS])
    logger.info(f"      Domains found: {domains}")

    for heldout in domains:
        train_fold = train_df[train_df['domain'] != heldout].reset_index(drop=True)
        test_fold = train_df[train_df['domain'] == heldout].reset_index(drop=True)

        if len(test_fold) < 20:
            logger.warning(f"      Skipping {heldout}: only {len(test_fold)} samples")
            continue

        logger.info(f"      LODO: {heldout} (train={len(train_fold)}, test={len(test_fold)})...")

        y_fold = train_fold['label'].astype(int).values

        # Three-way split: train (60%) / calibrate (20%) / validate (20%)
        train_inner, rest = train_test_split(
            train_fold, test_size=0.4, stratify=y_fold, random_state=RANDOM_STATE
        )
        y_rest = rest['label'].astype(int).values
        cal_inner, val_inner = train_test_split(
            rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE
        )

        train_inner = train_inner.reset_index(drop=True)
        cal_inner = cal_inner.reset_index(drop=True)
        val_inner = val_inner.reset_index(drop=True)

        # 1. Train on train_inner
        model = TSSClassifier(channel=channel, use_svd=False, transfer_mode=True)
        model.fit(train_inner, train_inner['label'].astype(int).values)

        # 2. Calibrate threshold on cal_inner
        y_cal = cal_inner['label'].astype(int).values
        y_proba_cal = model.predict_proba(cal_inner)[:, 1]
        optimal_thr, _ = optimize_threshold(y_cal, y_proba_cal)
        model.set_threshold(optimal_thr)

        # 3. In-domain F1 on val_inner (never seen by model or threshold)
        in_result = evaluate_model(model, val_inner)

        # 4. Out-domain F1 on held-out domain
        out_result = evaluate_model(model, test_fold)

        # Compute macro F1 for both
        y_in_true = val_inner['label'].astype(int).values
        y_in_pred = np.array(in_result['y_pred'])
        in_macro_f1 = f1_score(y_in_true, y_in_pred, average='macro', zero_division=0)

        y_out_true = test_fold['label'].astype(int).values
        y_out_pred = np.array(out_result['y_pred'])
        out_macro_f1 = f1_score(y_out_true, y_out_pred, average='macro', zero_division=0)

        domain_gap = in_result['f1'] - out_result['f1']
        domain_gap_macro = in_macro_f1 - out_macro_f1

        results.append({
            'channel': channel,
            'heldout_domain': heldout,
            'n_train': len(train_inner),
            'n_cal': len(cal_inner),
            'n_val_in': len(val_inner),
            'n_test_out': len(test_fold),
            'in_f1': in_result['f1'],
            'out_f1': out_result['f1'],
            'in_macro_f1': in_macro_f1,
            'out_macro_f1': out_macro_f1,
            'domain_gap': domain_gap,
            'domain_gap_macro': domain_gap_macro,
            'out_precision': out_result['precision'],
            'out_recall': out_result['recall'],
            'out_auc': out_result['auc'],
            'threshold': optimal_thr,
        })

        logger.info(f"         In-F1: {in_result['f1']:.3f} (val), "
                    f"Out-F1: {out_result['f1']:.3f}, Gap: {domain_gap:.3f}, "
                    f"macro_Gap: {domain_gap_macro:.3f}")

    return results


# =============================================================================
# CROSS-PLATFORM ANALYSIS - ENHANCED FOR LABEL REGIME TESTING
# =============================================================================

def compute_cross_platform(
    eval_results: Dict[str, Dict],
    source_name: str = 'dreaddit_test',
) -> List[Dict]:
    """
    Compute cross-platform transfer metrics using MACRO F1.

    Key hypothesis ():
    - Human labels (dreaddit_test, twitter_gold) should align better with B/C channels
    - Auto labels (twitter, reddit_combi) may align better with A channel (lexical)
    """
    results = []
    if source_name not in eval_results:
        return results

    source = eval_results[source_name]
    source_macro_f1 = source.get('macro_f1', source.get('f1', 0)) or 0

    # Include twitter_gold in targets
    for target_name in ['twitter', 'twitter_gold', 'reddit_combi']:
        if target_name not in eval_results:
            continue
        target = eval_results[target_name]
        target_macro_f1 = target.get('macro_f1', target.get('f1', 0)) or 0

        # Safe division: prevent ZeroDivisionError
        src_f1 = source.get('f1', 0) or 0
        src_macro = source_macro_f1 or 0
        retention_f1 = (target.get('f1', 0) / src_f1 * 100) if src_f1 > 0 else 0.0
        retention_macro_f1 = (target_macro_f1 / src_macro * 100) if src_macro > 0 else 0.0

        # Determine label type for analysis
        target_label_type = DATASET_INFO.get(target_name, {}).get('label_type', 'unknown')

        results.append({
            'source': source_name,
            'source_f1': src_f1,
            'source_macro_f1': src_macro,
            'target': target_name,
            'target_f1': target.get('f1', 0) or 0,
            'target_macro_f1': target_macro_f1,
            'target_label_type': target_label_type,
            'retention_rate': retention_f1,
            'retention_rate_macro': retention_macro_f1,
            'degradation': 100 - retention_f1,
            'f1_gap': src_f1 - (target.get('f1', 0) or 0),
            'macro_f1_gap': src_macro - target_macro_f1,
        })
    return results


# =============================================================================
# CHANNEL RANKING BY REGIME (replaces Friedman — invalid with n<3)
# =============================================================================

def run_channel_ranking_by_regime(eval_results: List[Dict]) -> Dict:
    """
    Rank channels separately within human and auto regimes using mean Macro-F1.
    Replaces Friedman tests which require n>=3 datasets per regime.

    This tests the DIVERGENCE HYPOTHESIS:
    - Human labels may favor structure/style channels (B, C)
    - Auto labels may favor lexical channels (A)
    """
    human_datasets = [ds for ds, info in DATASET_INFO.items()
                      if info.get('label_type') == 'human' and ds != 'dreaddit_train']
    auto_datasets = [ds for ds, info in DATASET_INFO.items()
                     if info.get('label_type') == 'auto']

    channels = sorted(set(r.get('channel', '') for r in eval_results))

    results = {
        'table_a_human': None,
        'table_b_auto': None,
        'divergence_analysis': None,
    }

    def compute_rankings(dataset_list: List[str], grouping: str) -> Optional[Dict]:
        filtered = [r for r in eval_results if r.get('dataset') in dataset_list]
        ds_found = sorted(set(r['dataset'] for r in filtered))
        if len(ds_found) < 1 or len(channels) < 2:
            return None

        scores_matrix = np.zeros((len(ds_found), len(channels)))
        for i, ds in enumerate(ds_found):
            for j, ch in enumerate(channels):
                sc = [r.get('macro_f1', 0) for r in filtered
                      if r.get('dataset') == ds and r.get('channel') == ch]
                scores_matrix[i, j] = sc[0] if sc else 0

        # Compute average ranks (higher score = lower rank = better)
        ranks = np.zeros_like(scores_matrix)
        for i in range(scores_matrix.shape[0]):
            ranks[i] = stats.rankdata(-scores_matrix[i])

        avg_ranks = np.mean(ranks, axis=0)
        rankings = {ch: float(rank) for ch, rank in zip(channels, avg_ranks)}
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])

        # Mean F1 per channel
        mean_f1 = {ch: float(scores_matrix[:, j].mean()) for j, ch in enumerate(channels)}

        return {
            'grouping': grouping,
            'metric': 'macro_f1',
            'datasets': ds_found,
            'channels': channels,
            'rankings': rankings,
            'sorted_rankings': sorted_rankings,
            'mean_f1_per_channel': mean_f1,
            'scores_matrix': scores_matrix.tolist(),
            'note': 'Rankings by mean Macro-F1 (Friedman removed: n<3 per regime)',
        }

    results['table_a_human'] = compute_rankings(human_datasets, 'human_labels')
    results['table_b_auto'] = compute_rankings(auto_datasets, 'auto_labels')

    # Divergence Analysis
    if results['table_a_human'] and results['table_b_auto']:
        human_ranks = results['table_a_human'].get('rankings', {})
        auto_ranks = results['table_b_auto'].get('rankings', {})

        if human_ranks and auto_ranks:
            divergence = {}
            for ch in channels:
                if ch in human_ranks and ch in auto_ranks:
                    divergence[ch] = {
                        'human_rank': human_ranks[ch],
                        'auto_rank': auto_ranks[ch],
                        'rank_diff': auto_ranks[ch] - human_ranks[ch],
                    }

            bc_channels = [ch for ch in ['B', 'C', 'BC'] if ch in channels]
            bc_human_advantage = []
            for ch in bc_channels:
                if ch in divergence:
                    bc_human_advantage.append(divergence[ch]['rank_diff'] > 0)

            hypothesis_supported = sum(bc_human_advantage) > len(bc_human_advantage) / 2 if bc_human_advantage else None

            results['divergence_analysis'] = {
                'channel_divergence': divergence,
                'bc_channels_human_advantage': bc_human_advantage,
                'hypothesis_supported': hypothesis_supported,
                'interpretation': (
                    "SUPPORTED: B/C channels rank better on human-labeled data"
                    if hypothesis_supported else
                    "NOT SUPPORTED: B/C channels do not show clear human-label advantage"
                ) if hypothesis_supported is not None else "Insufficient data"
            }

    return results


# =============================================================================
# ERROR ANALYSIS EXPORT (MULTI-CHANNEL COMPARISON)
# =============================================================================

def export_error_analysis(
    eval_results_with_preds: Dict[str, Dict],
    output_dir: Path,
    datasets: Dict[str, pd.DataFrame] = None,
    top_n: int = 200  # Default expanded to 200 samples
) -> Optional[str]:
    """
    Export comprehensive error analysis for ALL available channels.

    New Features :
    1. 'All_Channels_Compare' sheet: Side-by-side comparison of predictions
       from A, B, C, AB, BC, etc.
    2. Expanded sample size (top_n=200).
    3. Dynamic handling: Works even if only a subset of channels is trained.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed. Skipping Excel export. Install with: pip install openpyxl")
        return None

    excel_path = output_dir / 'error_analysis_qualitative.xlsx'

    # Define priority list for column ordering
    TARGET_CHANNELS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']

    # Collectors for specific analysis sheets
    shift_rows = []        # A failed, B succeeded
    reverse_shift_rows = [] # B failed, A succeeded
    gold_disagreement_rows = [] # Auto-label=1 vs Human-Model=0

    # Collector for the master comparison sheet
    all_channels_rows = []

    for ds_name, ds_results in eval_results_with_preds.items():
        # ---------------------------------------------------------
        # 1. Gather Data & Verify Integrity
        # ---------------------------------------------------------
        texts = ds_results.get('texts', [])
        if not texts:
            continue

        # Extract Ground Truth (y_true) from the first available channel
        # (Labels are consistent across channels for the same dataset)
        y_true = None
        for ch in ds_results:
            if isinstance(ds_results[ch], dict) and 'y_true' in ds_results[ch]:
                y_true = np.array(ds_results[ch]['y_true'])
                break

        if y_true is None:
            continue

        # Collect predictions for all available channels
        channel_preds = {}
        available_channels = []
        for ch in TARGET_CHANNELS:
            if ch in ds_results and 'y_pred' in ds_results[ch]:
                channel_preds[ch] = np.array(ds_results[ch]['y_pred'])
                available_channels.append(ch)

        # ---------------------------------------------------------
        # 2. Build Master Comparison Sheet
        # ---------------------------------------------------------
        # We iterate through samples and log them if ANY channel made a mistake
        # or just log the first top_n samples to give a representative view.

        count = 0
        for i in range(len(texts)):
            if count >= top_n:
                break

            # Check for errors in any channel to prioritize interesting cases
            has_error = False
            current_preds = {}

            for ch in available_channels:
                pred = int(channel_preds[ch][i])
                current_preds[ch] = pred
                if pred != y_true[i]:
                    has_error = True

            # You can remove 'if has_error' if you want random samples,
            # but keeping it focuses the analysis on difficult cases.
            if has_error:
                row_data = {
                    'Dataset': ds_name,
                    'Label_Type': DATASET_INFO.get(ds_name, {}).get('label_type', 'unknown'),
                    'Text_Preview': str(texts[i])[:1000],
                    'True_Label': int(y_true[i])
                }
                # Add columns dynamically
                for ch in available_channels:
                    row_data[f'Pred_{ch}'] = current_preds[ch]

                all_channels_rows.append(row_data)
                count += 1

        # ---------------------------------------------------------
        # 3. Standard Shift Analysis (Lexicon A vs Structure B)
        # ---------------------------------------------------------
        # Only possible if both A and B were trained
        if 'A' in channel_preds and 'B' in channel_preds:
            a_preds = channel_preds['A']
            b_preds = channel_preds['B']

            # Shift: A fails (False Pos/Neg), B succeeds
            mask_shift = (a_preds != y_true) & (b_preds == y_true)
            for idx in np.where(mask_shift)[0][:top_n]:
                shift_rows.append({
                    'Dataset': ds_name,
                    'Text': str(texts[idx])[:1000],
                    'True_Label': int(y_true[idx]),
                    'Pred_A': int(a_preds[idx]),
                    'Pred_B': int(b_preds[idx]),
                    'Analysis': 'Structure (B) Corrected Lexicon (A)'
                })

            # Reverse Shift: B fails, A succeeds
            mask_rev = (a_preds == y_true) & (b_preds != y_true)
            for idx in np.where(mask_rev)[0][:top_n]:
                reverse_shift_rows.append({
                    'Dataset': ds_name,
                    'Text': str(texts[idx])[:1000],
                    'True_Label': int(y_true[idx]),
                    'Pred_A': int(a_preds[idx]),
                    'Pred_B': int(b_preds[idx]),
                    'Analysis': 'Lexicon (A) outperformed Structure'
                })

    # ---------------------------------------------------------
    # 4. Gold Disagreement (Auto vs Human-Trained Model)
    # ---------------------------------------------------------
    # Specifically checks Twitter samples where Auto-Label says Stress (1)
    # but our best human-trained structure model says No Stress (0).
    if datasets is not None and 'twitter' in datasets and 'twitter' in eval_results_with_preds:
        twitter_results = eval_results_with_preds['twitter']

        # Prefer BC (Combined) -> B (Structure) -> C (Style)
        model_human_preds = None
        used_channel = None

        for ch in ['BC', 'B', 'C']:
            if ch in twitter_results and 'y_pred' in twitter_results[ch]:
                model_human_preds = np.array(twitter_results[ch]['y_pred'])
                used_channel = ch
                break

        if model_human_preds is not None:
            # Re-fetch labels from dataset to be sure
            twitter_labels = datasets['twitter']['label'].values
            texts = twitter_results.get('texts', [])

            # Logic: Auto=1 (Stress) BUT Model=0 (No Stress)
            mask_dis = (twitter_labels == 1) & (model_human_preds == 0)

            for idx in np.where(mask_dis)[0][:top_n]:
                 gold_disagreement_rows.append({
                    'Text': str(texts[idx])[:1000],
                    'Auto_Label': 1,
                    'Human_Model_Pred': 0,
                    'Model_Channel': used_channel,
                    'Interpretation': 'Auto label flagged stress, but Human-trained model sees none.'
                })

    # ---------------------------------------------------------
    # 5. Write to Excel
    # ---------------------------------------------------------
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: The new master comparison
        if all_channels_rows:
            pd.DataFrame(all_channels_rows).to_excel(writer, sheet_name='All_Channels_Compare', index=False)

        # Sheet 2: Shift Cases
        if shift_rows:
            pd.DataFrame(shift_rows).to_excel(writer, sheet_name='Shift_Cases_AvsB', index=False)

        # Sheet 3: Reverse Shift
        if reverse_shift_rows:
            pd.DataFrame(reverse_shift_rows).to_excel(writer, sheet_name='Reverse_Shift', index=False)

        # Sheet 4: Gold Disagreement
        if gold_disagreement_rows:
            pd.DataFrame(gold_disagreement_rows).to_excel(writer, sheet_name='Gold_Disagreement', index=False)

        # Sheet 5: Summary Stats
        summary_data = [
            {'Metric': 'Samples Analyzed per Sheet', 'Value': top_n},
            {'Metric': 'Total Shift Cases Found', 'Value': len(shift_rows)},
            {'Metric': 'Total Gold Disagreements', 'Value': len(gold_disagreement_rows)}
        ]
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"   📊 Error analysis exported to: {excel_path}")
    logger.info(f"      - Includes 'All_Channels_Compare' sheet with dynamic columns.")
    logger.info(f"      - Sample limit: {top_n}")

    return str(excel_path)


# =============================================================================
# K-MEANS STYLISTIC PROFILING ( - NEW)
# =============================================================================

def run_style_profiling(
    model: 'TSSClassifier',
    test_df: pd.DataFrame,
    output_dir: Path,
    n_clusters: int = 3,
) -> Optional[Dict]:
    """
    Unsupervised K-Means profiling of stylistic features (Channel C).

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ : STYLISTIC PROFILING                                    ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  Purpose: Prove that the model "understands" stylistic patterns          ║
    ║                                                                          ║
    ║  Method:                                                                 ║
    ║  1. Extract Channel C features for test set                              ║
    ║  2. Apply K-Means clustering (n_clusters=3)                              ║
    ║  3. Analyze cluster centroids to identify distinguishing features        ║
    ║  4. Compute stress prevalence per cluster                                ║
    ║                                                                          ║
    ║  Expected Findings:                                                      ║
    ║  - Cluster with high pronoun usage → high stress prevalence              ║
    ║  - Cluster with formal/diverse vocab → low stress prevalence             ║
    ║  - Cluster with emotional punctuation → medium/high stress               ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Args:
        model: Trained TSSClassifier (must include Channel C)
        test_df: Test DataFrame with labels
        output_dir: Directory for output files
        n_clusters: Number of clusters for K-Means

    Returns:
        Dictionary with profiling results, or None if Channel C not available
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("scikit-learn clustering not available")
        return None

    # Check if model has Channel C
    if model.transformer_ is None or model.transformer_.channel_c_ is None:
        logger.warning("K-Means profiling requires Channel C. Skipping.")
        return None

    logger.info("\n" + "="*80)
    logger.info("K-MEANS STYLISTIC PROFILING")
    logger.info("="*80)

    test_df = test_df.reset_index(drop=True)
    y_true = test_df['label'].astype(int).values

    # Extract Channel C features only
    try:
        c_features = model.transformer_.channel_c_.transform(test_df)
        c_feature_names = model.transformer_.channel_c_.get_feature_names_out()
    except Exception as e:
        logger.error(f"Failed to extract Channel C features: {e}")
        return None

    # Scale features for clustering
    scaler = StandardScaler()
    c_scaled = scaler.fit_transform(c_features)

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(c_scaled)

    # Compute silhouette score
    sil_score = silhouette_score(c_scaled, cluster_labels)
    logger.info(f"   Silhouette Score: {sil_score:.3f}")

    # Analyze clusters
    cluster_profiles = []

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_size = mask.sum()
        stress_prevalence = y_true[mask].mean() if cluster_size > 0 else 0.0

        # Get centroid in original (unscaled) feature space
        centroid_scaled = kmeans.cluster_centers_[cluster_id]
        centroid = scaler.inverse_transform(centroid_scaled.reshape(1, -1))[0]

        # Find top 5 elevated and depressed features compared to overall mean
        overall_mean = c_features.mean(axis=0)
        feature_diff = centroid - overall_mean

        top_elevated_idx = np.argsort(feature_diff)[-5:][::-1]
        top_depressed_idx = np.argsort(feature_diff)[:5]

        elevated_features = [(c_feature_names[i], float(feature_diff[i]))
                            for i in top_elevated_idx]
        depressed_features = [(c_feature_names[i], float(feature_diff[i]))
                             for i in top_depressed_idx]

        profile = {
            'cluster_id': cluster_id,
            'size': int(cluster_size),
            'stress_prevalence': float(stress_prevalence),
            'elevated_features': elevated_features,
            'depressed_features': depressed_features,
        }
        cluster_profiles.append(profile)

        logger.info(f"\n   Cluster {cluster_id}: {cluster_size} samples, "
                   f"stress_prevalence={stress_prevalence:.1%}")
        logger.info(f"      Elevated: {', '.join([f'{n}' for n, _ in elevated_features[:3]])}")
        logger.info(f"      Depressed: {', '.join([f'{n}' for n, _ in depressed_features[:3]])}")

    # Export to CSV
    export_rows = []
    for profile in cluster_profiles:
        for feat_name, feat_diff in profile['elevated_features']:
            export_rows.append({
                'cluster_id': profile['cluster_id'],
                'cluster_size': profile['size'],
                'stress_prevalence': profile['stress_prevalence'],
                'feature_type': 'elevated',
                'feature_name': feat_name,
                'feature_diff': feat_diff,
            })
        for feat_name, feat_diff in profile['depressed_features']:
            export_rows.append({
                'cluster_id': profile['cluster_id'],
                'cluster_size': profile['size'],
                'stress_prevalence': profile['stress_prevalence'],
                'feature_type': 'depressed',
                'feature_name': feat_name,
                'feature_diff': feat_diff,
            })

    csv_path = output_dir / 'kmeans_stylistic_profiles.csv'
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    logger.info(f"\n   📊 Exported: {csv_path}")

    return {
        'silhouette_score': sil_score,
        'n_clusters': n_clusters,
        'cluster_profiles': cluster_profiles,
    }


# =============================================================================
# SHAP EXPLAINABILITY FOR SHIFT CASES ( - NEW)
# =============================================================================

def analyze_shift_cases_with_shap(
    model_a: 'TSSClassifier',
    model_b: 'TSSClassifier',
    test_df: pd.DataFrame,
    output_dir: Path,
    max_cases: int = 50,
) -> Optional[Dict]:
    """
    SHAP analysis of "Shift Cases" where Channel A fails but Channel B succeeds.

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ : SHAP EXPLAINABILITY                                    ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  Purpose: Prove that Channel B relies on STRUCTURAL patterns, not lexical║
    ║                                                                          ║
    ║  Method:                                                                 ║
    ║  1. Identify "Shift Cases": A fails, B succeeds                          ║
    ║  2. Use SHAP LinearExplainer to explain B's correct predictions          ║
    ║  3. Analyze which features drive the correct decision                    ║
    ║                                                                          ║
    ║  Expected Findings:                                                      ║
    ║  - Top features should be structural (PRON_VERB, SVO_PRON_VERB_NOUN)    ║
    ║  - Not lexical content features                                          ║
    ║  - Evidence for "Structure > Content" hypothesis                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Args:
        model_a: Trained TSSClassifier for Channel A
        model_b: Trained TSSClassifier for Channel B
        test_df: Test DataFrame with labels
        output_dir: Directory for output files
        max_cases: Maximum number of shift cases to analyze

    Returns:
        Dictionary with SHAP analysis results, or None if SHAP not available
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None

    logger.info("\n" + "="*80)
    logger.info("SHAP EXPLAINABILITY FOR SHIFT CASES")
    logger.info("="*80)

    test_df = test_df.reset_index(drop=True)
    y_true = test_df['label'].astype(int).values

    # Get predictions from both models
    try:
        y_pred_a = model_a.predict(test_df)
        y_pred_b = model_b.predict(test_df)
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        return None

    # Find shift cases: A fails but B succeeds
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    shift_mask = ~a_correct & b_correct
    shift_indices = np.where(shift_mask)[0]

    n_shift = len(shift_indices)
    logger.info(f"   Found {n_shift} shift cases (A fails, B succeeds)")

    if n_shift == 0:
        logger.warning("   No shift cases found. Skipping SHAP analysis.")
        return None

    # Limit to max_cases
    shift_indices = shift_indices[:max_cases]
    shift_df = test_df.iloc[shift_indices].reset_index(drop=True)

    # Transform features for Channel B
    try:
        X_shift = model_b.transform_features(shift_df)
        feature_names = model_b.get_feature_names()
    except Exception as e:
        logger.error(f"Failed to transform features: {e}")
        return None

    # Get background data for SHAP (random sample)
    n_background = min(500, len(test_df))
    background_df = test_df.sample(n=n_background, random_state=42).reset_index(drop=True)
    X_background = model_b.transform_features(background_df)

    # Create SHAP LinearExplainer
    try:
        explainer = shap.LinearExplainer(
            model_b.classifier_,
            X_background,
            feature_perturbation='interventional',
        )
        shap_values = explainer.shap_values(X_shift)
    except Exception as e:
        logger.error(f"SHAP explainer failed: {e}")
        return None

    # Compute mean absolute SHAP values across shift cases
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Rank features by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    # Log top features
    logger.info(f"\n   Top 10 SHAP-important features for shift cases:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"      {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Identify structural features (POS patterns, SVO triples)
    structural_patterns = [f for f in importance_df.head(20)['feature']
                         if any(p in f for p in ['PRON', 'VERB', 'SVO', 'SV_', 'POS', '_VB', '_JJ', '_NN'])]
    logger.info(f"\n   Structural patterns in top 20: {len(structural_patterns)}")
    for pattern in structural_patterns[:5]:
        logger.info(f"      ✓ {pattern}")

    # Export feature importance
    importance_csv = output_dir / 'shap_feature_importance.csv'
    importance_df.to_csv(importance_csv, index=False)
    logger.info(f"\n   📊 Exported: {importance_csv}")

    # Export detailed shift cases with SHAP values
    detail_rows = []
    for i, idx in enumerate(shift_indices[:max_cases]):
        if i >= len(shap_values):
            break

        text = test_df.iloc[idx].get('cleaned_text', '')[:200]
        true_label = int(y_true[idx])

        # Get top 5 SHAP features for this case
        case_shap = shap_values[i]
        top_idx = np.argsort(np.abs(case_shap))[-5:][::-1]
        top_features = [(feature_names[j], float(case_shap[j])) for j in top_idx]

        detail_rows.append({
            'case_idx': int(idx),
            'true_label': true_label,
            'pred_a': int(y_pred_a[idx]),
            'pred_b': int(y_pred_b[idx]),
            'text_preview': text,
            'top_shap_1': f"{top_features[0][0]}: {top_features[0][1]:.3f}",
            'top_shap_2': f"{top_features[1][0]}: {top_features[1][1]:.3f}",
            'top_shap_3': f"{top_features[2][0]}: {top_features[2][1]:.3f}",
        })

    detail_csv = output_dir / 'shap_shift_cases_detailed.csv'
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)
    logger.info(f"   📊 Exported: {detail_csv}")

    return {
        'n_shift_cases': n_shift,
        'n_analyzed': len(shift_indices),
        'top_features': importance_df.head(20).to_dict('records'),
        'structural_patterns_in_top20': structural_patterns,
    }


# =============================================================================
# ADVANCED ANALYSIS SUITE
# =============================================================================

def run_advanced_analysis_suite(
    models: Dict[str, 'TSSClassifier'],
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Run the complete advanced analysis suite.

    Includes:
    1. K-Means Stylistic Profiling (requires Channel C)
    2. SHAP Explainability for Shift Cases (requires Channels A and B)
    """
    results = {}

    # 1. K-Means profiling (if Channel C available)
    if 'C' in models or 'BC' in models or 'ABC' in models:
        model_with_c = models.get('C') or models.get('BC') or models.get('ABC')
        kmeans_results = run_style_profiling(model_with_c, test_df, output_dir)
        if kmeans_results:
            results['kmeans_profiling'] = kmeans_results

    # 2. SHAP analysis (if Channels A and B available)
    if 'A' in models and 'B' in models:
        shap_results = analyze_shift_cases_with_shap(
            models['A'], models['B'], test_df, output_dir
        )
        if shap_results:
            results['shap_analysis'] = shap_results

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"{SYSTEM_NAME} Training & Evaluation ")
    parser.add_argument('--data', type=str, default='data/processed')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--artifacts', type=str, default='artifacts')
    parser.add_argument('--channels', type=str, default='all')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--no-dedup', action='store_true')
    parser.add_argument('--export_error_analysis', action='store_true', default=True)
    parser.add_argument('--run_advanced_analysis', action='store_true', default=False,
                       help='Run K-Means profiling and SHAP explainability analysis')

    args = parser.parse_args()
    n_bootstrap = args.bootstrap
    remove_duplicates = not args.no_dedup

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.channels.lower() == 'all':
        channels = ALL_CHANNELS
    else:
        channels = [c.strip().upper() for c in args.channels.split(',')]

    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║             TSS Training & Evaluation ()           ║
║                                                                              ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║  "Linguistic structure is the true vessel of psychological state."                           ║
║  "Linguistic structure is the true vessel of psychological state"            ║
║  ═══════════════════════════════════════════════════════════════════════════ ║
║                                                                              ║
║  CRITICAL UPDATES :                                                      ║
║  ├── PRIMARY METRIC: Macro F1 (fairness across datasets)                     ║
║  ├── METRIC BUNDLE: macro_F1, Pos_F1, Pos_Recall, PR_AUC, MCC, Prevalence   ║
║  ├── MAJORITY BASELINE: Printed before model results                         ║
║  ├── NO undersampling: FULL DATA with class_weight='balanced'                ║
║  ├── Twitter Gold: INCLUDED in all evaluations                               ║
║  ├── Channel Ranking by Regime: Table A (Human) vs Table B (Auto)            ║
║  └── Enhanced Error Analysis:                                                ║
║       ├── Shift Cases (A→B)                                                  ║
║       ├── Gold Disagreement (Auto=1, Model=0)                                ║
║       └── Export to Excel for Discussion section                             ║
║                                                                              ║
║                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    logger.info("="*80)
    logger.info(f"{SYSTEM_NAME} v{__version__} — TRAINING & EVALUATION ")
    logger.info("")
    logger.info("="*80)
    logger.info(f"\nBootstrap: {n_bootstrap} | CV folds: {args.cv_folds} | Dedup: {remove_duplicates}")
    logger.info("PRIMARY METRIC: Macro F1 | NO undersampling | class_weight='balanced'")

    # Load data
    logger.info("\n📂 Loading datasets (NO UNDERSAMPLING - FULL DATA)...")
    datasets = load_data(data_dir, remove_duplicates=remove_duplicates)

    if 'dreaddit_train' not in datasets:
        logger.error("dreaddit_train.csv not found!")
        sys.exit(1)

    train_df = datasets['dreaddit_train']

    # Log majority baselines for all evaluation datasets
    logger.info("\n📊 MAJORITY BASELINES:")
    for ds_name in ['dreaddit_test', 'twitter', 'twitter_gold', 'reddit_combi']:
        if ds_name not in datasets:
            continue
        df = datasets[ds_name]
        baseline = compute_majority_baseline(df['label'].values)
        label_type = DATASET_INFO.get(ds_name, {}).get('label_type', 'unknown')
        logger.info(f"   {ds_name} [{label_type}]: majority_F1={baseline['majority_f1']:.3f}, "
                   f"majority_macro_F1={baseline['majority_macro_f1']:.3f}, prev={baseline['prevalence']*100:.1f}%")

    all_models = {}
    all_artifacts = {}
    all_eval_results = []
    all_lodo_results = []
    all_cross_platform = []

    # For error analysis
    eval_results_with_preds = {}

    for channel in channels:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# CHANNEL: {channel} — {CHANNEL_DESCRIPTIONS.get(channel, '')}")
        logger.info('#'*80)

        model, artifact, threshold_opt = train_channel(
            train_df, channel=channel, n_folds=args.cv_folds,
            threshold_strategy='f1', use_svd=False, transfer_mode=True,
        )

        all_models[channel] = model
        all_artifacts[channel] = artifact

        model_path = artifacts_dir / f"tss_{channel.lower()}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"   ✓ Saved: {model_path}")

        # Evaluate
        logger.info(f"\n   📊 Evaluating with METRIC BUNDLE...")
        channel_eval = {}

        # Include twitter_gold in evaluation (human-labeled cross-platform)
        for ds_name in ['dreaddit_test', 'twitter', 'twitter_gold', 'reddit_combi']:
            if ds_name not in datasets:
                continue
            test_df = datasets[ds_name]
            result = evaluate_model(model, test_df)
            y_true = np.array(result['y_true'])
            y_pred = np.array(result['y_pred'])
            y_proba = np.array(result['y_proba'])

            # Compute comprehensive metric bundle
            metrics = compute_metric_bundle(y_true, y_pred, y_proba)

            # Bootstrap CI for macro F1 (PRIMARY METRIC)
            macro_f1_ci = bootstrap_f1(y_true, y_pred, n_iterations=n_bootstrap, metric='macro_f1')
            f1_ci = bootstrap_f1(y_true, y_pred, n_iterations=n_bootstrap, metric='f1')

            # Get dataset info
            ds_info = DATASET_INFO.get(ds_name, {})
            label_type = ds_info.get('label_type', 'unknown')

            entry = {
                'channel': channel,
                'dataset': ds_name,
                'label_type': label_type,
                'platform': ds_info.get('platform', 'unknown'),
                'n_samples': result['n_samples'],
                'threshold': result['threshold'],

                # PRIMARY METRIC
                'macro_f1': metrics['macro_f1'],
                'macro_f1_ci_lower': macro_f1_ci['ci_lower'],
                'macro_f1_ci_upper': macro_f1_ci['ci_upper'],

                # Positive class metrics
                'f1': metrics['f1'],
                'f1_ci_lower': f1_ci['ci_lower'],
                'f1_ci_upper': f1_ci['ci_upper'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],

                # Other metrics
                'mcc': metrics['mcc'],
                'pr_auc': metrics['pr_auc'],
                'roc_auc': metrics['roc_auc'],
                'balanced_accuracy': metrics['balanced_accuracy'],

                # Dataset info
                'prevalence': metrics['prevalence'],
                'n_positive': metrics['n_positive'],

                # Baselines
                'majority_f1': metrics['majority_f1'],
                'majority_macro_f1': metrics['majority_macro_f1'],
                'delta_f1': metrics['delta_f1'],
                'delta_macro_f1': metrics['delta_macro_f1'],
            }
            all_eval_results.append(entry)
            channel_eval[ds_name] = {**result, **metrics}

            # Store for error analysis
            if ds_name not in eval_results_with_preds:
                eval_results_with_preds[ds_name] = {
                    'texts': test_df['cleaned_text'].tolist(),
                }
            eval_results_with_preds[ds_name][channel] = {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
            }

            logger.info(f"      {ds_name:<15} [{label_type}] "
                       f"macro_F1={metrics['macro_f1']:.3f} [{macro_f1_ci['ci_lower']:.3f}, {macro_f1_ci['ci_upper']:.3f}] "
                       f"F1={metrics['f1']:.3f} Recall={metrics['recall']:.3f} MCC={metrics['mcc']:.3f} "
                       f"Δmacro_F1={metrics['delta_macro_f1']:+.3f}")

        # LODO
        logger.info(f"\n   🔄 LODO (nested 3-way split)...")
        lodo_results = run_lodo(train_df, channel, n_bootstrap=n_bootstrap)
        all_lodo_results.extend(lodo_results)

        # Cross-platform
        cp_results = compute_cross_platform(channel_eval, 'dreaddit_test')
        for cp in cp_results:
            cp['channel'] = channel
        all_cross_platform.extend(cp_results)

    # ==========================================================================
    # ERROR ANALYSIS EXPORT (ENHANCED )
    # ==========================================================================
    if args.export_error_analysis:
        logger.info("\n" + "="*80)
        logger.info("ERROR ANALYSIS EXPORT ( Enhanced)")
        logger.info("="*80)
        export_error_analysis(eval_results_with_preds, output_dir, datasets=datasets)

    # ==========================================================================
    # ADVANCED ANALYSIS SUITE ( - K-Means + SHAP)
    # ==========================================================================
    advanced_results = {}
    if args.run_advanced_analysis:
        # Use dreaddit_test as the primary test set for advanced analysis
        test_df_for_analysis = datasets.get('dreaddit_test')
        if test_df_for_analysis is not None:
            advanced_results = run_advanced_analysis_suite(
                all_models, test_df_for_analysis, output_dir
            )
        else:
            logger.warning("dreaddit_test not available for advanced analysis")

    # ==========================================================================
    # CHANNEL RANKING BY REGIME (replaces Friedman)
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("CHANNEL RANKING BY REGIME (Divergence Hypothesis)")
    logger.info("="*80)

    ranking_results = run_channel_ranking_by_regime(all_eval_results)

    # Table A: Human Labels
    table_a = ranking_results.get('table_a_human')
    if table_a:
        logger.info("\n📊 TABLE A: Human-Labeled Datasets")
        logger.info(f"   Datasets: {table_a.get('datasets', [])}")
        if table_a.get('sorted_rankings'):
            logger.info("   Channel Rankings (lower = better):")
            for ch, rank in table_a['sorted_rankings']:
                logger.info(f"      {ch}: {rank:.2f}")

    # Table B: Auto Labels
    table_b = ranking_results.get('table_b_auto')
    if table_b:
        logger.info("\n📊 TABLE B: Auto-Labeled Datasets")
        logger.info(f"   Datasets: {table_b.get('datasets', [])}")
        if table_b.get('sorted_rankings'):
            logger.info("   Channel Rankings (lower = better):")
            for ch, rank in table_b['sorted_rankings']:
                logger.info(f"      {ch}: {rank:.2f}")

    # Divergence Analysis
    divergence = ranking_results.get('divergence_analysis')
    if divergence:
        logger.info("\n📊 DIVERGENCE ANALYSIS:")
        logger.info(f"   {divergence.get('interpretation', 'N/A')}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_json = {
        'system': SYSTEM_NAME,
        'version': __version__,
        'script_version': '7.0.0',
        'timestamp': timestamp,
        'channels': channels,
        'n_bootstrap': n_bootstrap,
        'duplicates_removed': remove_duplicates,
        'primary_metric': 'macro_f1',
        'config': {
            'no_undersampling': True,
            'class_weight': 'balanced',
            'elasticnet_for_channel_b': True,
            'philosophy': 'Structural Synergy - HOW > WHAT',
        },
        'evaluation_results': all_eval_results,
        'lodo_results': all_lodo_results,
        'cross_platform': all_cross_platform,
        'channel_rankings': ranking_results,
        'advanced_analysis': advanced_results,  # K-Means + SHAP
        'training_artifacts': {ch: asdict(art) for ch, art in all_artifacts.items()},
    }

    json_path = output_dir / f"tss_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    pd.DataFrame(all_eval_results).to_csv(output_dir / f"tss_eval_{timestamp}.csv", index=False)
    pd.DataFrame(all_lodo_results).to_csv(output_dir / f"tss_lodo_{timestamp}.csv", index=False)
    pd.DataFrame(all_cross_platform).to_csv(output_dir / f"tss_cross_{timestamp}.csv", index=False)

    # ==========================================================================
    # COMPREHENSIVE SUMMARY
    # ==========================================================================
    logger.info("\n" + "="*100)
    logger.info("RESULTS SUMMARY ( - )")
    logger.info("="*100)

    logger.info(f"\n{'Chan':<6} {'Dataset':<16} {'Type':<6} {'macro_F1':>9} {'F1':>6} {'Recall':>7} "
               f"{'MCC':>6} {'PR-AUC':>7} {'Prev':>6} {'Δmacro':>7}")
    logger.info("-"*90)

    for r in sorted(all_eval_results, key=lambda x: (x['dataset'], -x['macro_f1'])):
        label_type = r.get('label_type', 'unk')[:5]
        logger.info(f"{r['channel']:<6} {r['dataset']:<16} {label_type:<6} "
                   f"{r['macro_f1']:>9.3f} {r['f1']:>6.3f} {r['recall']:>7.3f} "
                   f"{r['mcc']:>6.3f} {r['pr_auc']:>7.3f} {r['prevalence']*100:>5.1f}% "
                   f"{r['delta_macro_f1']:>+7.3f}")

    # Label regime analysis
    logger.info("\n" + "="*80)
    logger.info("LABEL REGIME ANALYSIS (Human vs Auto Labels)")
    logger.info("="*80)

    human_results = [r for r in all_eval_results if r.get('label_type') == 'human']
    auto_results = [r for r in all_eval_results if r.get('label_type') == 'auto']

    if human_results:
        human_macro_f1s = [r['macro_f1'] for r in human_results]
        logger.info(f"\n📋 HUMAN-LABELED: mean macro_F1 = {np.mean(human_macro_f1s):.3f} ± {np.std(human_macro_f1s):.3f}")
        for r in sorted(human_results, key=lambda x: (x['channel'], x['dataset'])):
            logger.info(f"   {r['channel']:<6} {r['dataset']:<15} macro_F1={r['macro_f1']:.3f} MCC={r['mcc']:.3f}")

    if auto_results:
        auto_macro_f1s = [r['macro_f1'] for r in auto_results]
        logger.info(f"\n🤖 AUTO-LABELED: mean macro_F1 = {np.mean(auto_macro_f1s):.3f} ± {np.std(auto_macro_f1s):.3f}")
        for r in sorted(auto_results, key=lambda x: (x['channel'], x['dataset'])):
            logger.info(f"   {r['channel']:<6} {r['dataset']:<15} macro_F1={r['macro_f1']:.3f} MCC={r['mcc']:.3f}")

    # Cross-platform transfer
    if all_cross_platform:
        logger.info("\n" + "="*80)
        logger.info("CROSS-PLATFORM TRANSFER (macro F1 retention)")
        logger.info("="*80)
        logger.info(f"\n{'Chan':<6} {'Target':<15} {'Type':>6} {'Src_mF1':>8} {'Tgt_mF1':>8} {'Retain':>8}")
        logger.info("-"*60)
        for cp in sorted(all_cross_platform, key=lambda x: -x.get('retention_rate_macro', 0)):
            label_type = cp.get('target_label_type', 'UNK')[:5]
            logger.info(f"{cp['channel']:<6} {cp['target']:<15} {label_type:>6} "
                       f"{cp.get('source_macro_f1', 0):>8.3f} {cp.get('target_macro_f1', 0):>8.3f} "
                       f"{cp.get('retention_rate_macro', 0):>7.1f}%")

    logger.info(f"\n📄 Results: {json_path}")
    logger.info("="*80)
    logger.info("✅ Training and evaluation complete!")

if __name__ == "__main__":
    main()
