#!/usr/bin/env python3
"""
================================================================================
TSS: Statistical Analysis v3.0  —  Instance-Level
================================================================================
UPGRADE from v2.0:
  All key statistical tests now operate on INSTANCE-LEVEL predictions
  (y_true, y_pred, y_prob per sample) rather than 4 aggregate scores.

  N = 4 dataset means  →  N = 12,000+ individual samples

DATA SOURCE:
  - Primary:  predictions_*.joblib  (saved by 03_masking_suite.py)
  - Fallback: masking_comprehensive_*.json  (aggregate scores only — warns)

HYPOTHESIS:
  Human labels align better with style/structure channels (B, C)
  while auto labels align with lexical surface channel (A).

KEY ANALYSES (all instance-level where applicable):
  1. Channel Orthogonality  — Spearman on per-sample y_prob  (N=samples)
  2. Label Regime Divergence — aggregate + instance-level bootstrap
  3. Degree of Divergence    — bootstrap resampling predictions  (N=samples)
  4. Lexical Interference    — paired bootstrap on predictions   (N=samples)
  5. McNemar's Test          — paired classifier comparison      (N=samples)
  6. Permutation Regime Div  — shuffle sample predictions        (N=samples)
  7. Effect Sizes            — Cohen's d per regime
  8. Relative Domain Robustness
  9. Global FDR Correction   — Benjamini-Hochberg

Usage:
  python scripts/04_statistical_analysis.py \
      --predictions outputs/predictions_*.joblib \
      --results outputs/masking_comprehensive_*.json \
      --output outputs

================================================================================
"""

import argparse
import json
import sys
import glob
import os
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Import shared dataset registry (single source of truth with auto-detection)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from tss.dataset_registry import (
    DATASET_INFO, get_dataset_info, get_eval_datasets,
    register_dataset, infer_label_type, infer_platform,
)

LEXICAL_CHANNELS   = ['A']
STRUCTURE_CHANNELS = ['B', 'C', 'BC']
MIXED_CHANNELS     = ['AB', 'AC', 'ABC']
ALL_CHANNELS       = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']


def _rebuild_eval_lists():
    """Rebuild EVAL/HUMAN/AUTO dataset lists from current DATASET_INFO."""
    global EVAL_DATASETS, HUMAN_EVAL, AUTO_EVAL
    all_ds, human, auto = get_eval_datasets(exclude_train=True)
    EVAL_DATASETS = all_ds
    HUMAN_EVAL    = human
    AUTO_EVAL     = auto


# Initialize default lists
EVAL_DATASETS: List[str] = []
HUMAN_EVAL:    List[str] = []
AUTO_EVAL:     List[str] = []
_rebuild_eval_lists()


def discover_datasets(predictions_data: list) -> int:
    """
    Auto-discover datasets from loaded predictions and update DATASET_INFO.

    Each prediction bundle contains 'dataset' and optionally 'label_type'.
    - If label_type is present in the bundle → use it directly.
    - If label_type is missing → infer from dataset name heuristics.
    - Unknown datasets are auto-registered and EVAL lists rebuilt.

    Returns:
        Number of newly discovered datasets.
    """
    discovered = 0
    for item in predictions_data:
        ds = item.get('dataset', '')
        if not ds or ds.endswith('_train'):
            continue

        # Get label_type: from bundle first, then from registry/auto-detect
        lt = item.get('label_type', '')
        if not lt or lt == 'unknown':
            lt = infer_label_type(ds)

        if not lt or lt == 'unknown':
            continue

        if ds not in DATASET_INFO:
            pf = infer_platform(ds)
            register_dataset(ds, lt, pf)
            discovered += 1
        elif DATASET_INFO[ds].get('label_type') == 'unknown':
            DATASET_INFO[ds]['label_type'] = lt
            logger.info(f"   Updated dataset '{ds}' label_type -> {lt}")
            discovered += 1

    if discovered > 0:
        _rebuild_eval_lists()
        logger.info(f"   {discovered} new/updated dataset(s). "
                    f"EVAL={len(EVAL_DATASETS)}, HUMAN={len(HUMAN_EVAL)}, AUTO={len(AUTO_EVAL)}")
    return discovered


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(predictions_file: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load instance-level predictions from joblib.

    Supports two formats:
      - Envelope (v3.0+): dict with '_meta' header and 'predictions' list
      - Legacy: plain list of prediction bundles

    Validates integrity, auto-discovers datasets, and rebuilds EVAL lists.

    Returns:
        structured[dataset][channel] = {
            'y_true': np.array, 'y_pred': np.array, 'y_prob': np.array|None,
            'label_type': str
        }

    Raises:
        SystemExit: if file is corrupt, empty, or critically inconsistent
    """
    # ── File size check ──
    file_size_mb = predictions_file.stat().st_size / (1024 * 1024)
    if file_size_mb < 0.001:
        logger.error(f"   Predictions file is empty or near-empty ({file_size_mb:.4f} MB)")
        logger.error("   Re-run 03_masking_suite.py to regenerate predictions.")
        sys.exit(1)

    # ── Load ──
    try:
        loaded = joblib.load(predictions_file)
    except Exception as e:
        logger.error(f"   Predictions file corrupt or unreadable: {e}")
        logger.error("   Re-run 03_masking_suite.py to regenerate predictions.")
        sys.exit(1)

    # ── Parse envelope vs legacy format ──
    meta = None
    if isinstance(loaded, dict) and '_meta' in loaded:
        meta = loaded['_meta']
        raw = loaded.get('predictions', [])
        logger.info(f"   Format: envelope v{meta.get('version', '?')} "
                    f"(saved {meta.get('timestamp', '?')})")
    elif isinstance(loaded, list):
        raw = loaded
        logger.info(f"   Format: legacy (plain list)")
    else:
        logger.error(f"   Unexpected format: {type(loaded).__name__}")
        sys.exit(1)

    if not raw:
        logger.error(f"   Predictions file contains no bundles!")
        sys.exit(1)

    # ── Validate required fields ──
    required_fields = {'dataset', 'channel', 'y_true', 'y_pred'}
    sample = raw[0]
    missing = required_fields - set(sample.keys())
    if missing:
        logger.error(f"   Bundles missing required fields: {missing}")
        sys.exit(1)

    # ── Cross-validate with metadata envelope ──
    n_bundles_actual = len(raw)
    n_samples_actual = sum(len(p['y_true']) for p in raw)

    if meta:
        n_bundles_expected = meta.get('n_bundles', n_bundles_actual)
        n_samples_expected = meta.get('n_samples', n_samples_actual)
        if n_bundles_actual != n_bundles_expected or n_samples_actual != n_samples_expected:
            logger.error(f"   INTEGRITY MISMATCH: meta says {n_bundles_expected} bundles/"
                         f"{n_samples_expected:,} samples, but file contains "
                         f"{n_bundles_actual}/{n_samples_actual:,}")
            logger.error("   File may have been truncated. Re-run 03_masking_suite.py.")
            sys.exit(1)
        logger.info(f"   Integrity: ✓ metadata matches ({n_bundles_actual} bundles, "
                    f"{n_samples_actual:,} samples)")

    logger.info(f"   File: {predictions_file.name} ({file_size_mb:.1f} MB)")

    # ── Auto-discover new datasets ──
    n_new = discover_datasets(raw)

    # ── Structure data ──
    structured = defaultdict(dict)
    n_samples_total = 0
    n_skipped_unknown = 0

    for item in raw:
        mm = item.get('mask_mode', 'none')
        if mm != 'none':
            continue

        ds = item['dataset']
        ch = item['channel']
        lt = item.get('label_type',
                       get_dataset_info(ds).get('label_type', 'unknown'))

        if lt == 'unknown':
            n_skipped_unknown += 1
            continue

        structured[ds][ch] = {
            'y_true': np.asarray(item['y_true']),
            'y_pred': np.asarray(item['y_pred']),
            'y_prob': np.asarray(item['y_prob']) if item.get('y_prob') is not None else None,
            'label_type': lt,
        }
        n_samples_total += len(item['y_true'])

    if n_skipped_unknown > 0:
        logger.warning(f"   Skipped {n_skipped_unknown} bundles with label_type='unknown'. "
                       f"Add them to tss/dataset_registry.py or ensure label_type is set in 03.")

    if n_samples_total == 0:
        logger.error("   No valid samples after filtering. Check dataset names and label_types.")
        sys.exit(1)

    logger.info(f"   Structured: {sum(len(v) for v in structured.values())} (ds, ch) pairs | "
                f"{n_samples_total:,} sample predictions | "
                f"Datasets: {sorted(structured.keys())}")
    return dict(structured)


def load_aggregate_results(results_file: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load aggregate JSON results (fallback / supplementary data)."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    eval_results = data.get('results', data.get('evaluation_results', []))
    cross_platform = data.get('cross_platform', data.get('transfer_results', []))
    return eval_results, cross_platform


def build_score_matrix_from_predictions(preds: Dict) -> pd.DataFrame:
    """Build (dataset x channel) Macro-F1 matrix from instance-level predictions."""
    rows = {}
    for ds, channels in preds.items():
        rows[ds] = {}
        for ch, data in channels.items():
            mf1 = f1_score(data['y_true'], data['y_pred'], average='macro', zero_division=0)
            rows[ds][ch] = float(mf1)
    return pd.DataFrame(rows).T


def build_score_matrix_from_json(eval_results: List[Dict]) -> pd.DataFrame:
    """Build (dataset x channel) Macro-F1 matrix from aggregate JSON."""
    rows = defaultdict(dict)
    for r in eval_results:
        ds = r.get('dataset', '')
        ch = r.get('channel', '')
        if ds in EVAL_DATASETS and ch in ALL_CHANNELS:
            rows[ds][ch] = r.get('macro_f1', 0)
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: FDR CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_fdr_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return {'corrected': {}, 'n_rejected': 0, 'n_tests': 0, 'method': 'BH'}

    names = list(p_values.keys())
    pvals = np.array([p_values[n] for n in names])
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]

    adjusted = np.zeros(n)
    adjusted[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    result = {}
    for j, i in enumerate(idx):
        result[names[i]] = {
            'p_original': float(pvals[i]),
            'p_adjusted': float(adjusted[j]),
            'reject_h0': bool(adjusted[j] < alpha),
        }

    return {
        'corrected': result,
        'n_rejected': sum(1 for v in result.values() if v['reject_h0']),
        'n_tests': n,
        'method': 'benjamini-hochberg',
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: COHEN'S D
# ─────────────────────────────────────────────────────────────────────────────

def compute_cohens_d(group1, group2):
    """Cohen's d effect size (pooled SD)."""
    g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    return float((g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0.0


# =============================================================================
# SECTION 1: CHANNEL ORTHOGONALITY  (INSTANCE-LEVEL)
# =============================================================================

def analyze_channel_orthogonality(preds: Dict) -> Dict:
    """
    Compute Spearman correlation between channel y_prob vectors at the
    SAMPLE level (N = thousands of predictions per dataset).

    Previously: correlation across 4 dataset-level means.
    Now: correlation across thousands of individual sample probabilities.
    """
    logger.info("Computing instance-level channel orthogonality ...")

    all_correlations = []
    per_dataset = {}

    for ds_name, channels in preds.items():
        if ds_name not in EVAL_DATASETS:
            continue
        ch_names = sorted(channels.keys())
        ds_corrs = {}

        for c1, c2 in combinations(ch_names, 2):
            p1 = channels[c1].get('y_prob')
            p2 = channels[c2].get('y_prob')
            if p1 is None or p2 is None:
                continue
            if len(p1) != len(p2):
                continue

            rho, p_val = spearmanr(p1, p2)
            entry = {
                'dataset': ds_name,
                'pair': f"{c1}_vs_{c2}",
                'rho': float(rho),
                'p_value': float(p_val),
                'n_samples': len(p1),
                'orthogonal': abs(rho) < 0.3,
            }
            all_correlations.append(entry)
            ds_corrs[f"{c1}_vs_{c2}"] = entry

        per_dataset[ds_name] = ds_corrs

    if not all_correlations:
        return {'status': 'no_y_prob_available'}

    # Aggregate across datasets for each pair
    pair_agg = defaultdict(list)
    for c in all_correlations:
        pair_agg[c['pair']].append(c)

    aggregated = {}
    for pair, entries in pair_agg.items():
        rhos = [e['rho'] for e in entries]
        ds_rhos = {e['dataset']: e['rho'] for e in entries}
        aggregated[pair] = {
            'mean_rho': float(np.mean(rhos)),
            'rhos_per_dataset': ds_rhos,
            'orthogonal': abs(np.mean(rhos)) < 0.3,
            'n_datasets': len(rhos),
        }

    # Primary A-B-C pairs
    primary_pairs = {}
    for pname in ['A_vs_B', 'A_vs_C', 'B_vs_C']:
        if pname in aggregated:
            primary_pairs[pname] = aggregated[pname]

    all_rhos = [c['rho'] for c in all_correlations]
    mean_abs_rho = float(np.mean(np.abs(all_rhos)))
    n_total = len(all_correlations)
    n_ortho = sum(1 for c in all_correlations if c['orthogonal'])
    total_samples = sum(c['n_samples'] for c in all_correlations)

    if mean_abs_rho < 0.3:
        interp = f"Strong orthogonality (mean|rho|={mean_abs_rho:.3f}, N={total_samples:,} samples)"
    elif mean_abs_rho < 0.5:
        interp = f"Moderate orthogonality (mean|rho|={mean_abs_rho:.3f}, N={total_samples:,} samples)"
    else:
        interp = f"Weak orthogonality (mean|rho|={mean_abs_rho:.3f}, N={total_samples:,} samples)"

    return {
        'method': 'instance_level_spearman',
        'total_sample_comparisons': total_samples,
        'all_correlations': all_correlations,
        'aggregated_by_pair': aggregated,
        'primary_pairs_ABC': primary_pairs,
        'mean_abs_rho': mean_abs_rho,
        'ortho_ratio': n_ortho / n_total if n_total else 0,
        'per_dataset': per_dataset,
        'interpretation': interp,
    }


# =============================================================================
# SECTION 2: LABEL REGIME DIVERGENCE
# =============================================================================

def analyze_label_regime_divergence(preds: Dict, score_matrix: pd.DataFrame) -> Dict:
    """
    Analyze per-channel regime gaps using both aggregate and instance data.
    """
    logger.info("Computing label regime divergence ...")

    channel_analysis = {}
    for ch in ALL_CHANNELS:
        human_f1s, auto_f1s = [], []
        for ds in HUMAN_EVAL:
            if ds in preds and ch in preds[ds]:
                d = preds[ds][ch]
                mf1 = f1_score(d['y_true'], d['y_pred'], average='macro', zero_division=0)
                human_f1s.append(float(mf1))
        for ds in AUTO_EVAL:
            if ds in preds and ch in preds[ds]:
                d = preds[ds][ch]
                mf1 = f1_score(d['y_true'], d['y_pred'], average='macro', zero_division=0)
                auto_f1s.append(float(mf1))

        if human_f1s and auto_f1s:
            hm, am = np.mean(human_f1s), np.mean(auto_f1s)
            channel_analysis[ch] = {
                'human_mean': float(hm), 'auto_mean': float(am),
                'regime_gap': float(hm - am),
                'human_advantage': bool(hm > am),
                'human_f1s': human_f1s, 'auto_f1s': auto_f1s,
            }

    # Rank within each regime
    human_ranking = sorted(channel_analysis.items(), key=lambda x: -x[1]['human_mean'])
    auto_ranking  = sorted(channel_analysis.items(), key=lambda x: -x[1]['auto_mean'])

    rank_divergence = {}
    for ch in channel_analysis:
        h_rank = next(i + 1 for i, (c, _) in enumerate(human_ranking) if c == ch)
        a_rank = next(i + 1 for i, (c, _) in enumerate(auto_ranking) if c == ch)
        rank_divergence[ch] = {
            'human_rank': h_rank, 'auto_rank': a_rank,
            'rank_shift': a_rank - h_rank,
        }

    # Hypothesis: structure channels have larger regime gap than lexical
    struct_gaps = [channel_analysis[ch]['regime_gap'] for ch in STRUCTURE_CHANNELS
                   if ch in channel_analysis]
    lex_gaps    = [channel_analysis[ch]['regime_gap'] for ch in LEXICAL_CHANNELS
                   if ch in channel_analysis]

    hypothesis_supported = False
    if struct_gaps and lex_gaps:
        hypothesis_supported = bool(np.mean(struct_gaps) > np.mean(lex_gaps))

    return {
        'channel_analysis': channel_analysis,
        'rank_divergence': rank_divergence,
        'human_ranking': [(c, v['human_mean']) for c, v in human_ranking],
        'auto_ranking':  [(c, v['auto_mean'])  for c, v in auto_ranking],
        'hypothesis_supported': hypothesis_supported,
        'interpretation': (
            "SUPPORTED: B/C/BC gap > A gap"
            if hypothesis_supported else "NOT SUPPORTED"
        ),
    }


# =============================================================================
# SECTION 3: DEGREE OF DIVERGENCE (DoD) — INSTANCE-LEVEL BOOTSTRAP
# =============================================================================

def compute_dod_bootstrap_instance(
    preds: Dict,
    channel_c: str,
    channel_base: str = 'A',
    n_boot: int = 10000,
) -> Dict:
    """
    Instance-level DoD bootstrap.

    For each bootstrap iteration:
      1. Resample (with replacement) the indices within each dataset
      2. Recompute Macro-F1 for both channel_c and channel_base on resampled data
      3. Compute delta = F1(channel_c) - F1(channel_base)
      4. Average deltas within human regime and auto regime
      5. DoD = mean(human_deltas) - mean(auto_deltas)

    This gives N = total_samples bootstrap power, not N = 2 per regime.
    """
    # Collect data bundles
    human_bundles = []  # list of (ds_name, y_true, y_pred_c, y_pred_base)
    auto_bundles  = []

    for ds_name, channels in preds.items():
        if ds_name not in EVAL_DATASETS:
            continue
        if channel_c not in channels or channel_base not in channels:
            continue

        y_true   = channels[channel_c]['y_true']
        y_pred_c = channels[channel_c]['y_pred']
        y_pred_b = channels[channel_base]['y_pred']

        lt = channels[channel_c].get('label_type',
             get_dataset_info(ds_name).get('label_type', 'unknown'))

        bundle = (ds_name, y_true, y_pred_c, y_pred_b)
        if lt == 'human':
            human_bundles.append(bundle)
        else:
            auto_bundles.append(bundle)

    if not human_bundles or not auto_bundles:
        return {'status': 'insufficient_data', 'comparison': f'{channel_c}_vs_{channel_base}'}

    # Point estimate
    def compute_regime_delta(bundles):
        deltas = []
        for _, yt, pc, pb in bundles:
            f1_c = f1_score(yt, pc, average='macro', zero_division=0)
            f1_b = f1_score(yt, pb, average='macro', zero_division=0)
            deltas.append(f1_c - f1_b)
        return np.mean(deltas)

    delta_human = float(compute_regime_delta(human_bundles))
    delta_auto  = float(compute_regime_delta(auto_bundles))
    dod_point   = delta_human - delta_auto

    # Bootstrap
    rng = np.random.RandomState(RANDOM_STATE)
    boot_dods = np.zeros(n_boot)

    for b in range(n_boot):
        h_deltas = []
        for _, yt, pc, pb in human_bundles:
            n = len(yt)
            idx = rng.randint(0, n, size=n)
            f1_c = f1_score(yt[idx], pc[idx], average='macro', zero_division=0)
            f1_b = f1_score(yt[idx], pb[idx], average='macro', zero_division=0)
            h_deltas.append(f1_c - f1_b)

        a_deltas = []
        for _, yt, pc, pb in auto_bundles:
            n = len(yt)
            idx = rng.randint(0, n, size=n)
            f1_c = f1_score(yt[idx], pc[idx], average='macro', zero_division=0)
            f1_b = f1_score(yt[idx], pb[idx], average='macro', zero_division=0)
            a_deltas.append(f1_c - f1_b)

        boot_dods[b] = np.mean(h_deltas) - np.mean(a_deltas)

    ci_lo = float(np.percentile(boot_dods, 2.5))
    ci_hi = float(np.percentile(boot_dods, 97.5))
    p_val = float(np.mean(boot_dods <= 0))  # one-tailed: H1 = DoD > 0
    se    = float(np.std(boot_dods))
    d_eff = float(dod_point / se) if se > 0 else 0.0

    total_n = sum(len(b[1]) for b in human_bundles + auto_bundles)

    return {
        'comparison': f'{channel_c}_vs_{channel_base}',
        'method': 'instance_bootstrap',
        'delta_human': delta_human,
        'delta_auto': delta_auto,
        'dod': dod_point,
        'ci_95': [ci_lo, ci_hi],
        'p_value': p_val,
        'cohens_d': d_eff,
        'se': se,
        'n_boot': n_boot,
        'n_samples': total_n,
        'n_human_datasets': len(human_bundles),
        'n_auto_datasets': len(auto_bundles),
        'significant': ci_lo > 0 or ci_hi < 0,
    }


def run_all_dod_comparisons(preds: Dict, n_boot: int = 10000) -> Dict:
    """Run DoD for all channel pairs vs baseline A."""
    results = {}
    all_pv = {}
    for ch in ['BC', 'C', 'B', 'AB', 'AC', 'ABC']:
        res = compute_dod_bootstrap_instance(preds, channel_c=ch, channel_base='A', n_boot=n_boot)
        results[f"{ch}_vs_A"] = res
        if 'p_value' in res and res.get('status') != 'insufficient_data':
            all_pv[f"dod_{ch}_vs_A"] = res['p_value']
    if all_pv:
        results['fdr_correction'] = apply_fdr_correction(all_pv)
    return results


# =============================================================================
# SECTION 4: LEXICAL INTERFERENCE  (INSTANCE-LEVEL)
# =============================================================================

def test_lexical_interference(preds: Dict, n_boot: int = 10000) -> Dict:
    """
    Test Lexical Interference at instance level.

    C -> AC: does adding lexical channel A degrade human-label performance?
    Bootstrap: resample predictions, recompute F1(C) and F1(AC), test if F1(C) > F1(AC).
    """
    results = {}
    test_pairs = [('C', 'AC', 'c_to_ac'), ('BC', 'ABC', 'bc_to_abc')]

    for ch_pure, ch_mixed, label in test_pairs:
        for regime_name, ds_list in [('human', HUMAN_EVAL), ('auto', AUTO_EVAL)]:
            regime_bundles = []
            for ds in ds_list:
                if ds in preds and ch_pure in preds[ds] and ch_mixed in preds[ds]:
                    y_true = preds[ds][ch_pure]['y_true']
                    y_pred_pure  = preds[ds][ch_pure]['y_pred']
                    y_pred_mixed = preds[ds][ch_mixed]['y_pred']
                    regime_bundles.append((ds, y_true, y_pred_pure, y_pred_mixed))

            if not regime_bundles:
                continue

            # Point estimates
            drops = {}
            for ds, yt, pp, pm in regime_bundles:
                f1_pure  = f1_score(yt, pp, average='macro', zero_division=0)
                f1_mixed = f1_score(yt, pm, average='macro', zero_division=0)
                drops[ds] = float(f1_pure - f1_mixed)

            mean_drop = float(np.mean(list(drops.values())))

            # Bootstrap the mean drop
            rng = np.random.RandomState(RANDOM_STATE)
            boot_drops = np.zeros(n_boot)
            for b in range(n_boot):
                b_drops = []
                for _, yt, pp, pm in regime_bundles:
                    n = len(yt)
                    idx = rng.randint(0, n, size=n)
                    f1_p = f1_score(yt[idx], pp[idx], average='macro', zero_division=0)
                    f1_m = f1_score(yt[idx], pm[idx], average='macro', zero_division=0)
                    b_drops.append(f1_p - f1_m)
                boot_drops[b] = np.mean(b_drops)

            ci_lo = float(np.percentile(boot_drops, 2.5))
            ci_hi = float(np.percentile(boot_drops, 97.5))
            p_drop = float(np.mean(boot_drops <= 0))  # H1: drop > 0

            total_n = sum(len(b[1]) for b in regime_bundles)

            results[f"{label}_{regime_name}"] = {
                'drops': drops,
                'mean_drop': mean_drop,
                'ci_95': [ci_lo, ci_hi],
                'p_value': p_drop,
                'interference_present': bool(mean_drop > 0 and ci_lo > -0.01),
                'n_samples': total_n,
                'method': 'instance_bootstrap',
            }

    # Summary
    h_key = 'c_to_ac_human'
    a_key = 'c_to_ac_auto'
    h_interf = results.get(h_key, {}).get('interference_present', None)
    a_interf = results.get(a_key, {}).get('interference_present', None)

    asymmetric = (h_interf is True and a_interf is not True)

    results['summary'] = {
        'human_interference': h_interf,
        'auto_interference': a_interf,
        'asymmetric': asymmetric,
        'interpretation': (
            "Lexical Interference CONFIRMED: adding A degrades C on human but not auto"
            if asymmetric else
            "Lexical Interference pattern partially observed"
        ),
    }
    return results


# =============================================================================
# SECTION 5: McNEMAR'S TEST  (INSTANCE-LEVEL)
# =============================================================================

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
    """McNemar's test comparing two classifiers on the same test set."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # b01: A wrong, B right;  b10: A right, B wrong
    b01 = int(np.sum(~correct_a & correct_b))
    b10 = int(np.sum(correct_a & ~correct_b))

    n_discordant = b01 + b10
    if n_discordant == 0:
        return {'statistic': 0, 'p_value': 1.0, 'b01': b01, 'b10': b10, 'n_samples': len(y_true)}

    # Continuity-corrected McNemar
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_val = float(1 - stats.chi2.cdf(stat, df=1))

    return {
        'statistic': float(stat), 'p_value': p_val,
        'b01': b01, 'b10': b10, 'n_discordant': n_discordant,
        'n_samples': len(y_true),
        'better': 'B' if b01 > b10 else 'A' if b10 > b01 else 'tie',
    }


def run_mcnemar_tests(preds: Dict) -> Dict:
    """Run McNemar tests for key channel comparisons on each dataset."""
    results = {}
    comparisons = [('C', 'A'), ('BC', 'A'), ('C', 'AC'), ('BC', 'ABC')]

    for ds_name, channels in preds.items():
        if ds_name not in EVAL_DATASETS:
            continue
        for ch_a, ch_b in comparisons:
            if ch_a not in channels or ch_b not in channels:
                continue
            y_true = channels[ch_a]['y_true']
            y_pred_a = channels[ch_a]['y_pred']
            y_pred_b = channels[ch_b]['y_pred']

            key = f"{ds_name}__{ch_a}_vs_{ch_b}"
            results[key] = mcnemar_test(y_true, y_pred_a, y_pred_b)
            results[key]['dataset'] = ds_name
            results[key]['label_type'] = channels[ch_a].get(
                'label_type', get_dataset_info(ds_name).get('label_type', 'unknown'))

    return results


# =============================================================================
# SECTION 6: PERMUTATION TEST FOR REGIME DIVERGENCE  (INSTANCE-LEVEL)
# =============================================================================

def permutation_test_regime_divergence(preds: Dict, n_perm: int = 10000) -> Dict:
    """
    Two-level permutation test for regime divergence:

    LEVEL 1 (channel-level, N=7):
        Tests whether the CATEGORY of channels (structure vs lexical) predicts
        regime gap magnitude. This is inherently small-N because there are only
        7 channels — acceptable as a high-level structural test.

    LEVEL 2 (instance-level bootstrap, N=samples):
        For each bootstrap iteration, resample predictions within each dataset,
        recompute F1, then compare regime gaps of structure vs lexical channels.
        This gives proper statistical power from N=12,000+ samples.
    """
    # ── Compute channel-level regime gaps from predictions ──
    channel_gaps = {}
    channel_gap_bundles = {}  # for instance-level test

    for ch in ALL_CHANNELS:
        human_bundles, auto_bundles = [], []
        human_f1s, auto_f1s = [], []

        for ds in HUMAN_EVAL:
            if ds in preds and ch in preds[ds]:
                d = preds[ds][ch]
                human_f1s.append(f1_score(d['y_true'], d['y_pred'], average='macro', zero_division=0))
                human_bundles.append((ds, d['y_true'], d['y_pred']))
        for ds in AUTO_EVAL:
            if ds in preds and ch in preds[ds]:
                d = preds[ds][ch]
                auto_f1s.append(f1_score(d['y_true'], d['y_pred'], average='macro', zero_division=0))
                auto_bundles.append((ds, d['y_true'], d['y_pred']))

        if human_f1s and auto_f1s:
            channel_gaps[ch] = float(np.mean(human_f1s) - np.mean(auto_f1s))
            channel_gap_bundles[ch] = {'human': human_bundles, 'auto': auto_bundles}

    structure_gaps = [channel_gaps[ch] for ch in STRUCTURE_CHANNELS if ch in channel_gaps]
    lexical_gaps   = [channel_gaps[ch] for ch in LEXICAL_CHANNELS  if ch in channel_gaps]

    if not structure_gaps or not lexical_gaps:
        return {'status': 'insufficient_data'}

    obs_diff = float(np.mean(structure_gaps) - np.mean(lexical_gaps))

    # ── LEVEL 1: Channel-level permutation (N = n_channels) ──
    all_gaps = structure_gaps + lexical_gaps
    n_s = len(structure_gaps)

    rng = np.random.RandomState(RANDOM_STATE)
    perm_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(all_gaps)
        perm_diffs[i] = shuffled[:n_s].mean() - shuffled[n_s:].mean()

    p_channel = float(np.mean(perm_diffs >= obs_diff))

    # ── LEVEL 2: Instance-level bootstrap (N = total samples) ──
    rng2 = np.random.RandomState(RANDOM_STATE + 1)
    n_boot = min(n_perm, 5000)  # cap for speed since each iteration recomputes many F1s
    boot_diffs = np.zeros(n_boot)

    struct_chs = [ch for ch in STRUCTURE_CHANNELS if ch in channel_gap_bundles]
    lex_chs    = [ch for ch in LEXICAL_CHANNELS   if ch in channel_gap_bundles]

    for b in range(n_boot):
        struct_gap_boot, lex_gap_boot = [], []

        for ch_list, gap_list in [(struct_chs, struct_gap_boot), (lex_chs, lex_gap_boot)]:
            for ch in ch_list:
                bundles = channel_gap_bundles[ch]
                h_f1s = []
                for _, yt, yp in bundles['human']:
                    idx = rng2.randint(0, len(yt), size=len(yt))
                    h_f1s.append(f1_score(yt[idx], yp[idx], average='macro', zero_division=0))
                a_f1s = []
                for _, yt, yp in bundles['auto']:
                    idx = rng2.randint(0, len(yt), size=len(yt))
                    a_f1s.append(f1_score(yt[idx], yp[idx], average='macro', zero_division=0))
                gap_list.append(np.mean(h_f1s) - np.mean(a_f1s))

        boot_diffs[b] = np.mean(struct_gap_boot) - np.mean(lex_gap_boot)

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p_boot = float(np.mean(boot_diffs <= 0))

    total_n = sum(len(b[1]) for ch in channel_gap_bundles
                  for regime in channel_gap_bundles[ch].values()
                  for b in regime)

    # ── Combine results ──
    # Use instance-level p as primary (more power)
    p_primary = p_boot
    significant = ci_lo > 0 or ci_hi < 0

    return {
        'channel_gaps': channel_gaps,
        'structure_gaps': structure_gaps,
        'lexical_gaps': lexical_gaps,
        'observed_diff': obs_diff,
        # Level 1: channel-level permutation
        'p_permutation_channel_level': p_channel,
        'n_channels_shuffled': len(all_gaps),
        'note_channel_level': (
            f"N={len(all_gaps)} channels — low power by design; "
            "tests structural category assignment, not sample-level effect"
        ),
        # Level 2: instance-level bootstrap
        'p_bootstrap_instance_level': p_boot,
        'ci_95': [ci_lo, ci_hi],
        'n_boot': n_boot,
        'n_samples': total_n,
        # Primary verdict uses instance-level
        'p_permutation': p_primary,
        'significant': significant,
        'interpretation': (
            f"Structure channels show {obs_diff:+.4f} larger regime gap. "
            f"Instance-bootstrap (PRIMARY): p={p_boot:.4f}, "
            f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}], N={total_n:,}. "
            f"Channel-level (supplementary, N={len(all_gaps)}): p={p_channel:.4f}. "
            f"{'SIGNIFICANT' if significant else 'Not significant'} (based on instance-level CI)."
        ),
    }


# =============================================================================
# SECTION 7: EFFECT SIZES  (Cohen's d, instance-recomputed F1)
# =============================================================================

def compute_effect_sizes(preds: Dict) -> Dict:
    """Compute Cohen's d for key comparisons (from recomputed F1)."""
    scores = defaultdict(lambda: defaultdict(list))

    for ds, channels in preds.items():
        if ds not in EVAL_DATASETS:
            continue
        lt = get_dataset_info(ds).get('label_type', 'unknown')
        for ch, data in channels.items():
            mf1 = f1_score(data['y_true'], data['y_pred'], average='macro', zero_division=0)
            scores[ch][lt].append(float(mf1))
            scores[ch]['all'].append(float(mf1))

    es = {'channel_comparisons': {}}

    for c1, c2 in [('C', 'A'), ('BC', 'A'), ('C', 'AC'), ('BC', 'ABC')]:
        if c1 not in scores or c2 not in scores:
            continue
        for regime in ['human', 'auto', 'all']:
            g1, g2 = scores[c1].get(regime, []), scores[c2].get(regime, [])
            if g1 and g2:
                d = compute_cohens_d(g1, g2)
                sz = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
                es['channel_comparisons'][f'{c1}_vs_{c2}_{regime}'] = {
                    'd': d, 'size': sz,
                    'g1_mean': float(np.mean(g1)), 'g2_mean': float(np.mean(g2)),
                }

    human_all = [f1_score(preds[ds][ch]['y_true'], preds[ds][ch]['y_pred'],
                          average='macro', zero_division=0)
                 for ds in HUMAN_EVAL for ch in preds.get(ds, {})]
    auto_all  = [f1_score(preds[ds][ch]['y_true'], preds[ds][ch]['y_pred'],
                          average='macro', zero_division=0)
                 for ds in AUTO_EVAL for ch in preds.get(ds, {})]
    es['human_vs_auto'] = compute_cohens_d(human_all, auto_all) if human_all and auto_all else None

    return es


# =============================================================================
# SECTION 8: RELATIVE DOMAIN ROBUSTNESS (RDR)
# =============================================================================

def compute_relative_domain_robustness(score_matrix: pd.DataFrame) -> Dict:
    """RDR(ch) = 1 - (max_F1 - min_F1) / mean_F1."""
    rdr = {}
    for ch in score_matrix.columns:
        vals = score_matrix[ch].dropna().values
        if len(vals) >= 2 and vals.mean() > 0:
            rdr[ch] = float(1 - (vals.max() - vals.min()) / vals.mean())

    struct_rdr = [rdr[ch] for ch in STRUCTURE_CHANNELS if ch in rdr]
    lex_rdr    = [rdr[ch] for ch in LEXICAL_CHANNELS   if ch in rdr]

    return {
        'rdr_scores': rdr,
        'structure_mean_rdr': float(np.mean(struct_rdr)) if struct_rdr else None,
        'lexical_mean_rdr': float(np.mean(lex_rdr)) if lex_rdr else None,
        'structure_more_robust': (
            bool(np.mean(struct_rdr) > np.mean(lex_rdr))
            if struct_rdr and lex_rdr else None
        ),
        'interpretation': (
            "Structure channels show higher cross-domain robustness"
            if struct_rdr and lex_rdr and np.mean(struct_rdr) > np.mean(lex_rdr)
            else "No clear robustness advantage"
        ),
    }


# =============================================================================
# SECTION 9: CROSS-PLATFORM TRANSFER (aggregate — uses JSON data)
# =============================================================================

def analyze_transfer_significance(cross_platform_results: List[Dict]) -> Dict:
    """Analyze cross-platform transfer rates by channel type (aggregate data)."""
    if not cross_platform_results:
        return {'analysis': 'No cross-platform results available'}

    channel_transfers = defaultdict(list)
    for cp in cross_platform_results:
        channel_transfers[cp.get('channel', 'unknown')].append(
            cp.get('retention_rate_macro', cp.get('retention_rate', 0)))

    channel_stats = {}
    for ch, rates in channel_transfers.items():
        channel_stats[ch] = {
            'mean_retention': float(np.mean(rates)),
            'std_retention': float(np.std(rates)),
            'n_transfers': len(rates),
        }

    bc_rates = [r for ch in STRUCTURE_CHANNELS for r in channel_transfers.get(ch, [])]
    a_rates  = [r for ch in LEXICAL_CHANNELS  for r in channel_transfers.get(ch, [])]

    comparison = None
    if len(bc_rates) >= 2 and len(a_rates) >= 2:
        try:
            stat, p_val = mannwhitneyu(bc_rates, a_rates, alternative='greater')
            comparison = {
                'bc_mean': float(np.mean(bc_rates)), 'a_mean': float(np.mean(a_rates)),
                'mann_whitney_stat': float(stat), 'p_value': float(p_val),
                'significant': p_val < 0.05,
            }
        except ValueError:
            pass

    dti = {}
    for ch, s in channel_stats.items():
        if s['mean_retention'] > 0:
            cv = s['std_retention'] / s['mean_retention'] if s['mean_retention'] > 0 else 0
            dti[ch] = float(s['mean_retention'] / (1 + cv))

    return {
        'channel_stats': channel_stats,
        'bc_vs_a_comparison': comparison,
        'domain_transferability_index': dti,
        'interpretation': (
            f"B/C channels {'significantly' if comparison and comparison['significant'] else 'do not significantly'} "
            f"outperform A in cross-platform transfer"
        ) if comparison else "Insufficient data",
    }


# =============================================================================
# COMPREHENSIVE REPORT GENERATION
# =============================================================================

def generate_statistical_report(
    predictions_file: Optional[Path],
    results_file: Optional[Path],
    output_dir: Path,
    n_boot: int = 10000,
) -> Dict:
    """Generate comprehensive instance-level statistical report."""

    # -- LOAD DATA --
    preds = None
    eval_results_json = []
    cross_platform = []

    if predictions_file and predictions_file.exists():
        logger.info(f"Loading instance-level predictions: {predictions_file}")
        preds = load_predictions(predictions_file)
    else:
        logger.warning("No predictions file -- statistical power will be severely limited!")

    if results_file and results_file.exists():
        logger.info(f"Loading aggregate results: {results_file}")
        eval_results_json, cross_platform = load_aggregate_results(results_file)

    # Build score matrices
    if preds:
        score_matrix = build_score_matrix_from_predictions(preds)
        instance_level = True
        logger.info(f"   Score matrix: {score_matrix.shape[0]} datasets x {score_matrix.shape[1]} channels")
    elif eval_results_json:
        score_matrix = build_score_matrix_from_json(eval_results_json)
        instance_level = False
        logger.warning("=" * 72)
        logger.warning("⚠ RUNNING IN AGGREGATE-ONLY MODE (N=4 dataset means)")
        logger.warning("  Instance-level tests (DoD, McNemar, Interference) DISABLED.")
        logger.warning("  Results are NOT publication-ready.")
        logger.warning("  Re-run 03_masking_suite.py then retry with predictions.joblib.")
        logger.warning("=" * 72)
    else:
        logger.error("No data available!")
        sys.exit(1)

    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_version': '3.0',
        'instance_level': instance_level,
        'primary_metric': 'macro_f1',
        'n_boot': n_boot,
    }

    if instance_level:
        total_samples = sum(
            len(preds[ds][ch]['y_true'])
            for ds in preds for ch in preds[ds]
        )
        report['total_sample_predictions'] = total_samples
        logger.info(f"   Total sample predictions: {total_samples:,}")

    # === 1. CHANNEL ORTHOGONALITY ===
    logger.info("\n" + "=" * 80)
    logger.info("1. CHANNEL ORTHOGONALITY ANALYSIS (instance-level y_prob)")
    logger.info("=" * 80)

    if preds:
        ortho = analyze_channel_orthogonality(preds)
    else:
        ortho = {'status': 'no_predictions_file'}
    report['channel_orthogonality'] = ortho

    if isinstance(ortho.get('mean_abs_rho'), (int, float)):
        logger.info(f"\n   Mean |rho| = {ortho['mean_abs_rho']:.3f}")
    logger.info(f"   Orthogonal ratio = {ortho.get('ortho_ratio', 0):.1%}")

    if ortho.get('primary_pairs_ABC'):
        logger.info("\n   Primary pairs (A-B-C independence):")
        for pair, info in ortho['primary_pairs_ABC'].items():
            tag = "ORTHOGONAL" if info.get('orthogonal') else "correlated"
            logger.info(f"      {pair}: mean rho={info['mean_rho']:.3f} -> {tag}")

    logger.info(f"   -> {ortho.get('interpretation', 'N/A')}")

    # === 2. LABEL REGIME DIVERGENCE ===
    logger.info("\n" + "=" * 80)
    logger.info("2. LABEL REGIME DIVERGENCE ANALYSIS")
    logger.info("=" * 80)

    if preds:
        label_analysis = analyze_label_regime_divergence(preds, score_matrix)
    else:
        label_analysis = {'status': 'no_predictions', 'hypothesis_supported': False}
    report['label_regime_analysis'] = label_analysis

    if label_analysis.get('channel_analysis'):
        logger.info(f"\n   {'Ch':<5} {'Human':>8} {'Auto':>8} {'Gap':>8} {'Favors':>8}")
        logger.info("   " + "-" * 42)
        for ch, info in sorted(label_analysis['channel_analysis'].items()):
            fav = "Human" if info['human_advantage'] else "Auto"
            logger.info(f"   {ch:<5} {info['human_mean']:>8.3f} {info['auto_mean']:>8.3f} "
                        f"{info['regime_gap']:>+8.3f} {fav:>8}")

    logger.info(f"\n   -> {label_analysis.get('interpretation', 'N/A')}")

    # === 3. DEGREE OF DIVERGENCE (DoD) ===
    logger.info("\n" + "=" * 80)
    logger.info("3. DEGREE OF DIVERGENCE (DoD) -- Instance-Level Bootstrap")
    logger.info("=" * 80)

    if preds:
        dod_results = run_all_dod_comparisons(preds, n_boot=n_boot)
    else:
        dod_results = {'status': 'no_predictions'}
    report['dod_analysis'] = dod_results

    for name, res in dod_results.items():
        if name == 'fdr_correction':
            continue
        if isinstance(res, dict) and 'dod' in res:
            sig = "SIG *" if res.get('significant') else "ns"
            logger.info(
                f"\n   {name}: DoD={res['dod']:+.4f} "
                f"[{res['ci_95'][0]:+.4f},{res['ci_95'][1]:+.4f}] "
                f"p={res.get('p_value', 1):.4f} d={res.get('cohens_d', 0):.2f} "
                f"N={res.get('n_samples', '?'):,} {sig}"
            )

    if 'fdr_correction' in dod_results:
        fdr = dod_results['fdr_correction']
        logger.info(f"\n   FDR: {fdr['n_rejected']}/{fdr['n_tests']} significant after BH")

    # === 4. LEXICAL INTERFERENCE ===
    logger.info("\n" + "=" * 80)
    logger.info("4. LEXICAL INTERFERENCE -- Instance-Level Bootstrap")
    logger.info("=" * 80)

    if preds:
        interference = test_lexical_interference(preds, n_boot=n_boot)
    else:
        interference = {'summary': {'interpretation': 'No predictions'}}
    report['lexical_interference'] = interference

    for key in ['c_to_ac_human', 'c_to_ac_auto', 'bc_to_abc_human', 'bc_to_abc_auto']:
        if key in interference:
            info = interference[key]
            tag = "DROP" if info.get('interference_present') else "GAIN"
            ci = info.get('ci_95', [0, 0])
            logger.info(
                f"   {key}: delta={info['mean_drop']:+.4f} "
                f"[{ci[0]:+.4f},{ci[1]:+.4f}] "
                f"p={info.get('p_value', 1):.4f} N={info.get('n_samples', '?'):,} -> {tag}"
            )

    logger.info(f"\n   -> {interference.get('summary', {}).get('interpretation', 'N/A')}")

    # === 5. McNEMAR'S TESTS ===
    logger.info("\n" + "=" * 80)
    logger.info("5. McNEMAR'S TEST -- Paired Classifier Comparison")
    logger.info("=" * 80)

    if preds:
        mcnemar_results = run_mcnemar_tests(preds)
    else:
        mcnemar_results = {}
    report['mcnemar_tests'] = mcnemar_results

    for key, res in sorted(mcnemar_results.items()):
        sig = "SIG" if res.get('p_value', 1) < 0.05 else "ns"
        lt = res.get('label_type', '?')[0].upper()
        logger.info(
            f"   [{lt}] {key}: chi2={res.get('statistic', 0):.1f} "
            f"p={res.get('p_value', 1):.4f} "
            f"(b01={res.get('b01', 0)}, b10={res.get('b10', 0)}) "
            f"N={res.get('n_samples', '?'):,} {sig}"
        )

    # === 6. PERMUTATION REGIME DIVERGENCE ===
    logger.info("\n" + "=" * 80)
    logger.info("6. PERMUTATION TEST: Structure vs Lexical Regime Divergence")
    logger.info("=" * 80)

    if preds:
        perm_regime = permutation_test_regime_divergence(preds)
    else:
        perm_regime = {'status': 'no_predictions'}
    report['permutation_regime_divergence'] = perm_regime

    if 'observed_diff' in perm_regime:
        logger.info(f"   Gap diff (structure - lexical): {perm_regime['observed_diff']:+.4f}")
        logger.info(f"   ┌─ Level 1 (SUPPLEMENTARY — low-power, N={perm_regime.get('n_channels_shuffled', '?')} channels):")
        logger.info(f"   │  Channel-level permutation p={perm_regime.get('p_permutation_channel_level', '?'):.4f}")
        logger.info(f"   │  Note: N<10, expect non-significance by design. Do NOT cite this alone.")
        logger.info(f"   └─ Level 2 (PRIMARY — instance-level, N={perm_regime.get('n_samples', '?'):,} samples):")
        logger.info(f"      Bootstrap p={perm_regime.get('p_bootstrap_instance_level', '?'):.4f} "
                    f"CI=[{perm_regime.get('ci_95', [0,0])[0]:+.4f}, {perm_regime.get('ci_95', [0,0])[1]:+.4f}]")
        logger.info(f"      ★ VERDICT uses Level 2 only.")
        logger.info(f"   -> {perm_regime.get('interpretation', 'N/A')}")

    # === 7. EFFECT SIZES ===
    logger.info("\n" + "=" * 80)
    logger.info("7. EFFECT SIZE ANALYSIS (Cohen's d)")
    logger.info("=" * 80)

    if preds:
        effect_sizes = compute_effect_sizes(preds)
    else:
        effect_sizes = {}
    report['effect_sizes'] = effect_sizes

    if effect_sizes.get('human_vs_auto') is not None:
        logger.info(f"   Human vs Auto (overall): d={effect_sizes['human_vs_auto']:.3f}")
    for comp, info in sorted(effect_sizes.get('channel_comparisons', {}).items(),
                             key=lambda x: abs(x[1].get('d', 0)), reverse=True):
        logger.info(f"   {comp}: d={info['d']:+.3f} ({info['size']})")

    # === 8. RDR ===
    logger.info("\n" + "=" * 80)
    logger.info("8. RELATIVE DOMAIN ROBUSTNESS (RDR)")
    logger.info("=" * 80)

    rdr = compute_relative_domain_robustness(score_matrix)
    report['domain_robustness'] = rdr

    if rdr.get('rdr_scores'):
        for ch, sc in sorted(rdr['rdr_scores'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {ch}: RDR={sc:.3f}")
        logger.info(f"   Structure mean: {rdr.get('structure_mean_rdr', 'N/A')}")
        logger.info(f"   Lexical mean:   {rdr.get('lexical_mean_rdr', 'N/A')}")
        logger.info(f"   -> {rdr.get('interpretation', 'N/A')}")

    # === 9. GLOBAL FDR ===
    logger.info("\n" + "=" * 80)
    logger.info("9. GLOBAL FDR CORRECTION (Benjamini-Hochberg)")
    logger.info("=" * 80)

    all_pv = {}
    for name, res in dod_results.items():
        if isinstance(res, dict) and 'p_value' in res:
            all_pv[f'dod_{name}'] = res['p_value']
    for name, res in interference.items():
        if isinstance(res, dict) and 'p_value' in res:
            all_pv[f'interf_{name}'] = res['p_value']
    for name, res in mcnemar_results.items():
        if isinstance(res, dict) and res.get('p_value') is not None:
            all_pv[f'mcnemar_{name}'] = res['p_value']
    if perm_regime.get('p_bootstrap_instance_level') is not None:
        all_pv['perm_regime_divergence'] = perm_regime['p_bootstrap_instance_level']
    elif perm_regime.get('p_permutation') is not None:
        all_pv['perm_regime_divergence'] = perm_regime['p_permutation']

    if all_pv:
        fdr_global = apply_fdr_correction(all_pv)
        report['global_fdr_correction'] = fdr_global
        logger.info(f"   {fdr_global['n_tests']} tests, "
                    f"{fdr_global['n_rejected']} significant after FDR")
        for name, res in sorted(fdr_global['corrected'].items()):
            tag = 'SIG' if res['reject_h0'] else 'ns'
            logger.info(f"   {tag:>3} {name}: p={res['p_original']:.4f} -> "
                        f"p_adj={res['p_adjusted']:.4f}")
    else:
        report['global_fdr_correction'] = {'note': 'No p-values collected'}

    # === 10. CROSS-PLATFORM TRANSFER (aggregate) ===
    logger.info("\n" + "=" * 80)
    logger.info("10. CROSS-PLATFORM TRANSFER (aggregate)")
    logger.info("=" * 80)

    transfer = analyze_transfer_significance(cross_platform)
    report['transfer_analysis'] = transfer

    if transfer.get('channel_stats'):
        for ch, s in sorted(transfer['channel_stats'].items()):
            logger.info(f"   {ch}: {s['mean_retention']:.1f}% +/- {s['std_retention']:.1f}%")
    if transfer.get('domain_transferability_index'):
        logger.info("\n   Domain Transferability Index (DTI):")
        for ch, dti_v in sorted(transfer['domain_transferability_index'].items(),
                                key=lambda x: x[1], reverse=True):
            logger.info(f"      {ch}: DTI={dti_v:.2f}")

    # === HYPOTHESIS VERDICT ===
    logger.info("\n" + "=" * 80)
    logger.info("HYPOTHESIS VERDICT: Labeling-Regime Divergence")
    logger.info("=" * 80)

    verdicts = []

    # 1. Label regime gap
    if label_analysis.get('hypothesis_supported'):
        verdicts.append(("Label regime gap", "SUPPORTED", "B/C/BC gap > A gap"))
    else:
        verdicts.append(("Label regime gap", "NOT SUPPORTED", ""))

    # 2. DoD BC-A
    bc_dod = dod_results.get('BC_vs_A', {})
    if bc_dod.get('significant') and bc_dod.get('dod', 0) > 0:
        verdicts.append(("DoD BC-A", "SUPPORTED",
                         f"DoD={bc_dod['dod']:+.4f}, d={bc_dod.get('cohens_d', 0):.2f}"))
    elif bc_dod.get('dod', 0) > 0:
        verdicts.append(("DoD BC-A", "PARTIAL", f"DoD={bc_dod.get('dod', 0):+.4f} (CI includes 0)"))
    else:
        verdicts.append(("DoD BC-A", "NOT SUPPORTED", ""))

    # 3. Lexical interference
    if interference.get('summary', {}).get('asymmetric'):
        verdicts.append(("Lexical interference", "SUPPORTED", "C>AC on human only"))
    else:
        verdicts.append(("Lexical interference", "PARTIAL", ""))

    # 4. Channel orthogonality
    if isinstance(ortho.get('mean_abs_rho'), (int, float)) and ortho['mean_abs_rho'] < 0.5:
        verdicts.append(("Channel orthogonality", "SUPPORTED",
                         f"mean|rho|={ortho['mean_abs_rho']:.3f}"))
    else:
        verdicts.append(("Channel orthogonality", "WEAK", ""))

    # 5. Regime permutation (uses instance-level bootstrap as primary)
    if perm_regime.get('significant'):
        p_inst = perm_regime.get('p_bootstrap_instance_level', perm_regime.get('p_permutation', 0))
        p_ch = perm_regime.get('p_permutation_channel_level', 'N/A')
        verdicts.append(("Regime permutation", "SUPPORTED",
                         f"instance p={p_inst:.4f}, N={perm_regime.get('n_samples', '?'):,} "
                         f"(channel-level p={p_ch} — supplementary)"))
    else:
        verdicts.append(("Regime permutation", "NOT SIGNIFICANT", ""))

    # 6. McNemar (bonus: C vs A on human data — exact match only)
    mcnemar_ca_human = [v for k, v in mcnemar_results.items()
                        if k.endswith('__C_vs_A') and v.get('label_type') == 'human'
                        and v.get('p_value', 1) < 0.05]
    if mcnemar_ca_human:
        verdicts.append(("McNemar C vs A (human)", "SUPPORTED",
                         f"{len(mcnemar_ca_human)} sig. datasets"))
    else:
        verdicts.append(("McNemar C vs A (human)", "NOT SIGNIFICANT", ""))

    n_sup = sum(1 for _, v, _ in verdicts if v == "SUPPORTED")
    n_total = len(verdicts)
    overall = ("STRONGLY SUPPORTED" if n_sup >= 5 else
               "SUPPORTED" if n_sup >= 4 else
               "PARTIALLY SUPPORTED" if n_sup >= 2 else "WEAK SUPPORT")

    for test, verdict, detail in verdicts:
        mk = "+" if verdict == "SUPPORTED" else "~" if "PARTIAL" in verdict else "-"
        logger.info(f"   [{mk}] {test}: {verdict} "
                    f"{('(' + detail + ')') if detail else ''}")

    logger.info(f"\n   == OVERALL: {overall} ({n_sup}/{n_total} tests) ==")

    report['hypothesis_verdict'] = {
        'individual_tests': [{'test': t, 'verdict': v, 'detail': d} for t, v, d in verdicts],
        'n_supported': n_sup, 'n_total': n_total, 'overall': overall,
    }

    # === SAVE ===
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / f"statistical_report_{ts}.json"

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=json_default)
    logger.info(f"\n   Report saved to: {out_path}")

    return report


# =============================================================================
# LATEX EXPORT
# =============================================================================

def export_latex_tables(report: Dict, output_dir: Path):
    """Export key tables as LaTeX for the paper."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tex_path = output_dir / f"statistical_tables_{ts}.tex"
    lines = [
        "% Auto-generated by 04_statistical_analysis.py v3.0",
        "% Instance-level statistical analysis",
        "",
    ]

    # Table 1: DoD comparisons
    dod = report.get('dod_analysis', {})
    lines.append("% --- Table: Degree of Divergence (Instance-Level Bootstrap) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Degree of Divergence (DoD) --- Instance-Level Bootstrap}")
    lines.append(r"\begin{tabular}{lrrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & $\Delta_H$ & $\Delta_A$ & DoD & 95\% CI & $p$ & Sig. \\")
    lines.append(r"\midrule")
    for name, res in sorted(dod.items()):
        if not isinstance(res, dict) or 'dod' not in res:
            continue
        sig = r"\checkmark" if res.get('significant') else "--"
        ci = res.get('ci_95', [0, 0])
        lines.append(
            f"{name} & {res['delta_human']:+.3f} & {res['delta_auto']:+.3f} & "
            f"{res['dod']:+.3f} & [{ci[0]:+.3f}, {ci[1]:+.3f}] & "
            f"{res.get('p_value', 1):.3f} & {sig} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 2: Orthogonality
    ortho = report.get('channel_orthogonality', {})
    if ortho.get('primary_pairs_ABC'):
        lines.append("% --- Table: Channel Orthogonality (Instance-Level Spearman) ---")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Channel Orthogonality --- Instance-Level Spearman $\rho$}")
        lines.append(r"\begin{tabular}{lrl}")
        lines.append(r"\toprule")
        lines.append(r"Pair & Mean $\rho$ & Status \\")
        lines.append(r"\midrule")
        for pair, info in ortho['primary_pairs_ABC'].items():
            status = "Orthogonal" if info.get('orthogonal') else "Correlated"
            lines.append(f"{pair} & {info['mean_rho']:.3f} & {status} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # Table 3: Hypothesis verdict
    verdict = report.get('hypothesis_verdict', {})
    if verdict.get('individual_tests'):
        lines.append("% --- Table: Hypothesis Verdict ---")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Hypothesis Verdict: Labeling-Regime Divergence}")
        lines.append(r"\begin{tabular}{lll}")
        lines.append(r"\toprule")
        lines.append(r"Test & Verdict & Detail \\")
        lines.append(r"\midrule")
        for t in verdict['individual_tests']:
            lines.append(f"{t['test']} & {t['verdict']} & {t['detail']} \\\\")
        lines.append(r"\midrule")
        lines.append(f"\\textbf{{Overall}} & \\textbf{{{verdict['overall']}}} & "
                     f"{verdict['n_supported']}/{verdict['n_total']} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"   LaTeX tables saved to: {tex_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TSS Statistical Analysis v3.0 (Instance-Level)")
    parser.add_argument('--predictions', type=str, default=None,
                        help='Predictions joblib file (glob OK)')
    parser.add_argument('--results', type=str, default=None,
                        help='Aggregate JSON results (glob OK)')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--export-latex', action='store_true',
                        help='Export LaTeX tables')
    parser.add_argument('--allow-aggregate', action='store_true',
                        help='Allow running with aggregate JSON only (no joblib). '
                             'WARNING: This gives N=4 statistical power — not publication-ready.')
    parser.add_argument('--n-boot', type=int, default=10000,
                        help='Bootstrap iterations')
    args = parser.parse_args()

    # Resolve file paths (support glob patterns)
    predictions_file = None
    results_file = None

    if args.predictions:
        p = Path(args.predictions)
        if p.exists():
            predictions_file = p
        else:
            matches = sorted(glob.glob(str(p)))
            if matches:
                predictions_file = Path(matches[-1])

    if args.results:
        p = Path(args.results)
        if p.exists():
            results_file = p
        else:
            matches = sorted(glob.glob(str(p)))
            if matches:
                results_file = Path(matches[-1])

    # Auto-discover files if not specified
    if predictions_file is None:
        candidates = sorted(glob.glob('outputs/predictions_*.joblib'))
        if candidates:
            predictions_file = Path(candidates[-1])
            logger.info(f"Auto-discovered predictions: {predictions_file}")

    if results_file is None:
        candidates = sorted(glob.glob('outputs/masking_comprehensive_*.json') +
                           glob.glob('outputs/tss_results_*.json'))
        if candidates:
            results_file = Path(candidates[-1])
            logger.info(f"Auto-discovered results: {results_file}")

    if predictions_file is None and results_file is None:
        logger.error("No data files found. Provide --predictions and/or --results")
        sys.exit(1)

    # ── Guard: refuse aggregate-only mode unless explicitly allowed ──
    if predictions_file is None and not args.allow_aggregate:
        logger.error("=" * 72)
        logger.error("FATAL: No predictions file (*.joblib) found!")
        logger.error("")
        logger.error("  Without instance-level predictions, statistical analysis has only")
        logger.error("  N=4 dataset means — insufficient for publication-quality results.")
        logger.error("")
        logger.error("  SOLUTIONS (in order of preference):")
        logger.error("  1. Re-run 03_masking_suite.py to regenerate predictions:")
        logger.error("     python scripts/03_masking_suite.py --channels A,B,C,AB,AC,BC,ABC")
        logger.error("     Then check that outputs/predictions_*.joblib exists and is >1MB.")
        logger.error("")
        logger.error("  2. Point to an existing joblib file:")
        logger.error("     python scripts/04_statistical_analysis.py --predictions path/to/predictions.joblib")
        logger.error("")
        logger.error("  3. Force aggregate-only mode (NOT recommended for papers):")
        logger.error("     python scripts/04_statistical_analysis.py --allow-aggregate")
        logger.error("=" * 72)
        sys.exit(1)

    # ── Pre-flight: validate joblib file size ──
    if predictions_file is not None:
        file_size_mb = predictions_file.stat().st_size / (1024 * 1024)
        if file_size_mb < 0.01:
            logger.error(f"Predictions file {predictions_file} is suspiciously small ({file_size_mb:.4f} MB).")
            logger.error("Re-run 03_masking_suite.py to regenerate.")
            sys.exit(1)
        logger.info(f"Predictions file: {predictions_file} ({file_size_mb:.1f} MB)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "INSTANCE-LEVEL" if predictions_file else "⚠ AGGREGATE-ONLY (--allow-aggregate)"

    print(f"""
+============================================================================+
|           TSS STATISTICAL ANALYSIS v3.0  ({mode})            |
|                                                                            |
|  PRIMARY METRIC: Macro F1                                                  |
|                                                                            |
|  KEY ANALYSES (instance-level where applicable):                           |
|  +-- Channel Orthogonality   (Spearman rho on y_prob, N=samples)          |
|  +-- Label Regime Divergence (recomputed from predictions)                 |
|  +-- DoD Bootstrap           (resample predictions, N=samples)             |
|  +-- Lexical Interference    (paired bootstrap, N=samples)                 |
|  +-- McNemar's Test          (paired classifier, N=samples)                |
|  +-- Regime Permutation      (shuffle predictions)                         |
|  +-- Effect Sizes            (Cohen's d per regime)                        |
|  +-- RDR + Transfer          (aggregate)                                   |
|  +-- Global FDR              (Benjamini-Hochberg)                          |
+============================================================================+
    """)

    report = generate_statistical_report(
        predictions_file, results_file, output_dir, n_boot=args.n_boot)

    if args.export_latex:
        export_latex_tables(report, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL ANALYSIS v3.0 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
