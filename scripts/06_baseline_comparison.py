#!/usr/bin/env python3
"""
================================================================================
TSS vs Baseline Statistical Comparison (v3.0)
================================================================================

Compares TSS channels against baseline models (MentalBERT, LLaMA) using
identical test data and unified metric computation.

USAGE:
    python scripts/06_baseline_comparison.py \
        --tss-predictions outputs/predictions_*.joblib \
        --baselines outputs/mentalbert_predictions_*.joblib \
                    outputs/llama_predictions_*.joblib \
        --output outputs

STATISTICAL TESTS:
    1. McNemar's test (per dataset, paired classifier comparison)
    2. Paired bootstrap CI for Macro-F1 difference
    3. Cohen's d effect sizes
    4. Global summary table (LaTeX)

INPUT FORMAT:
    Baseline joblib files must contain:
    {
        '_meta': {'model': str, ...},
        'predictions': [
            {'model': str, 'dataset': str, 'label_type': str,
             'y_true': array, 'y_pred': array, 'y_prob': array|None,
             'metrics': dict}
        ]
    }
"""

import sys, os, json, glob, logging, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import joblib

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_tss_predictions(predictions_file: Path) -> Dict:
    """Load TSS predictions (envelope or legacy format)."""
    loaded = joblib.load(predictions_file)

    if isinstance(loaded, dict) and '_meta' in loaded:
        raw = loaded['predictions']
    elif isinstance(loaded, list):
        raw = loaded
    else:
        raise ValueError(f"Unknown format: {type(loaded)}")

    # Structure: {dataset: {channel: {y_true, y_pred, y_prob}}}
    tss = defaultdict(dict)
    for item in raw:
        mm = item.get('mask_mode', 'none')
        if mm != 'none':
            continue
        ds = item['dataset']
        ch = item['channel']
        tss[ds][ch] = {
            'y_true': np.asarray(item['y_true']),
            'y_pred': np.asarray(item['y_pred']),
            'y_prob': np.asarray(item['y_prob']) if item.get('y_prob') is not None else None,
            'label_type': item.get('label_type', 'unknown'),
        }

    logger.info(f"TSS: {sum(len(v) for v in tss.values())} (ds, ch) pairs | "
                f"Datasets: {sorted(tss.keys())}")
    return dict(tss)


def load_baseline_predictions(baseline_file: Path) -> Dict:
    """Load baseline predictions (MentalBERT or LLaMA)."""
    loaded = joblib.load(baseline_file)

    if isinstance(loaded, dict) and '_meta' in loaded:
        meta = loaded['_meta']
        preds = loaded['predictions']
    else:
        raise ValueError(f"Unknown format for {baseline_file}")

    model_name = meta.get('model', 'Unknown')

    # Structure: {dataset: {y_true, y_pred, y_prob, metrics}}
    result = {}
    for item in preds:
        ds = item['dataset']
        result[ds] = {
            'y_true': np.asarray(item['y_true']),
            'y_pred': np.asarray(item['y_pred']),
            'y_prob': np.asarray(item['y_prob']) if item.get('y_prob') is not None else None,
            'label_type': item.get('label_type', 'unknown'),
            'metrics': item.get('metrics', {}),
        }

    logger.info(f"{model_name}: {len(result)} datasets | "
                f"Datasets: {sorted(result.keys())}")
    return model_name, result


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> Dict:
    """McNemar's test for paired classifiers."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency: b_correct_a_wrong vs a_correct_b_wrong
    b01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    b10 = int(np.sum(correct_a & ~correct_b))   # A right, B wrong

    n = b01 + b10
    if n == 0:
        return {'chi2': 0, 'p_value': 1.0, 'n_discordant': 0, 'significant': False}

    # McNemar with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0
    from scipy.stats import chi2 as chi2_dist
    p_value = float(1 - chi2_dist.cdf(chi2, df=1))

    return {
        'chi2': float(chi2),
        'p_value': p_value,
        'a_right_b_wrong': b10,
        'b_right_a_wrong': b01,
        'n_discordant': n,
        'significant': p_value < 0.05,
        'winner': 'A' if b10 > b01 else 'B' if b01 > b10 else 'tie',
    }


def paired_bootstrap_f1_diff(y_true, y_pred_a, y_pred_b,
                              n_boot=10000, seed=42) -> Dict:
    """Bootstrap CI for Macro-F1 difference (A - B)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1_a = f1_score(yt, y_pred_a[idx], average='macro', zero_division=0)
        f1_b = f1_score(yt, y_pred_b[idx], average='macro', zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    p_value = float(np.mean(diffs <= 0))  # P(A <= B)

    return {
        'mean_diff': float(np.mean(diffs)),
        'ci_95': [ci_lo, ci_hi],
        'p_value': p_value,
        'significant': ci_lo > 0 or ci_hi < 0,
        'n_boot': len(diffs),
    }


def cohens_d(y_true, y_pred_a, y_pred_b, n_boot=2000, seed=42) -> float:
    """Cohen's d for Macro-F1 difference via bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1s_a, f1s_b = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1s_a.append(f1_score(yt, y_pred_a[idx], average='macro', zero_division=0))
        f1s_b.append(f1_score(yt, y_pred_b[idx], average='macro', zero_division=0))

    f1s_a = np.array(f1s_a)
    f1s_b = np.array(f1s_b)
    pooled_std = np.sqrt((f1s_a.var() + f1s_b.var()) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((f1s_a.mean() - f1s_b.mean()) / pooled_std)


# =============================================================================
# RECOMPUTE METRICS (unified computation for all models)
# =============================================================================

def compute_metrics_unified(y_true, y_pred, y_prob=None) -> Dict:
    """Compute metrics with IDENTICAL settings across all models."""
    out = {}
    out['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    out['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    out['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    out['n'] = len(y_true)
    return out


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(tss_preds: Dict, baselines: Dict[str, Dict],
                   tss_channel: str = 'ABC', n_boot: int = 10000) -> Dict:
    """
    Run full statistical comparison between TSS and baselines.

    Args:
        tss_preds: {dataset: {channel: {y_true, y_pred, ...}}}
        baselines: {model_name: {dataset: {y_true, y_pred, ...}}}
        tss_channel: Which TSS channel to use as primary (default: ABC)
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'tss_channel': tss_channel,
        'n_boot': n_boot,
        'comparisons': {},
    }

    all_datasets = set()
    for ds in tss_preds:
        all_datasets.add(ds)
    for model_preds in baselines.values():
        for ds in model_preds:
            all_datasets.add(ds)

    # Remove training sets
    eval_datasets = sorted(ds for ds in all_datasets if not ds.endswith('_train'))

    logger.info(f"\nEval datasets: {eval_datasets}")
    logger.info(f"TSS channel: {tss_channel}")
    logger.info(f"Baselines: {list(baselines.keys())}")

    print(f"\n{'='*90}")
    print(f"{'Dataset':<18} {'Model':<20} {'Macro-F1':>10} {'MCC':>10} {'McNemar p':>12} {'ΔF1 CI':>22}")
    print(f"{'='*90}")

    for ds in eval_datasets:
        # TSS predictions for this dataset
        tss_ds = tss_preds.get(ds, {}).get(tss_channel, None)
        if tss_ds is None:
            logger.warning(f"   No TSS {tss_channel} for {ds}")
            continue

        y_true_tss = tss_ds['y_true']
        y_pred_tss = tss_ds['y_pred']
        tss_metrics = compute_metrics_unified(y_true_tss, y_pred_tss, tss_ds.get('y_prob'))

        print(f"\n{ds:<18} {'TSS-'+tss_channel:<20} {tss_metrics['macro_f1']:>10.4f} "
              f"{tss_metrics['mcc']:>10.4f} {'(reference)':>12}")

        report['comparisons'][ds] = {
            'tss': {
                'channel': tss_channel,
                'metrics': tss_metrics,
                'n': len(y_true_tss),
            },
            'baselines': {},
        }

        for model_name, model_preds in baselines.items():
            if ds not in model_preds:
                continue

            bl = model_preds[ds]
            y_true_bl = bl['y_true']
            y_pred_bl = bl['y_pred']

            # Verify same test data
            if len(y_true_bl) != len(y_true_tss):
                logger.warning(f"   ⚠️ {model_name} {ds}: n={len(y_true_bl)} vs TSS n={len(y_true_tss)}")
                logger.warning(f"      CANNOT compare — different test sets!")
                report['comparisons'][ds]['baselines'][model_name] = {
                    'error': f'Size mismatch: {len(y_true_bl)} vs {len(y_true_tss)}',
                }
                continue

            if not np.array_equal(y_true_bl, y_true_tss):
                logger.warning(f"   ⚠️ {model_name} {ds}: y_true differs! Labels may be shuffled.")

            bl_metrics = compute_metrics_unified(y_true_bl, y_pred_bl, bl.get('y_prob'))

            # McNemar's test
            mcnemar = mcnemar_test(y_true_tss, y_pred_tss, y_pred_bl)

            # Paired bootstrap
            boot = paired_bootstrap_f1_diff(y_true_tss, y_pred_tss, y_pred_bl,
                                            n_boot=n_boot)

            # Effect size
            d = cohens_d(y_true_tss, y_pred_tss, y_pred_bl)

            ci_str = f"[{boot['ci_95'][0]:+.3f}, {boot['ci_95'][1]:+.3f}]"
            sig = "***" if mcnemar['p_value'] < 0.001 else "**" if mcnemar['p_value'] < 0.01 else "*" if mcnemar['p_value'] < 0.05 else ""

            print(f"{'':18} {model_name:<20} {bl_metrics['macro_f1']:>10.4f} "
                  f"{bl_metrics['mcc']:>10.4f} {mcnemar['p_value']:>10.4f}{sig:<2} {ci_str:>22}")

            report['comparisons'][ds]['baselines'][model_name] = {
                'metrics': bl_metrics,
                'mcnemar': mcnemar,
                'bootstrap_diff': boot,
                'cohens_d': d,
                'n': len(y_true_bl),
            }

    print(f"\n{'='*90}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY: TSS vs Baselines (Macro-F1)")
    print(f"{'='*70}")

    for ds, comp in report['comparisons'].items():
        tss_f1 = comp['tss']['metrics']['macro_f1']
        label_type = tss_preds.get(ds, {}).get(tss_channel, {}).get('label_type', '?')
        print(f"\n  {ds} [{label_type}] (n={comp['tss']['n']:,}):")
        print(f"    TSS-{tss_channel}: {tss_f1:.4f}")
        for model_name, bl in comp.get('baselines', {}).items():
            if 'error' in bl:
                print(f"    {model_name}: {bl['error']}")
            else:
                bl_f1 = bl['metrics']['macro_f1']
                diff = tss_f1 - bl_f1
                sig = "✓" if bl['bootstrap_diff']['significant'] else "✗"
                d = bl['cohens_d']
                print(f"    {model_name}: {bl_f1:.4f} (Δ={diff:+.4f}, d={d:+.2f}, sig={sig})")

    return report


def export_latex(report: Dict, output_dir: Path):
    """Export comparison as LaTeX table."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tex_path = output_dir / f"baseline_comparison_{ts}.tex"

    lines = [
        "% Auto-generated by 06_baseline_comparison.py",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{TSS vs Baseline Models (Macro-F1)}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Dataset & Model & Macro-F1 & MCC & McNemar $p$ & $\Delta$F1 95\% CI & Sig. \\",
        r"\midrule",
    ]

    tss_ch = report.get('tss_channel', 'ABC')

    for ds, comp in report['comparisons'].items():
        tss_f1 = comp['tss']['metrics']['macro_f1']
        tss_mcc = comp['tss']['metrics']['mcc']
        lines.append(f"{ds} & TSS-{tss_ch} & {tss_f1:.3f} & {tss_mcc:.3f} & -- & -- & -- \\\\")

        for mn, bl in comp.get('baselines', {}).items():
            if 'error' in bl:
                continue
            bl_f1 = bl['metrics']['macro_f1']
            bl_mcc = bl['metrics']['mcc']
            mc_p = bl['mcnemar']['p_value']
            ci = bl['bootstrap_diff']['ci_95']
            sig = r"\checkmark" if bl['bootstrap_diff']['significant'] else "--"
            lines.append(
                f" & {mn} & {bl_f1:.3f} & {bl_mcc:.3f} & "
                f"{mc_p:.3f} & [{ci[0]:+.3f}, {ci[1]:+.3f}] & {sig} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"   LaTeX: {tex_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TSS vs Baseline Statistical Comparison")
    parser.add_argument('--tss-predictions', type=str, required=True,
                        help='TSS predictions joblib (glob OK)')
    parser.add_argument('--baselines', type=str, nargs='+', required=True,
                        help='Baseline prediction joblib files')
    parser.add_argument('--tss-channel', type=str, default='ABC',
                        help='TSS channel to compare (default: ABC)')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--export-latex', action='store_true')
    args = parser.parse_args()

    # Resolve TSS file
    tss_file = None
    p = Path(args.tss_predictions)
    if p.exists():
        tss_file = p
    else:
        matches = sorted(glob.glob(str(p)))
        if matches:
            tss_file = Path(matches[-1])
    if tss_file is None:
        logger.error(f"TSS predictions not found: {args.tss_predictions}")
        sys.exit(1)

    # Resolve baseline files
    baseline_files = []
    for pattern in args.baselines:
        p = Path(pattern)
        if p.exists():
            baseline_files.append(p)
        else:
            matches = sorted(glob.glob(str(p)))
            baseline_files.extend(Path(m) for m in matches)

    if not baseline_files:
        logger.error("No baseline files found!")
        sys.exit(1)

    # Load
    logger.info(f"TSS predictions: {tss_file}")
    tss_preds = load_tss_predictions(tss_file)

    baselines = {}
    for bf in baseline_files:
        logger.info(f"Baseline: {bf}")
        model_name, preds = load_baseline_predictions(bf)
        baselines[model_name] = preds

    # Run comparison
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_comparison(tss_preds, baselines,
                            tss_channel=args.tss_channel,
                            n_boot=args.n_boot)

    # Save report
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"baseline_comparison_{ts}.json"

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=json_default)
    logger.info(f"\n   Report: {report_path}")

    if args.export_latex:
        export_latex(report, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
