"""
Shared Dataset Registry for TSS Pipeline
=========================================
Single source of truth for dataset metadata AND file mapping.
Used by both 03_masking_suite.py and 04_statistical_analysis.py.

To add a new dataset:
  1. Add a CSV to data/ with a descriptive name (e.g., facebook_gold.csv)
  2. The system will auto-discover it and infer label_type from the name.
  3. OR add it explicitly to DATASET_INFO below for full control.

Auto-discovery rules (applied when dataset name is not found here):
  - Name contains 'gold/human/expert/dreaddit' → 'human'
  - Name contains 'combi/auto/distant/proxy'   → 'auto'
  - Otherwise → 'unknown' (logged as warning)
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL DATASET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Known datasets. New CSVs in data/ are auto-discovered at runtime.

DATASET_INFO: Dict[str, Dict[str, str]] = {
    'dreaddit_train': {'label_type': 'human', 'platform': 'reddit'},
    'dreaddit_test':  {'label_type': 'human', 'platform': 'reddit'},
    'twitter':        {'label_type': 'auto',  'platform': 'twitter'},
    'twitter_gold':   {'label_type': 'human', 'platform': 'twitter'},
    'reddit_combi':   {'label_type': 'auto',  'platform': 'reddit'},
}

# Maps dataset_name → CSV filename.
# Pre-populated for known datasets; extended by scan_data_directory() at runtime.
FILE_MAPPING: Dict[str, str] = {
    'dreaddit_train': 'dreaddit_train.csv',
    'dreaddit_test':  'dreaddit_test.csv',
    'twitter':        'twitter_processed.csv',
    'twitter_gold':   'twitter_gold_processed.csv',
    'reddit_combi':   'reddit_combi_processed.csv',
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECTION HEURISTICS
# ─────────────────────────────────────────────────────────────────────────────

_HUMAN_HINTS = ['gold', 'human', 'manual', 'annotated', 'expert', 'dreaddit']
_AUTO_HINTS  = ['combi', 'auto', 'distant', 'proxy', 'weak', 'silver']

_PLATFORM_HINTS = {
    'twitter':   'twitter',
    'tweet':     'twitter',
    'reddit':    'reddit',
    'dreaddit':  'reddit',
    'facebook':  'facebook',
    'fb':        'facebook',
    'instagram': 'instagram',
    'insta':     'instagram',
    'weibo':     'weibo',
}


def infer_label_type(dataset_name: str) -> str:
    """Infer label_type from dataset name using heuristics."""
    name_lower = dataset_name.lower()
    for hint in _HUMAN_HINTS:
        if hint in name_lower:
            return 'human'
    for hint in _AUTO_HINTS:
        if hint in name_lower:
            return 'auto'
    return 'unknown'


def infer_platform(dataset_name: str) -> str:
    """Infer platform from dataset name using heuristics."""
    name_lower = dataset_name.lower()
    for hint, platform in _PLATFORM_HINTS.items():
        if hint in name_lower:
            return platform
    return 'unknown'


def _csv_to_dataset_name(filename: str) -> str:
    """
    Convert CSV filename to canonical dataset name.

    'facebook_gold_processed.csv' → 'facebook_gold'
    'dreaddit_test.csv'           → 'dreaddit_test'
    'twitter_processed.csv'       → 'twitter'
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ['_processed', '_cleaned', '_final', '_v2', '_v3']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def get_dataset_info(dataset_name: str, warn_on_unknown: bool = True) -> Dict[str, str]:
    """
    Get metadata for a dataset, auto-detecting if not in registry.

    Returns:
        {'label_type': str, 'platform': str}
    """
    if dataset_name in DATASET_INFO:
        return DATASET_INFO[dataset_name]

    lt = infer_label_type(dataset_name)
    pf = infer_platform(dataset_name)

    if lt == 'unknown' and warn_on_unknown:
        logger.warning(
            f"Dataset '{dataset_name}' not in registry and label_type could not "
            f"be inferred. Add it to tss/dataset_registry.py or use "
            f"--dataset-config. Defaulting to label_type='unknown'."
        )

    info = {'label_type': lt, 'platform': pf}
    DATASET_INFO[dataset_name] = info
    return info


def register_dataset(name: str, label_type: str, platform: str = 'unknown',
                     filename: Optional[str] = None):
    """Explicitly register a new dataset at runtime."""
    DATASET_INFO[name] = {'label_type': label_type, 'platform': platform}
    if filename:
        FILE_MAPPING[name] = filename
    logger.info(f"Registered dataset: '{name}' (label_type={label_type}, platform={platform})")


def scan_data_directory(data_dir: str) -> int:
    """
    Scan data/ directory for CSV files and auto-register any new datasets.

    This eliminates the "maintenance trap" — simply drop a new CSV in data/
    and the pipeline picks it up automatically.

    Returns:
        Number of newly discovered datasets.
    """
    discovered = 0
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        ds_name = _csv_to_dataset_name(filename)

        if ds_name not in FILE_MAPPING:
            FILE_MAPPING[ds_name] = filename
            if ds_name not in DATASET_INFO:
                lt = infer_label_type(ds_name)
                pf = infer_platform(ds_name)
                DATASET_INFO[ds_name] = {'label_type': lt, 'platform': pf}
                logger.info(f"   Auto-discovered: '{ds_name}' from {filename} "
                            f"(label_type={lt}, platform={pf})")
                discovered += 1

    if discovered > 0:
        logger.info(f"   {discovered} new dataset(s) discovered from {data_dir}/")

    return discovered


def get_eval_datasets(exclude_train: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """Return (all_eval, human_eval, auto_eval) dataset name lists."""
    all_ds = [ds for ds in DATASET_INFO
              if not (exclude_train and ds.endswith('_train'))]
    human = [ds for ds in all_ds
             if DATASET_INFO[ds]['label_type'] == 'human']
    auto = [ds for ds in all_ds
            if DATASET_INFO[ds]['label_type'] == 'auto']
    return all_ds, human, auto


def get_train_datasets() -> List[str]:
    """Return list of training dataset names."""
    return [ds for ds in DATASET_INFO if ds.endswith('_train')]


def get_test_datasets() -> List[str]:
    """Return list of non-training dataset names (for evaluation)."""
    return [ds for ds in DATASET_INFO if not ds.endswith('_train')]


def get_file_mapping() -> Dict[str, str]:
    """Return current dataset_name → filename mapping."""
    return dict(FILE_MAPPING)
