#!/usr/bin/env python3
"""
================================================================================
TSS: Error Analysis Extractor (Standalone) - FIXED VERSION
================================================================================
"""

import argparse
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# 1. Smart Path Setup
# =============================================================================
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# [Arabic comment removed - see English translation above]
if (current_dir / 'tss').exists():
    project_root = current_dir
# [Arabic comment removed - see English translation above]
elif (current_dir.parent / 'tss').exists():
    project_root = current_dir.parent
else:
    # [Arabic comment removed - see English translation above]
    project_root = current_dir

# [Arabic comment removed - see English translation above]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"DEBUG: Running from: {current_file}")
print(f"DEBUG: Project Root identified as: {project_root}")

# =============================================================================

# =============================================================================

# [Arabic comment removed - see English translation above]
DATASET_INFO = {
    'dreaddit_train': {'label_type': 'human', 'platform': 'reddit'},
    'dreaddit_test': {'label_type': 'human', 'platform': 'reddit'},
    'twitter': {'label_type': 'auto', 'platform': 'twitter'},
    'twitter_gold': {'label_type': 'human', 'platform': 'twitter'},
    'reddit_combi': {'label_type': 'auto', 'platform': 'reddit'},
}

# =============================================================================

# =============================================================================
try:
    # [Arabic comment removed - see English translation above]
    from tss.pipeline import TSSClassifier
    print("DEBUG: Successfully imported 'tss.pipeline'.")
except ImportError as e:
    print("\n" + "!"*80)
    print("CRITICAL IMPORT ERROR:")
    print("!"*80)
    print(f"Error Message: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    print("-" * 80)
    print("Make sure the 'tss' folder is in the project root.")
    print("!"*80 + "\n")
    sys.exit(1)

# [Arabic comment removed - see English translation above]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# [Arabic comment removed - see English translation above]
# =============================================================================
def export_error_analysis(
    eval_results_with_preds: Dict[str, Dict],
    output_dir: Path,
    datasets: Dict[str, pd.DataFrame] = None,
    top_n: int = 200
) -> str:
    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl not installed. Please run: pip install openpyxl")
        return ""

    excel_path = output_dir / 'error_analysis_qualitative.xlsx'
    TARGET_CHANNELS = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']

    shift_rows = []
    reverse_shift_rows = []
    gold_disagreement_rows = []
    all_channels_rows = []

    for ds_name, ds_results in eval_results_with_preds.items():
        texts = ds_results.get('texts', [])
        if not texts: continue

        y_true = None
        for ch in ds_results:
            if isinstance(ds_results[ch], dict) and 'y_true' in ds_results[ch]:
                y_true = np.array(ds_results[ch]['y_true'])
                break

        if y_true is None: continue

        channel_preds = {}
        available_channels = []
        for ch in TARGET_CHANNELS:
            if ch in ds_results and 'y_pred' in ds_results[ch]:
                channel_preds[ch] = np.array(ds_results[ch]['y_pred'])
                available_channels.append(ch)

        # 1. Comparison Sheet
        count = 0
        for i in range(len(texts)):
            if count >= top_n: break

            has_error = False
            current_preds = {}
            for ch in available_channels:
                pred = int(channel_preds[ch][i])
                current_preds[ch] = pred
                if pred != y_true[i]: has_error = True

            if has_error:
                row_data = {
                    'Dataset': ds_name,
                    'Label_Type': DATASET_INFO.get(ds_name, {}).get('label_type', 'unknown'),
                    'Text_Preview': str(texts[i])[:1000],
                    'True_Label': int(y_true[i])
                }
                for ch in available_channels:
                    row_data[f'Pred_{ch}'] = current_preds[ch]

                all_channels_rows.append(row_data)
                count += 1

        # 2. Shift Cases
        if 'A' in channel_preds and 'B' in channel_preds:
            a_preds = channel_preds['A']
            b_preds = channel_preds['B']

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

    # 3. Gold Disagreement
    if datasets is not None and 'twitter' in datasets and 'twitter' in eval_results_with_preds:
        twitter_results = eval_results_with_preds['twitter']
        model_human_preds = None
        used_channel = None

        for ch in ['BC', 'B', 'C']:
            if ch in twitter_results and 'y_pred' in twitter_results[ch]:
                model_human_preds = np.array(twitter_results[ch]['y_pred'])
                used_channel = ch
                break

        if model_human_preds is not None:
            twitter_labels = datasets['twitter']['label'].values
            texts = twitter_results.get('texts', [])

            mask_dis = (twitter_labels == 1) & (model_human_preds == 0)
            for idx in np.where(mask_dis)[0][:top_n]:
                 gold_disagreement_rows.append({
                    'Text': str(texts[idx])[:1000],
                    'Auto_Label': 1,
                    'Human_Model_Pred': 0,
                    'Model_Channel': used_channel,
                    'Interpretation': 'Auto label flagged stress, but Human-trained model sees none.'
                })

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        if all_channels_rows:
            pd.DataFrame(all_channels_rows).to_excel(writer, sheet_name='All_Channels_Compare', index=False)
        if shift_rows:
            pd.DataFrame(shift_rows).to_excel(writer, sheet_name='Shift_Cases_AvsB', index=False)
        if reverse_shift_rows:
            pd.DataFrame(reverse_shift_rows).to_excel(writer, sheet_name='Reverse_Shift', index=False)
        if gold_disagreement_rows:
            pd.DataFrame(gold_disagreement_rows).to_excel(writer, sheet_name='Gold_Disagreement', index=False)

    return str(excel_path)

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract Error Analysis Cases")
    parser.add_argument('--models_dir', type=str, default='artifacts', help='Directory containing .joblib models')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing processed .csv files')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to extract')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # [Arabic comment removed - see English translation above]
    if not models_dir.is_absolute(): models_dir = project_root / models_dir
    if not data_dir.is_absolute(): data_dir = project_root / data_dir
    if not output_dir.is_absolute(): output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        logger.error(f"Models directory not found at: {models_dir}")
        sys.exit(1)
    if not data_dir.exists():
        logger.error(f"Data directory not found at: {data_dir}")
        sys.exit(1)


    datasets_to_load = ['dreaddit_test', 'twitter', 'twitter_gold']
    datasets = {}

    logger.info(f"Loading datasets from: {data_dir}")
    for name in datasets_to_load:
        possible_names = [f"{name}.csv", f"{name}_processed.csv"]
        loaded = False
        for fname in possible_names:
            path = data_dir / fname
            if path.exists():
                df = pd.read_csv(path)
                df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
                datasets[name] = df
                logger.info(f"   Loaded {name}: {len(df)} samples")
                loaded = True
                break
        if not loaded:
            logger.warning(f"   ⚠️ Could not find dataset: {name}")

    if not datasets:
        logger.error("No datasets found!")
        sys.exit(1)


    channels = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    eval_results_with_preds = {}

    for name, df in datasets.items():
        eval_results_with_preds[name] = {'texts': df['cleaned_text'].tolist()}

    logger.info(f"Loading models from: {models_dir}")
    models_found = 0
    for ch in channels:
        model_path = models_dir / f"tss_{ch.lower()}.joblib"
        if model_path.exists():
            try:
                logger.info(f"   Processing Channel {ch}...")
                model = joblib.load(model_path)
                models_found += 1

                for ds_name, df in datasets.items():
                    if 'label' not in df.columns: continue
                    y_true = df['label'].astype(int).values
                    result = model.predict(df)

                    if ch not in eval_results_with_preds[ds_name]:
                        eval_results_with_preds[ds_name][ch] = {}

                    eval_results_with_preds[ds_name][ch]['y_true'] = y_true.tolist()
                    eval_results_with_preds[ds_name][ch]['y_pred'] = result.tolist()

            except Exception as e:
                logger.error(f"   Error processing model {ch}: {e}")
                traceback.print_exc()

    if models_found == 0:
        logger.error("No .joblib models found!")
        sys.exit(1)


    logger.info(f"Exporting analysis (Top {args.samples} samples)...")
    file_path = export_error_analysis(
        eval_results_with_preds,
        output_dir,
        datasets=datasets,
        top_n=args.samples
    )

    if file_path:
        logger.info("\n" + "="*60)
        logger.info(f"✅ DONE! File saved at:\n{file_path}")
        logger.info("="*60)
    else:
        logger.error("Failed to export file.")

if __name__ == "__main__":
    main()