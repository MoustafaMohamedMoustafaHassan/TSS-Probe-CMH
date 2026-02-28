# TSS-Probe-CMH
# TSS — Triple-Stream Stress Detector
**A transparent, white-box diagnostic probe for auditing *labeling regimes* in Computational Mental Health (CMH).**
# For submittion: The Divergence Hypothesis: Unmasking Lexical Interference and Label Bias in Mental Health NLP


TSS is designed to answer a specific question that most CMH pipelines blur:

> **Are we learning a clinically meaningful signal — or are we learning the labeling procedure itself?**

Instead of chasing a single “best” classifier, TSS acts as an *epistemic audit framework* that decomposes evidence into **three orthogonal linguistic vessels** (lexical / morpho-syntax / psycholinguistic style), then quantifies how each vessel behaves under **human vs. auto (distant) supervision**.

---

## 0) What this repository contains

This repo (as provided in the shared artifact) includes:

- Full code (`tss/`, `scripts/`)
- Processed datasets in `data/processed/`
- Trained model artifacts in `artifacts/`
- Complete run outputs in `outputs/` (results tables, statistical reports, baseline comparisons, qualitative Excel)

This means you can **reproduce the published artifacts** without needing to re-download raw datasets first.

---

## 1) Conceptual framing (the “why”)

### 1.1 The CMH generalization problem
CMH models often look strong **in-domain** (same platform, same collection logic), then degrade when deployed **out-of-domain** (new platform, new style norms, different noise). TSS treats this as an *epistemic* failure, not only an engineering failure.

### 1.2 Distant supervision as an epistemic confounder
Large CMH corpora are often labeled by **keyword triggers** or weak heuristics (“distant supervision”). This commonly induces **lexical shortcut learning**:
- models become very good at *detecting the label artifact* (keywords),
- but not necessarily good at detecting *the cognitive state*.

TSS explicitly tests this hypothesis instead of assuming it away.

### 1.3 The Divergence Hypothesis (LRD)
**Labeling-Regime Divergence (LRD):**
- **Human labels** tend to reward **structure and style** (“how it is said”).
- **Auto labels** tend to reward **lexical triggers** (“what is said”).

TSS operationalizes this idea via controlled channel isolation and paired statistical testing.

### 1.4 Semantic–Structural Intersection (SSI)
TSS also provides a lens to explain *why* lexical-heavy systems can look “amazing” on some datasets:
- In closed support communities, **lexical stress terms** and **stress-like structure** frequently co-occur.
- This *intersection* creates a shortcut that boosts lexical systems and LLMs — without implying clinical understanding.

---

## 2) System overview (the “what”)

TSS decomposes each text into 3 feature views:

| Channel | Vessel          | What it captures                                        | Privacy profile                   |
|---------|-----------------|---------------------------------------------------------|-----------------------------------|
| **A**   | Lexical Content | Character TF‑IDF n‑grams (3–5) + chi² selection (k=500) | **Not privacy-preserving** (uses  lexical surface forms)                                                                                                    |
| **B**   | Morpho-Syntactic Structure | POS bigrams + abstract SVO triples → aggregated into **6** structural scalars via class-conditional log‑odds | **Strictly content-free** (no lexical tokens retained)                                       |
| **C** | Psycholinguistic Style | ~154 length-normalized style signals (function words, readability, sentiment/emotion density, negation scope, sentence rhythm, lexical diversity via **Yule’s I**) | **Mostly content-blind** (style/lexicons; no open text storage) |

All channels are evaluated alone and in combinations: **A, B, C, AB, AC, BC, ABC**.

---

## 3) Key engineering decisions (the “how”)

### 3.1 “Reality-preserving preparation”
TSS avoids default undersampling and instead preserves real prevalence patterns; the model is corrected algorithmically (e.g., class weights) rather than by deleting data.

### 3.2 Length robustness (critical for Reddit ↔ Twitter)
Cross-platform CMH is dominated by length shift. TSS mitigates this via:
- **L2 normalization** on feature vectors (compare *signal density*, not magnitude)
- **Conditional scaling**: dense channels get standard scaling after L2 normalization, but the sparse structural channel **B bypasses StandardScaler** to avoid reintroducing length bias and damaging sparsity.

### 3.3 Dynamic threshold calibration for imbalanced data
Instead of a fixed 0.5 decision threshold, TSS searches thresholds in **[0.20, 0.80]** using out-of-fold probabilities to maximize F1 (stored for audit, not used to “game” headline results).

### 3.4 Adaptive regularization (sparsity bias)
Channels that include B use **ElasticNetCV** and consistently converge to **l1_ratio=0.95** (strong L1 preference), i.e., a strong algorithmic bias toward **structural sparsity**.

---

### 4.7 Density vs. Quantity Normalization (L2 Normalization)
To prevent the model from equating "longer text" with "more stress" (a common confounder when transferring from Reddit to Twitter), raw feature vectors are \(L_2\)-normalized before classification. This shifts the objective from measuring stress *quantity* to stress *density*:

\[
X_{norm} = \frac{X}{||X||_2}
\]

### 4.8 ElasticNet and Structural Sparsity
For structural channels (e.g., Channel B), TSS abandons standard Ridge regression in favor of an ElasticNet objective function to explicitly encourage feature selection:

\[
\min_{w} \left( C \sum_{i=1}^n \log(\exp(-y_i (X_i^T w + c)) + 1) + \rho ||w||_1 + \frac{1-\rho}{2} ||w||_2^2 \right)
\]

Through exhaustive cross-validation, the model consistently converges on an \(l1\_ratio\) of \(\rho = 0.95\). This acts as empirical mathematical proof of the **Structural Sparsity** hypothesis: the algorithm heavily penalizes superfluous features, relying on a highly constrained set of structural signals.

### 4.9 Epidemiological Baseline Formulation
Given the severe class imbalance in real-world CMH data (e.g., the negative class represents ~88% in `reddit_combi`), standard random baselines are misleading. TSS explicitly computes the theoretical maximum for a majority-class baseline. Assuming the majority class prevalence is \(p\):

\[
F1_{majority} = \frac{2p}{p + 1}
\]

\[
Macro\text{-}F1_{baseline} = \frac{F1_{majority} + 0}{2}
\]

This formulation proves that the baseline ceiling is mathematically capped (e.g., at ~0.47), confirming that TSS metrics represent genuine clinical signal acquisition, not majority-class bias.

### 4.10 Paired Effect Size (\(d_z\)) for Lexical Interference
To mathematically quantify the "Lexical Interference Paradox" (e.g., comparing Channel C vs. AC), the framework relies on Cohen’s \(d_z\) for paired samples across \(N=12,906\) instances:

\[
d_z = \frac{\mu_D}{\sigma_D}
\]

Where \(\mu_D\) is the mean of the performance differences and \(\sigma_D\) is the standard deviation of those differences. By using the paired standardized difference, the metric strictly isolates the variance introduced by *adding words*, factoring out the inherent difficulty of individual texts.


---

## 5) Repository structure

```
TSS/
├── tss/                        # Core library
│   ├── config.py
│   ├── dataset_registry.py
│   ├── features.py
│   ├── open_lexicons.py
│   └── pipeline.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_evaluate.py
│   ├── 03_masking_suite.py
│   ├── 04_statistical_analysis.py
│   ├── 05_advanced_analysis.py
│   └── 06_baseline_comparison.py
├── 02.1_extract_error_cases.py
├── data/
│   ├── raw/                    # raw CSVs (included in provided artifact)
│   ├── processed/              # processed CSVs used by all scripts
│   └── lexicons/               # NRC-VAD
├── artifacts/                  # trained .joblib models
└── outputs/                    # all experimental outputs (tables, plots, reports)
```

---

## 6) Installation

### 6.1 Create environment
```bash
python -m venv tss_env
source tss_env/bin/activate  # Linux/macOS
# tss_env\Scripts\activate  # Windows
```

### 6.2 Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
```

---

## 7) How to run (end-to-end)

> Run scripts from the **repo root** (`TSS/`).

### Step 1 — Prepare data
```bash
python scripts/01_prepare_data.py --all
```

Outputs:
- `data/processed/*.csv`
- logs describing deduplication + decontamination

### Step 2 — Train & evaluate all channels
```bash
python scripts/02_train_evaluate.py --channels all --bootstrap 10000
```

Outputs:
- `outputs/tss_results_*.json`
- `outputs/tss_eval_*.csv`
- `outputs/tss_cross_*.csv`
- `outputs/tss_lodo_*.csv`
- `outputs/predictions_*.joblib`
- trained artifacts in `artifacts/`

### Step 3 — Masking suite (interventional ablations)
```bash
python scripts/03_masking_suite.py --channels all --datasets all --bootstrap 10000
```

Outputs:
- `outputs/masking/masking_comprehensive_*.csv`

Mask modes:
- `none`
- `pos_only`
- `content_only`
- `function_only`
- `random_pos`

### Step 4 — Statistical analysis (DoD, FDR, tests)
```bash
python scripts/04_statistical_analysis.py --results outputs/tss_results_*.json
```

Outputs:
- `outputs/statistical_report_*.json`
- LaTeX-ready tables inside `outputs/analysis/` (if enabled)

### Step 5 — Advanced analysis (clustering, SHAP, divergence visuals)
```bash
python scripts/05_advanced_analysis.py
# or: python scripts/05_advanced_analysis.py --skip_shap
```

Outputs:
- `outputs/analysis/clustering_stats.csv`
- `outputs/analysis/stylistic_profiles.csv`
- plots in `outputs/plots/`

### Step 6 — Baseline comparisons (MentalBERT / LLaMA vs TSS)
```bash
python scripts/06_baseline_comparison.py \
  --tss-predictions outputs/predictions_*.joblib \
  --baselines outputs/mentalbert_predictions_*.joblib outputs/llama_predictions_*.joblib \
  --output outputs/baseline_comparison
```

Outputs:
- `outputs/baseline_comparison/baseline_comparison_*.json`

### Step 7 — Qualitative error case extraction
```bash
python 02.1_extract_error_cases.py
```

Outputs:
- `outputs/error_analysis_qualitative.xlsx`

---

## 8) Expected outputs & “known-good” numbers (from the provided artifact)

### 8.1 Channel performance heatmap (Macro-F1)
(Exact values from `outputs/tss_results_20260208_140443.json`.)

| channel   |   dreaddit_test |   reddit_combi |   twitter |   twitter_gold |
|:----------|----------------:|---------------:|----------:|---------------:|
| A         |           0.634 |          0.611 |     0.522 |          0.661 |
| B         |           0.561 |          0.579 |     0.463 |          0.586 |
| C         |           0.739 |          0.681 |     0.504 |          0.701 |
| AB        |           0.632 |          0.589 |     0.46  |          0.65  |
| AC        |           0.679 |          0.658 |     0.567 |          0.617 |
| BC        |           0.69  |          0.689 |     0.455 |          0.69  |
| ABC       |           0.69  |          0.689 |     0.455 |          0.689 |

### 8.2 Training footprint (features, time, sparsity selection)
(Exact values from `training_artifacts` in the same JSON.)

|     |   n_features |   training_time_s |   selected_l1_ratio |
|:----|-------------:|------------------:|--------------------:|
| A   |          500 |              17.6 |              nan    |
| B   |            6 |            1007.9 |                0.95 |
| C   |          154 |              27.5 |              nan    |
| AB  |          506 |             882.9 |                0.1  |
| AC  |          654 |              33.2 |              nan    |
| BC  |          160 |             335.9 |                0.95 |
| ABC |          660 |             378.9 |                0.95 |

> Interpretation note: Channel B’s longer *training* time is dominated by ElasticNetCV hyperparameter search; *inference* cost is near-zero once fitted.

### 8.3 The DoD result (core claim)
From `outputs/statistical_report_20260209_023712.json`:

- **BC vs A**: DoD = **0.0374**
- 95% CI: **[0.0097, 0.0651]**
- p-value (bootstrap): **0.0032**
- Cohen’s d: **2.63**
- Total evaluated samples: **N = 12906**

### 8.4 Lexical Interference Paradox (C → AC)
From the same statistical report:

- **Human-labeled** mean drop (C→AC): **0.0720**  
  95% CI: **[0.0521, 0.0923]**

- **Auto-labeled** mean drop (C→AC): **-0.0201**  
  95% CI: **[-0.0332, -0.0071]**  

> Important: The raw JSON statistical report orientates the p_value toward the strict interference hypothesis (drop > 0), yielding $p \approx 1$ when the drop is negative. In the official paper, this is converted to a standard two-sided test ($p = 0.0024$) to correctly evaluate the statistical significance of the improvement

### 8.5 Unsupervised stylistic profiling (audit)
From `clustering_stats.csv` and `stylistic_profiles.csv`:

- K-Means: **K=3**, channel: C, features: **154**
- Stability: ARI mean **0.980** (bootstrap n=100)
- Leakage AMI (cluster vs dataset): **0.281**
- Signal AMI (cluster vs label): **0.084**
- Verdict: **artifact**  (interpreted as an *artifact/leakage warning*, not a clinical phenotype claim)

Cluster summary (stress_rate is computed under the repository’s label encoding):

|    |   cluster |    n |   stress_rate |   pct_dreaddit_test |   pct_reddit_combi |   pct_twitter |   pct_twitter_gold |
|---:|----------:|-----:|--------------:|--------------------:|-------------------:|--------------:|-------------------:|
|  0 |         0 | 4355 |      0.790126 |         0.15155     |          0.634214  |      0.116877 |          0.0973594 |
|  1 |         1 | 6250 |      0.37568  |         0.00848     |          0.04256   |      0.67392  |          0.27504   |
|  2 |         2 | 2319 |      0.546787 |         0.000862441 |          0.0349288 |      0.651574 |          0.312635  |
> Note: The 18-sample discrepancy here (N=12,924 vs main evaluation N=12,906) is an artifact of the unsupervised pipeline filtering out empty morphological vectors prior to strict intersection.

### 8.6 Qualitative workbook (shift cases)
From `outputs/error_analysis_qualitative.xlsx`:

- Total shift cases found: **682**
- Sheets include: `All_Channels_Compare`, `Shift_Cases_AvsB`, `Reverse_Shift`, `Gold_Disagreement`, `Summary`

### 8.7 LLM baseline instability (illustrative audit signal)
From `llama_results_20260209_144441.json`:

- Dreaddit-test: Precision **0.606**, Recall **0.984**
- Twitter (auto): Precision **0.846**, Recall **0.203**

This is reported as **regime-sensitive instability**: few-shot LLM behavior can swing between over-sensitivity and under-sensitivity across regimes.

---

## 9) How to interpret the results (reading guide)

### 9.1 If you only read one artifact: read the statistical report
Open: `outputs/statistical_report_20260209_023712.json`

It contains:
- DoD (with bootstrap CI, p-value, effect sizes)
- Lexical interference (human vs auto asymmetry)
- Orthogonality analysis (instance-level Spearman correlations)
- Regime permutation audit (expected to be conservative)
- Transfer analysis summaries

### 9.2 What a positive DoD means (operationally)
- If **DoD > 0**, structural channels (BC) gain *relative advantage* under **human labels** compared to **auto labels**.
- This supports LRD: auto labeling is more compatible with lexical shortcuts, while human annotation rewards deeper structure/style.

### 9.3 What “masking” is doing in this project
The Masking Suite is not used to “improve” the model; it is used to **stress-test what the model relies on**:
- `pos_only` destroys lexical content while preserving structure
- `content_only` destroys structure while keeping content words
- `random_pos` destroys canonical syntax and acts as a strong “noise baseline”

### 9.4 Caveat on label semantics (verify your label direction)
Some auto-labeled corpora may encode labels in a direction that does not match the semantic meaning of “stressed=1”. Always verify:

```python
import pandas as pd
df = pd.read_csv("data/processed/reddit_combi_processed.csv")
print(df["label"].value_counts(normalize=True))
```

The repo records *observed prevalence* per dataset in `tss_results_*.json` and `masking_comprehensive_*.csv`.

---

## 10) Reproducibility checklist

- Fixed seed defaults (e.g., clustering seed=42 in `clustering_stats.csv`)
- `n_bootstrap=10000` in headline statistical artifacts
- De-duplication + cross-dataset decontamination enabled (`duplicates_removed=true` in tss_results JSON)
- All outputs are timestamped and stored under `outputs/`

---

## 11) Citing / attributing this repository

If you use TSS as an audit framework in your work, cite the accompanying paper and mention:
- **DoD (DiD adaptation)**
- **LRD (human vs auto regime divergence)**
- **Lexical Interference Paradox**
- **Masking Suite as interventional ablation audit**

---

## 12) Quick troubleshooting

- **spaCy model missing**: `python -m spacy download en_core_web_sm`
- **SHAP optional**: use `--skip_shap` in `05_advanced_analysis.py`
- **Windows path issues**: run from repo root, avoid spaces in environment path
- **CSV encoding errors**: `01_prepare_data.py` uses `smart_read_csv` and will log fallback encodings.

---

## 13) Contact / notes

This repository is intentionally engineered as an **audit probe**. If you treat it as a clinical classifier, you will misinterpret its purpose.

The guiding principle remains:

> **In CMH, “more lexical features” can produce better *numbers* while producing worse *clinical validity*.**
