"""
================================================================================
TSS: Transferable Stress Signals — Pipeline Module
================================================================================
Unified pipeline for training and evaluation of the TSS probe.

Architecture:
    TSSTransformer  — Combines Channels A, B, C into a single feature matrix.
    TSSClassifier   — Wraps transformer + normalizer + scaler + logistic
                      regression into an end-to-end classification pipeline.

Key Design Decisions:
    1. L2 Normalizer for cross-platform transfer:
       Converts feature vectors to unit vectors so that the model compares
       "stress density" rather than "stress quantity", preventing length-bias
       when transferring from long Reddit posts to short Twitter tweets.

    2. Conditional StandardScaler:
       StandardScaler is DISABLED for channels that include B (sparse POS
       features) because re-scaling after L2 normalization reintroduces
       length bias.  Dense channels (A, C, AC) use both Normalizer + Scaler.

    3. Channel-specific regularization:
       Channels containing B use ElasticNet via LogisticRegressionCV with
       5-fold internal CV over l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95].
       Channels without B (A, C, AC) use standard L2 (Ridge).

    4. Threshold calibration:
       Search range [0.20, 0.80] — deliberately wide to handle imbalanced
       LODO folds.  Domain-specific thresholds are stored in artifacts for
       optional per-domain analysis but are NOT used in headline results.

    5. SHAP accessor methods:
       get_coef, get_intercept, and transform_features enable
       post-hoc interpretability via SHAP LinearExplainer.
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, fbeta_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, Normalizer

from .features import (
    ChannelA_CharTFIDF,
    ChannelB_LexicalPolarity,
    ChannelC_Extended,
    FORBIDDEN_COLS,
)
from .config import __version__, ALL_DOMAINS, DOMAIN_MAP

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_THRESHOLD = 0.50

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThresholdOptimizer:
    """Stores optimised thresholds."""
    global_threshold: float
    domain_thresholds: Dict[str, float]
    f1_at_threshold: float

    def get_threshold(self, domain: str = 'unknown') -> float:
        return self.domain_thresholds.get(domain, self.global_threshold)


@dataclass
class TrainingArtifact:
    """Training results with regularization info."""
    channel: str
    cv_f1: float
    cv_f1_ci: Dict[str, float]
    cv_precision: float
    cv_recall: float
    cv_auc: float
    global_threshold: float
    domain_thresholds: Dict[str, float]
    n_samples: int
    n_features: int
    training_time: float
    # Regularization info for reproducibility
    regularization_type: str = 'L2'
    selected_l1_ratio: Optional[float] = None
    l1_ratios_grid: Optional[List[float]] = None

# =============================================================================
# TSS TRANSFORMER
# =============================================================================

class TSSTransformer(BaseEstimator, TransformerMixin):
    """TSS Feature Transformer combining channels A, B, C."""

    def __init__(
        self,
        channel: str = 'ABC',
        use_svd: bool = False,
        transfer_mode: bool = True,
        k_best_a: int = 500,
    ):
        self.channel = channel.upper()
        self.use_svd = use_svd
        self.transfer_mode = transfer_mode
        self.k_best_a = k_best_a

        self.channel_a_: Optional[ChannelA_CharTFIDF] = None
        self.channel_b_: Optional[ChannelB_LexicalPolarity] = None
        self.channel_c_: Optional[ChannelC_Extended] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'TSSTransformer':
        X = X.reset_index(drop=True)
        self._feature_names = []

        if 'A' in self.channel:
            self.channel_a_ = ChannelA_CharTFIDF(k_best=self.k_best_a)
            self.channel_a_.fit(X, y)
            self._feature_names.extend(self.channel_a_.get_feature_names_out())

        if 'B' in self.channel:
            self.channel_b_ = ChannelB_LexicalPolarity()
            self.channel_b_.fit(X, y)
            self._feature_names.extend(self.channel_b_.get_feature_names_out())

        if 'C' in self.channel:
            self.channel_c_ = ChannelC_Extended(
                use_svd=self.use_svd,
                transfer_mode=self.transfer_mode,
                use_open_lexicons=True,
            )
            self.channel_c_.fit(X, y)
            self._feature_names.extend(self.channel_c_.get_feature_names_out())

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        parts = []

        if self.channel_a_ is not None:
            parts.append(self.channel_a_.transform(X))
        if self.channel_b_ is not None:
            parts.append(self.channel_b_.transform(X))
        if self.channel_c_ is not None:
            parts.append(self.channel_c_.transform(X))

        if not parts:
            raise ValueError("No channels activated")

        return np.hstack(parts).astype(np.float32)

    def get_feature_names_out(self) -> List[str]:
        return self._feature_names

# =============================================================================
# TSS CLASSIFIER
# =============================================================================

class TSSClassifier(BaseEstimator, ClassifierMixin):
    """
    Complete TSS Classification Pipeline .

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  Updates:                                                           ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  1. LENGTH NORMALIZATION (L2 Normalizer) - CRITICAL FIX:                 ║
    ║     - Problem: Reddit (long) vs Twitter (short) text length mismatch     ║
    ║     - Solution: Normalizer(norm='l2') converts to unit vectors           ║
    ║     - Now model compares "stress DENSITY" not "stress QUANTITY"          ║
    ║                                                                          ║
    ║  2. CONDITIONAL SCALER :                                       ║
    ║     ╔════════════════════════════════════════════════════════════════╗   ║
    ║     ║  WHY StandardScaler DESTROYS Channel B after Normalizer:       ║   ║
    ║     ║  - Normalizer makes short/long texts equal (||v|| = 1)        ║   ║
    ║     ║  - StandardScaler then tries to make variance = 1             ║   ║
    ║     ║  - Twitter (sparse) gets multiplied by huge numbers           ║   ║
    ║     ║  - The length bias RETURNS → Recall collapses again!          ║   ║
    ║     ╚════════════════════════════════════════════════════════════════╝   ║
    ║                                                                          ║
    ║     SOLUTION:                                                            ║
    ║     - Channels WITH B (B, BC, AB, ABC): Normalizer ONLY (no Scaler)      ║
    ║     - Channels WITHOUT B (A, C, AC): Normalizer + Scaler                 ║
    ║                                                                          ║
    ║  3. Channel-specific Regularization with AUTO-TUNING:                    ║
    ║     - Channels with B: LogisticRegressionCV + ElasticNet                 ║
    ║       • 5-fold internal CV for hyperparameter selection                  ║
    ║       • l1_ratios grid: [0.1, 0.5, 0.7, 0.9, 0.95]                       ║
    ║       • Data-driven selection → no arbitrary choices                     ║
    ║     - Channels without B (A, C, AC): L2 (Ridge)                          ║
    ║                                                                          ║
    ║  4. SHAP Accessor Methods:                                               ║
    ║     - get_coef, get_intercept, transform_features                  ║
    ║     - get_selected_l1_ratio for reporting tuned hyperparameter         ║
    ║     - For SHAP LinearExplainer analysis of shift cases                   ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    # Default l1_ratios grid for ElasticNet CV
    DEFAULT_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95]

    def __init__(
        self,
        channel: str = 'ABC',
        use_svd: bool = False,
        transfer_mode: bool = True,
        C: float = 1.0,
        threshold: float = DEFAULT_THRESHOLD,
        l1_ratios: Optional[List[float]] = None,  #  Grid for CV search
    ):
        self.channel = channel
        self.use_svd = use_svd
        self.transfer_mode = transfer_mode
        self.C = C
        self.threshold = threshold
        self.l1_ratios = l1_ratios or self.DEFAULT_L1_RATIOS

        self.transformer_: Optional[TSSTransformer] = None
        self.normalizer_: Optional[Normalizer] = None  #  L2 Normalizer for length robustness
        self.scaler_: Optional[StandardScaler] = None
        self.classifier_: Optional[LogisticRegression] = None
        self.threshold_: float = threshold
        self.domain_thresholds_: Dict[str, float] = {}
        self.selected_l1_ratio_: Optional[float] = None  #  Store selected ratio

    def _create_classifier(self):
        """
        Create channel-specific classifier with auto-tuning.

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ : CHANNEL-SPECIFIC REGULARIZATION WITH AUTO-TUNING                  ║
        ╠══════════════════════════════════════════════════════════════════════════╣
        ║  Channels with B (B, BC, AB, ABC):                                       ║
        ║    - LogisticRegressionCV with ElasticNet penalty                        ║
        ║    - 5-fold internal CV to find optimal l1_ratio                         ║
        ║    - l1_ratios grid: [0.1, 0.5, 0.7, 0.9, 0.95]                          ║
        ║    - Data-driven selection → no arbitrary hyperparameter choices         ║
        ║    - Handles sparse POS+SVO feature space effectively                    ║
        ║                                                                          ║
        ║  Channels without B (A, C, AC):                                          ║
        ║    - Ridge (penalty='l2', solver='lbfgs')                                ║
        ║    - Standard approach for dense feature spaces                          ║
        ║                                                                          ║
        ╚══════════════════════════════════════════════════════════════════════════╝
        """
        # Check if channel includes B (needs ElasticNet for sparse POS+SVO features)
        uses_channel_b = 'B' in self.channel.upper()

        if uses_channel_b:
            # LogisticRegressionCV with ElasticNet for Channel B
            # Auto-tunes l1_ratio via internal 5-fold CV
            return LogisticRegressionCV(
                cv=5,                         # 5-fold internal CV for hyperparameter selection
                solver='saga',                # Required for ElasticNet
                penalty='elasticnet',         # The regularization type
                l1_ratios=self.l1_ratios,     # Grid: [0.1, 0.5, 0.7, 0.9, 0.95]
                Cs=[self.C],                  # Fix C, only tune l1_ratio
                class_weight='balanced',      # Critical for imbalanced data
                max_iter=5000,                # SAGA needs more iterations
                random_state=42,
                n_jobs=-1,                    # Use all CPU cores for speed
                scoring='f1',                 # Optimize for F1 score
                refit=True,                   # Refit with best params
            )
        else:
            # Standard L2 for Channels A, C, AC
            return LogisticRegression(
                C=self.C,
                penalty='l2',
                solver='lbfgs',
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'TSSClassifier':
        X = X.reset_index(drop=True)
        y = np.asarray(y)

        self.transformer_ = TSSTransformer(
            channel=self.channel,
            use_svd=self.use_svd,
            transfer_mode=self.transfer_mode,
        )
        features = self.transformer_.fit_transform(X, y)

        # ═══════════════════════════════════════════════════════════════════════
        # LENGTH NORMALIZATION (L2 Normalizer)
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX for cross-platform transfer (Reddit → Twitter)
        #
        # Problem:
        #   Reddit (long text): 50 stress patterns → features sum to 50
        #   Twitter (short text): 3 stress patterns → features sum to 3
        #   Model learned "high numbers = stress" from Reddit
        #   Result: Collapsed Recall on Twitter (0.099)
        #
        # Solution:
        #   Normalizer(norm='l2') converts feature vectors to unit vectors
        #   Reddit: [50 patterns] → normalized to ||v|| = 1
        #   Twitter: [3 patterns] → normalized to ||v|| = 1
        #   Now model compares "stress DENSITY" not "stress QUANTITY"
        # ═══════════════════════════════════════════════════════════════════════
        # 4. Normalize (L2) to handle text length variance
        self.normalizer_ = Normalizer(norm='l2')
        features_transformed = self.normalizer_.fit_transform(features)

        # 5. Scale features (StandardScaler) - CONDITIONAL
        # Channel B (Sparse Counts) performs better with Normalizer ONLY.
        # Other channels (A, C) benefit from Scaling.
        if 'B' not in self.channel.upper():
            self.scaler_ = StandardScaler(with_mean=False)
            features_transformed = self.scaler_.fit_transform(features_transformed)
        else:
            self.scaler_ = None  # Disable scaler for Channel B


        # Use channel-specific classifier with auto-tuning
        self.classifier_ = self._create_classifier()
        self.classifier_.fit(features_transformed, y)

        # Store selected l1_ratio from LogisticRegressionCV
        if hasattr(self.classifier_, 'l1_ratio_'):
            # FIX: Extract scalar value from numpy array
            raw_val = self.classifier_.l1_ratio_
            self.selected_l1_ratio_ = float(raw_val[0]) if hasattr(raw_val, '__len__') else float(raw_val)

            logger.info(f"   [CV] Selected l1_ratio: {self.selected_l1_ratio_:.2f} "
                       f"(from grid {self.l1_ratios})")

        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # TSS: SHAP ACCESSOR METHODS + HYPERPARAMETER REPORTING
    # ═══════════════════════════════════════════════════════════════════════════

    def get_coef(self) -> np.ndarray:
        """Get classifier coefficients for SHAP LinearExplainer."""
        if self.classifier_ is None:
            raise RuntimeError("Model not fitted")
        return self.classifier_.coef_

    def get_intercept(self) -> np.ndarray:
        """Get classifier intercept for SHAP LinearExplainer."""
        if self.classifier_ is None:
            raise RuntimeError("Model not fitted")
        return self.classifier_.intercept_

    def get_selected_l1_ratio(self) -> Optional[float]:
        """
        Get the l1_ratio selected by LogisticRegressionCV (for channels with B).

        Returns:
            float: The selected l1_ratio if Channel B was used and CV completed.
            None: If Channel B was not used or model not fitted.

        Usage in paper:
            "The l1_ratio was selected via 5-fold cross-validation from the
             grid [0.1, 0.5, 0.7, 0.9, 0.95]. For Channel B, the data-driven
             selection yielded l1_ratio = {model.get_selected_l1_ratio:.2f}."
        """
        return self.selected_l1_ratio_

    def get_regularization_info(self) -> Dict[str, Any]:
        """
        Get comprehensive regularization information for reporting.

        Returns:
            dict: Contains channel, regularization type, selected parameters.
        """
        info = {
            'channel': self.channel,
            'uses_elasticnet': 'B' in self.channel.upper(),
        }

        if info['uses_elasticnet']:
            info['regularization'] = 'ElasticNet (LogisticRegressionCV)'
            info['l1_ratios_grid'] = self.l1_ratios
            info['selected_l1_ratio'] = self.selected_l1_ratio_
            info['cv_folds'] = 5
            info['scoring'] = 'f1'
        else:
            info['regularization'] = 'L2 (Ridge)'

        return info

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame to normalized+scaled feature matrix (for SHAP)."""
        if self.transformer_ is None or self.normalizer_ is None:
            raise RuntimeError("Model not fitted")
        X = X.reset_index(drop=True)
        features = self.transformer_.transform(X)
        features_transformed = self.normalizer_.transform(features)

        # Apply scaler only if it exists (not for Channel B)
        if self.scaler_ is not None:
            features_transformed = self.scaler_.transform(features_transformed)

        return features_transformed

    def get_feature_names(self) -> List[str]:
        """Get feature names for SHAP analysis."""
        if self.transformer_ is None:
            raise RuntimeError("Model not fitted")
        return self.transformer_.get_feature_names_out()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("Model not fitted")
        X = X.reset_index(drop=True)
        features = self.transformer_.transform(X)
        features_transformed = self.normalizer_.transform(features)  # Apply L2 normalization
        if self.scaler_ is not None:
            features_transformed = self.scaler_.transform(features_transformed)

        return self.classifier_.predict_proba(features_transformed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using global threshold (always)."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = threshold

    def set_domain_thresholds(self, thresholds: Dict[str, float]) -> None:
        self.domain_thresholds_ = thresholds.copy()

# =============================================================================
# THRESHOLD OPTIMISATION
# =============================================================================

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = 'f1',
    min_threshold: float = 0.20,
    max_threshold: float = 0.80,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Range [0.20, 0.80] — deliberately wide to handle imbalanced LODO folds.
    """
    best_thr = 0.5
    best_score = 0.0

    for thr in np.arange(min_threshold, max_threshold + 0.005, 0.01):
        y_pred = (y_proba >= thr).astype(int)

        if strategy == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif strategy == 'balanced':
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            score = 1 - abs(p - r)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score


def optimize_domain_thresholds(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = 'f1',
) -> Dict[str, float]:
    """Compute per-domain thresholds (for artifact storage only)."""
    thresholds = {}

    if 'domain' not in df.columns:
        return thresholds

    df_reset = df.reset_index(drop=True)
    for domain in df_reset['domain'].unique():
        mask = (df_reset['domain'] == domain).values
        if mask.sum() < 20:
            continue
        thr, _ = optimize_threshold(y_true[mask], y_proba[mask], strategy)
        thresholds[domain] = thr

    return thresholds

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_channel(
    df: pd.DataFrame,
    channel: str,
    n_folds: int = 5,
    threshold_strategy: str = 'f1',
    C: float = 1.0,
    use_svd: bool = False,
    transfer_mode: bool = True,
) -> Tuple[TSSClassifier, TrainingArtifact, ThresholdOptimizer]:
    """
    Train TSS model for a specific channel combination.

    Threshold is calibrated on out-of-fold predictions (no leakage).
    Domain thresholds are stored in artifacts but NOT used for headline results.
    """
    print(f"\n{'='*60}")
    print(f"Training Channel: {channel}")
    print('='*60)

    start_time = time.time()
    df = df.reset_index(drop=True)
    y = df['label'].astype(int).values

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(cv.split(df, y)):
        print(f"   Fold {fold+1}/{n_folds}...")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]

        fold_model = TSSClassifier(
            channel=channel, use_svd=use_svd,
            transfer_mode=transfer_mode, C=C,

        )
        fold_model.fit(train_df, y_train)
        oof_proba[val_idx] = fold_model.predict_proba(val_df)[:, 1]

    # Global threshold from OOF predictions
    global_thr, best_f1 = optimize_threshold(y, oof_proba, threshold_strategy)
    print(f"   Global threshold: {global_thr:.3f} (OOF F1={best_f1:.3f})")

    # Domain thresholds (artifact only, not used in evaluation)
    domain_thresholds = {}
    if 'domain' in df.columns:
        domain_thresholds = optimize_domain_thresholds(df, y, oof_proba, threshold_strategy)
        for domain, thr in domain_thresholds.items():
            print(f"      {domain}: {thr:.3f}")

    oof_pred = (oof_proba >= global_thr).astype(int)

    cv_f1 = f1_score(y, oof_pred, zero_division=0)
    cv_precision = precision_score(y, oof_pred, zero_division=0)
    cv_recall = recall_score(y, oof_pred, zero_division=0)
    cv_auc = roc_auc_score(y, oof_proba)

    # Bootstrap CI
    np.random.seed(42)
    bootstrap_f1s = []
    for _ in range(1000):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        bootstrap_f1s.append(f1_score(y[idx], oof_pred[idx], zero_division=0))

    f1_ci = {
        'mean': float(np.mean(bootstrap_f1s)),
        'std': float(np.std(bootstrap_f1s)),
        'ci_lower': float(np.percentile(bootstrap_f1s, 2.5)),
        'ci_upper': float(np.percentile(bootstrap_f1s, 97.5)),
    }

    # Train final model on ALL data
    print("   Training final model on all data...")
    model = TSSClassifier(
        channel=channel, use_svd=use_svd,
        transfer_mode=transfer_mode, C=C,
    )
    model.fit(df, y)
    model.set_threshold(global_thr)
    model.set_domain_thresholds(domain_thresholds)

    training_time = time.time() - start_time
    n_features = len(model.transformer_.get_feature_names_out())

    # Get regularization info
    reg_info = model.get_regularization_info()

    artifact = TrainingArtifact(
        channel=channel, cv_f1=cv_f1, cv_f1_ci=f1_ci,
        cv_precision=cv_precision, cv_recall=cv_recall, cv_auc=cv_auc,
        global_threshold=global_thr, domain_thresholds=domain_thresholds,
        n_samples=len(df), n_features=n_features, training_time=training_time,
        # Regularization info for reproducibility
        regularization_type=reg_info.get('regularization', 'L2'),
        selected_l1_ratio=reg_info.get('selected_l1_ratio'),
        l1_ratios_grid=reg_info.get('l1_ratios_grid'),
    )
    threshold_opt = ThresholdOptimizer(
        global_threshold=global_thr,
        domain_thresholds=domain_thresholds,
        f1_at_threshold=cv_f1,
    )

    print(f"\n   Results:")
    print(f"      F1:        {cv_f1:.3f} [{f1_ci['ci_lower']:.3f}, {f1_ci['ci_upper']:.3f}]")
    print(f"      Precision: {cv_precision:.3f}")
    print(f"      Recall:    {cv_recall:.3f}")
    print(f"      AUC:       {cv_auc:.3f}")
    print(f"      Features:  {n_features}")
    print(f"      Time:      {training_time:.1f}s")

    # Print regularization info
    if reg_info.get('uses_elasticnet') and reg_info.get('selected_l1_ratio') is not None:
        print(f"      Regularization: ElasticNet (CV-tuned)")
        print(f"      Selected l1_ratio: {reg_info['selected_l1_ratio']:.2f} "
              f"(from grid {reg_info['l1_ratios_grid']})")

    return model, artifact, threshold_opt

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(
    model: TSSClassifier,
    test_df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test data using GLOBAL threshold only.

    Design: We intentionally do NOT support per-sample domain thresholding
    in evaluation to avoid ambiguity about what the reported numbers represent.
    If a custom threshold is provided, it overrides the model's stored threshold.
    """
    test_df = test_df.reset_index(drop=True)
    y_true = test_df['label'].astype(int).values
    y_proba = model.predict_proba(test_df)[:, 1]

    thr = threshold if threshold is not None else model.threshold_
    y_pred = (y_proba >= thr).astype(int)

    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'accuracy': accuracy_score(y_true, y_pred),
        'threshold': thr,
        'n_samples': len(test_df),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist(),
    }

# =============================================================================
# DEGREE OF DIVERGENCE (DoD) — core metric from the paper
# =============================================================================

def compute_dod(
    human_scores: Dict[str, float],
    auto_scores: Dict[str, float],
    channels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Degree of Divergence between human-labeled and auto-labeled datasets.

    DoD quantifies how much a channel combination's performance shifts when
    moving from human to auto labels.  Positive DoD means the channel is
    *more* effective on human data (supporting the hypothesis that structural
    features align better with human annotation).

    Args:
        human_scores: {channel: macro_f1} on human-labeled data
        auto_scores:  {channel: macro_f1} on auto-labeled data
        channels:     list of channel keys to compare (default: all common keys)

    Returns:
        dict with per-channel DoD and aggregate statistics
    """
    if channels is None:
        channels = sorted(set(human_scores.keys()) & set(auto_scores.keys()))

    dods = {}
    for ch in channels:
        h = human_scores.get(ch, 0.0)
        a = auto_scores.get(ch, 0.0)
        dods[ch] = h - a

    values = list(dods.values())

    return {
        'per_channel': dods,
        'mean_dod': float(np.mean(values)) if values else 0.0,
        'std_dod': float(np.std(values)) if values else 0.0,
        'channels': channels,
    }


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=f1_score,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    **metric_kwargs,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for any sklearn metric.

    Args:
        y_true:  ground-truth labels
        y_pred:  predicted labels (or probabilities for AUC)
        metric_fn: sklearn metric function
        n_boot:  number of bootstrap iterations
        seed:    random seed for reproducibility
        ci:      confidence level (default 0.95)

    Returns:
        dict with mean, std, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    scores = []
    alpha = (1 - ci) / 2

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx], **metric_kwargs))

    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'ci_lower': float(np.percentile(scores, alpha * 100)),
        'ci_upper': float(np.percentile(scores, (1 - alpha) * 100)),
    }


def json_default(obj):
    """JSON serializer for numpy/pandas types (use as json.dumps default)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TSSTransformer', 'TSSClassifier',
    'ThresholdOptimizer', 'TrainingArtifact',
    'train_channel', 'evaluate_model',
    'optimize_threshold', 'optimize_domain_thresholds',
    'compute_dod', 'bootstrap_metric', 'json_default',
]
