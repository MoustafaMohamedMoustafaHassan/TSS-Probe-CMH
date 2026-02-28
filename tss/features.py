"""
================================================================================
TSS: Feature Transformers
================================================================================
Three-channel feature extraction for interpretable stress detection:

    Channel A — Character TF-IDF (Lexical Surface Patterns)
        Character-level n-gram TF-IDF with SelectKBest feature selection.

    Channel B — POS Bigrams + Abstract SVO Triples (Structural Synergy)
        Content-free morpho-syntactic features: POS bigram sequences and
        abstract Subject-Verb-Object triples extracted via dependency parsing.
        Extracts ONLY POS tags, never actual words, ensuring privacy-preserving,
        domain-agnostic structural patterns.

    Channel C — Psycholinguistic Features (Style & Readability)
        Function-word distributions, readability indices, sentiment scores,
        emotion density, negation scope analysis, sentence rhythm features,
        raw intensity signals (caps ratio, elongated words, punctuation
        density), and Yule's I for length-independent lexical diversity.

Core Hypothesis:
    HOW people express stress (structure) reveals more than WHAT they say
    (content), especially when labels are human-assigned.
================================================================================
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict, Counter
from typing import Dict, FrozenSet, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from .config import (
    ALLOWED_POS_TAGS,
    NEGATION_TOKENS,
    ABSOLUTIST_TERMS,
    VALENCE_LEXICON,
    AROUSAL_LEXICON,
    FEAR_WORDS,
    ANGER_WORDS,
    SADNESS_WORDS,
    ANTICIPATION_WORDS,
    JOY_WORDS,
)

from .open_lexicons import (
    load_open_lexicons,
    extract_open_lexicon_features,
    get_all_feature_names as get_open_lexicon_names,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

FORBIDDEN_COLS: FrozenSet[str] = frozenset({
    'label', 'post_id', 'id', 'subreddit', 'subreddit_raw',
    'split', 'index', 'Unnamed: 0', 'author', 'user_id', 'thread_id'
})

# Basic POS ratios
EXPECTED_PSYCHO_COLS_BASIC: Tuple[str, ...] = (
    'pronoun_ratio', 'verb_ratio', 'adjective_ratio', 'adverb_ratio',
    'modal_ratio', 'negation_ratio', 'fragmentation_index', 'avg_sentence_length',
)

# Extended features including structural features
EXPECTED_PSYCHO_COLS_EXTENDED: Tuple[str, ...] = (
    # Basic POS ratios
    'pronoun_ratio', 'verb_ratio', 'adjective_ratio', 'adverb_ratio',
    'modal_ratio', 'negation_ratio', 'fragmentation_index', 'avg_sentence_length',
    # Length-normalized signals
    'q_mark_per_token', 'exc_mark_per_token', 'caps_word_ratio', 'elongated_per_token',
    # Psycholinguistic - normalized
    'absolutist_per_token', 'negation_per_token',
    # Emotion ratios
    'fear_per_token', 'anger_per_token', 'sadness_per_token', 'joy_per_token',
    # VAD (0-1 scale)
    'avg_valence', 'avg_arousal',
    # VADER
    'vader_compound',
    # Readability
    'flesch_kincaid_grade', 'gunning_fog_index',
    # Lexical diversity
    'type_token_ratio',
    # Negation scope
    'negated_adj_ratio', 'negated_verb_ratio',
    # SVO structural
    'svo_per_sentence', 'first_person_ratio', 'has_stress_verb',
)


# =============================================================================
# CHANNEL A: CHARACTER TF-IDF (Unchanged from )
# =============================================================================

class ChannelA_CharTFIDF(BaseEstimator, TransformerMixin):
    """
    Channel A: Character-level TF-IDF with SelectKBest.

    Captures lexical surface patterns through character n-grams,
    which can be effective for detecting writing style.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 3000,
        k_best: int = 500,
        min_df: int = 2,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.k_best = k_best
        self.min_df = min_df

        self.tfidf_: Optional[TfidfVectorizer] = None
        self.selector_: Optional[SelectKBest] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ChannelA_CharTFIDF':
        X = X.reset_index(drop=True)

        texts = X['cleaned_text'].fillna('').astype(str).values

        self.tfidf_ = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            strip_accents='unicode',
            lowercase=True,
        )

        tfidf_matrix = self.tfidf_.fit_transform(texts)

        if y is not None and self.k_best > 0:
            k = min(self.k_best, tfidf_matrix.shape[1])
            self.selector_ = SelectKBest(chi2, k=k)
            self.selector_.fit(tfidf_matrix, y)

            mask = self.selector_.get_support()
            all_names = self.tfidf_.get_feature_names_out()
            self._feature_names = [f"A_{n}" for n in all_names[mask]]
        else:
            self._feature_names = [f"A_{n}" for n in self.tfidf_.get_feature_names_out()]

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.tfidf_ is None:
            raise RuntimeError("ChannelA not fitted")

        X = X.reset_index(drop=True)

        texts = X['cleaned_text'].fillna('').astype(str).values
        tfidf_matrix = self.tfidf_.transform(texts)

        if self.selector_ is not None:
            tfidf_matrix = self.selector_.transform(tfidf_matrix)

        if sparse.issparse(tfidf_matrix):
            return tfidf_matrix.toarray().astype(np.float32)
        return tfidf_matrix.astype(np.float32)

    def get_feature_names_out(self) -> List[str]:
        return self._feature_names


# =============================================================================
# CHANNEL B: POS BIGRAMS (STRUCTURAL SYNERGY)!
# =============================================================================

class ChannelB_LexicalPolarity(BaseEstimator, TransformerMixin):
    """
    Channel B: POS Bigrams for Structural Synergy

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  Structural Synergy Architecture                            ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  PHILOSOPHY: "How sentences are structured matters more than what words are used."                    ║
    ║  "HOW sentences are structured matters more than WHAT words are used"     ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  CHANGE: Unigrams → POS Bigrams                                          ║
    ║  ├── Old: "feel_VB" (single token + POS)                                 ║
    ║  └── New: "PRON_VB" (POS sequence pattern)                               ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  RATIONALE:                                                              ║
    ║  - Pattern "PRON_VERB" captures "I feel", "I can't", "I hate"            ║
    ║  - Pattern "DET_NOUN" captures "the problem", "a disaster"               ║
    ║  - These structural patterns generalize across domains and platforms     ║
    ║  - Vocabulary is naturally small (~25×25 = 625 max), no entropy filter   ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  POS-ONLY SAFETY MODE:                                                   ║
    ║  - When is_pos_only_mode=True, all filters are disabled                  ║
    ║  - This prevents vocabulary collapse with small feature space            ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Features (6 total):
    - B_pos_mass: Sum of positive log-odds (stress-indicative patterns)
    - B_neg_mass: Sum of negative log-odds (non-stress patterns)
    - B_polarity: Ratio of positive to total mass
    - B_density: Pattern coverage (matched bigrams / total bigrams)
    - B_load: Average pattern weight
    - B_abs: Absolutist term ratio (unchanged from )
    """

    # Adaptive smoothing parameters for cross-platform transfer
    MIN_SMOOTHING = 3.0      # For very short texts (< 10 tokens)
    MAX_SMOOTHING = 20.0     # For long texts (> 80 tokens)
    REFERENCE_LENGTH = 50.0  # Typical Reddit post length

    # Lazy-loaded spaCy model for SVO extraction
    _spacy_nlp = None

    @classmethod
    def _get_spacy_nlp(cls):
        """Lazy-load spaCy model for dependency parsing (SVO extraction)."""
        if cls._spacy_nlp is None:
            try:
                import spacy
                cls._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
                logger.info("   [Channel B] spaCy model loaded for SVO extraction")
            except OSError:
                logger.warning("   [Channel B] spaCy model not found — downloading en_core_web_sm ...")
                try:
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    import spacy
                    cls._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
                    logger.info("   [Channel B] spaCy model downloaded and loaded successfully")
                except Exception as e:
                    logger.warning(f"   [Channel B] Auto-download failed: {e}. "
                                   "Run manually: python -m spacy download en_core_web_sm")
                    cls._spacy_nlp = False
            except ImportError:
                logger.warning("   [Channel B] spaCy not installed. SVO triples disabled.")
                cls._spacy_nlp = False
        return cls._spacy_nlp if cls._spacy_nlp else None

    def __init__(
        self,
        entropy_threshold: float = 0.95,
        min_freq: int = 3,
        use_lemmatization: bool = True,
        use_adaptive_smoothing: bool = True,
        is_pos_only_mode: bool = True,  #  Default to POS Bigrams!
        use_svo_triples: bool = True,   #  Abstract POS-SVO Triples
    ):
        self.entropy_threshold = entropy_threshold
        self.min_freq = min_freq
        self.use_lemmatization = use_lemmatization
        self.use_adaptive_smoothing = use_adaptive_smoothing
        self.is_pos_only_mode = is_pos_only_mode
        self.use_svo_triples = use_svo_triples
        self.logodds_: Dict[str, float] = {}

    def _extract_pos_bigrams(self, pairs: List) -> List[str]:
        """
        Extract POS bigrams from token_pos_pairs.

        Input: [('I', 'PRP', 'i'), ('feel', 'VBP', 'feel'), ('bad', 'JJ', 'bad')]
        Output: ['PRP_VBP', 'VBP_JJ']
        """
        if not pairs or len(pairs) < 2:
            return []

        bigrams = []
        prev_pos = None

        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                current_pos = item[1]  # POS tag

                if prev_pos is not None:
                    bigram = f"{prev_pos}_{current_pos}"
                    bigrams.append(bigram)

                prev_pos = current_pos

        return bigrams

    def _extract_pos_svo_triples(self, text: str) -> List[str]:
        """
        Extract ABSTRACT POS-SVO Triples from text using dependency parsing.

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║  CRITICAL: CONTENT-FREE ARCHITECTURE                   ║
        ╠══════════════════════════════════════════════════════════════════════════╣
        ║  This method extracts ONLY POS tags - NEVER actual words!                ║
        ║                                                                          ║
        ║  Example: "I hate stress" → "SVO_PRON_VERB_NOUN"                         ║
        ║           NOT: "SVO_I_hate_stress"                                       ║
        ║                                                                          ║
        ║  Privacy-preserving: No content leakage, domain-agnostic patterns        ║
        ║  Structural hypothesis: WHO did WHAT to WHOM (structurally)              ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        Output formats:
          - "SVO_PRON_VERB_NOUN" (Subject-Verb-Object)
          - "SV_PRON_VERB"       (Subject-Verb only, no object)

        Args:
            text: Raw or cleaned text string

        Returns:
            List of abstract SVO pattern strings (e.g., ["SVO_PRON_VERB_NOUN", "SV_PRON_AUX"])
        """
        nlp = self._get_spacy_nlp()
        if nlp is None or not isinstance(text, str) or len(text.strip()) == 0:
            return []

        triples = []

        try:
            # Process with spaCy (limited to first 500 chars for efficiency)
            doc = nlp(text[:500])

            for sent in doc.sents:
                # Find ROOT (typically verb)
                root = None
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                        root = token
                        break

                if root is None:
                    continue

                # Find subject (nsubj or nsubjpass)
                subject = None
                for child in root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        break

                if subject is None:
                    continue

                # Find object (dobj or pobj through prep)
                obj = None
                for child in root.children:
                    if child.dep_ == "dobj":
                        obj = child
                        break
                    elif child.dep_ == "prep":
                        # Look for pobj in prepositional phrase
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = grandchild
                                break
                        if obj:
                            break

                # ═══════════════════════════════════════════════════════════════════
                # CRITICAL: Extract ONLY POS tags - NEVER actual word text!
                # ═══════════════════════════════════════════════════════════════════
                subj_pos = subject.pos_  # e.g., "PRON", "NOUN"
                verb_pos = root.pos_      # e.g., "VERB", "AUX"

                if obj:
                    obj_pos = obj.pos_    # e.g., "NOUN", "PROPN"
                    # Full SVO triple
                    triple = f"SVO_{subj_pos}_{verb_pos}_{obj_pos}"
                else:
                    # SV pattern only (no object)
                    triple = f"SV_{subj_pos}_{verb_pos}"

                triples.append(triple)

        except Exception as e:
            logger.debug(f"[Channel B] SVO extraction error: {e}")

        return triples

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ChannelB_LexicalPolarity':
        if y is None:
            return self

        X = X.reset_index(drop=True)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        pos_counts: Dict[str, int] = defaultdict(int)
        neg_counts: Dict[str, int] = defaultdict(int)

        n_empty_pairs = 0
        n_total_features = 0

        for i, (_, row) in enumerate(X.iterrows()):
            label = y[i]
            pairs = row.get('token_pos_pairs', [])

            if isinstance(pairs, str):
                try:
                    import ast
                    pairs = ast.literal_eval(pairs)
                except:
                    pairs = []

            if not isinstance(pairs, list):
                pairs = []

            if len(pairs) == 0:
                n_empty_pairs += 1
                continue

            # ═══════════════════════════════════════════════════════════════════
            # Extract POS Bigrams instead of token_pos unigrams
            # ═══════════════════════════════════════════════════════════════════
            if self.is_pos_only_mode:
                # POS Bigram mode: structural patterns only
                features = self._extract_pos_bigrams(pairs)

                # ═══════════════════════════════════════════════════════════════
                # Add Abstract POS-SVO Triples (CONTENT-FREE!)
                # ═══════════════════════════════════════════════════════════════
                if self.use_svo_triples:
                    text = row.get('cleaned_text', '')
                    if not isinstance(text, str):
                        text = ''
                    svo_features = self._extract_pos_svo_triples(text)
                    features.extend(svo_features)
            else:
                # Legacy mode: token_pos unigrams (for backward compatibility)
                features = []
                for item in pairs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tok, pos = item[0], item[1]
                        lemma = item[2] if len(item) > 2 else tok
                        if self.use_lemmatization:
                            key = f"{lemma.lower()}_{pos}"
                        else:
                            key = f"{tok.lower()}_{pos}"
                        features.append(key)

            for key in features:
                n_total_features += 1
                if label == 1:
                    pos_counts[key] += 1
                else:
                    neg_counts[key] += 1

        # Log statistics
        all_keys_raw = set(pos_counts.keys()) | set(neg_counts.keys())
        logger.info(f"   [Channel B] Mode: {'POS Bigrams' if self.is_pos_only_mode else 'Token+POS Unigrams'}")
        logger.info(f"   [Channel B] Total unique features: {len(all_keys_raw)}")
        logger.info(f"   [Channel B] Total features processed: {n_total_features}")

        if n_empty_pairs > 0:
            logger.warning(f"   [Channel B] Empty token_pos_pairs: {n_empty_pairs}/{len(X)} ({100*n_empty_pairs/len(X):.1f}%)")

        if n_total_features == 0:
            logger.error("   [Channel B] ⚠️ NO FEATURES FOUND! Check token_pos_pairs column.")
            return self

        # Calculate log-odds with Laplace smoothing
        total_pos = sum(pos_counts.values()) + 1
        total_neg = sum(neg_counts.values()) + 1

        all_keys = set(pos_counts.keys()) | set(neg_counts.keys())

        n_filtered_freq = 0
        n_filtered_entropy = 0

        # ═══════════════════════════════════════════════════════════════════════
        # POS-Only Safety Mode - disable filters for small vocabulary
        # ═══════════════════════════════════════════════════════════════════════
        if self.is_pos_only_mode:
            # POS vocabulary is naturally small (~625 max), don't filter!
            current_min_freq = 0
            current_entropy_threshold = 1.0  # Accept everything
            logger.info(f"   [Channel B] POS-Only Safety: Filters DISABLED (vocab is naturally small)")
        else:
            current_min_freq = self.min_freq
            current_entropy_threshold = self.entropy_threshold

        for key in all_keys:
            pos_count = pos_counts.get(key, 0)
            neg_count = neg_counts.get(key, 0)
            total = pos_count + neg_count

            # Filter by minimum frequency
            if total < current_min_freq:
                n_filtered_freq += 1
                continue

            p = (pos_count + 1) / total_pos
            n = (neg_count + 1) / total_neg

            # Calculate entropy for filtering
            p_stress = pos_count / total if total > 0 else 0.5
            p_no = neg_count / total if total > 0 else 0.5

            entropy = 0
            if p_stress > 0:
                entropy -= p_stress * np.log2(p_stress + 1e-10)
            if p_no > 0:
                entropy -= p_no * np.log2(p_no + 1e-10)

            # Keep terms below entropy threshold
            if entropy < current_entropy_threshold:
                self.logodds_[key] = np.log(p / n)
            else:
                n_filtered_entropy += 1

        logger.info(f"   [Channel B] Filtered by min_freq (<{current_min_freq}): {n_filtered_freq}")
        logger.info(f"   [Channel B] Filtered by entropy (>{current_entropy_threshold:.2f}): {n_filtered_entropy}")
        logger.info(f"   [Channel B] ✓ Final vocabulary size: {len(self.logodds_)}")

        if len(self.logodds_) == 0:
            logger.error("   [Channel B] ⚠️ VOCABULARY IS EMPTY! All features will be zeros.")
        elif len(self.logodds_) < 20:
            logger.warning(f"   [Channel B] ⚠️ Small vocabulary ({len(self.logodds_)}). Expected for POS-only mode.")

        # Log top discriminative patterns
        if self.logodds_ and self.is_pos_only_mode:
            sorted_patterns = sorted(self.logodds_.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            logger.info(f"   [Channel B] Top discriminative POS patterns:")
            for pattern, score in sorted_patterns:
                direction = "STRESS" if score > 0 else "NO-STRESS"
                logger.info(f"      {pattern}: {score:+.3f} ({direction})")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)

        features = []

        for _, row in X.iterrows():
            pairs = row.get('token_pos_pairs', [])

            if isinstance(pairs, str):
                try:
                    import ast
                    pairs = ast.literal_eval(pairs)
                except:
                    pairs = []

            if not isinstance(pairs, list):
                pairs = []

            # ═══════════════════════════════════════════════════════════════════
            # Use POS Bigrams or legacy token+pos based on mode
            # ═══════════════════════════════════════════════════════════════════
            if self.is_pos_only_mode:
                feature_keys = self._extract_pos_bigrams(pairs)

                # ═══════════════════════════════════════════════════════════════
                # Add Abstract POS-SVO Triples (CONTENT-FREE!)
                # ═══════════════════════════════════════════════════════════════
                if self.use_svo_triples:
                    text = row.get('cleaned_text', '')
                    if not isinstance(text, str):
                        text = ''
                    svo_features = self._extract_pos_svo_triples(text)
                    feature_keys.extend(svo_features)

                doc_length = len(feature_keys) if feature_keys else 1
            else:
                feature_keys = []
                for item in pairs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tok, pos = item[0], item[1]
                        lemma = item[2] if len(item) > 2 else tok
                        if self.use_lemmatization:
                            key = f"{lemma.lower()}_{pos}"
                        else:
                            key = f"{tok.lower()}_{pos}"
                        feature_keys.append(key)
                doc_length = len(pairs) if pairs else 1

            # Adaptive smoothing for cross-platform transfer
            if self.use_adaptive_smoothing:
                length_ratio = doc_length / self.REFERENCE_LENGTH
                smoothing_factor = self.MIN_SMOOTHING + (self.MAX_SMOOTHING - self.MIN_SMOOTHING) * min(length_ratio, 1.0)
            else:
                smoothing_factor = 20.0

            effective_length = doc_length + smoothing_factor

            pos_m, neg_m, cnt, abs_count = 0.0, 0.0, 0, 0

            for key in feature_keys:
                if key in self.logodds_:
                    s = self.logodds_[key]
                    if s > 0:
                        pos_m += s
                    else:
                        neg_m += abs(s)
                    cnt += 1

            # Count absolutist terms (from original tokens, not POS)
            for item in pairs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    tok = item[0]
                    if isinstance(tok, str) and tok.lower() in ABSOLUTIST_TERMS:
                        abs_count += 1

            total = pos_m + neg_m

            # Log-scaled features for length invariance
            features.append([
                np.log1p(pos_m) / np.log1p(effective_length),     # B_pos_mass
                np.log1p(neg_m) / np.log1p(effective_length),     # B_neg_mass
                pos_m / total if total > 0 else 0.5,              # B_polarity
                cnt / effective_length,                            # B_density
                np.log1p(total) / max(np.log1p(cnt), 1),          # B_load
                abs_count / effective_length                       # B_abs
            ])

        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self) -> List[str]:
        return ["B_pos_mass", "B_neg_mass", "B_polarity", "B_density", "B_load", "B_abs"]


# =============================================================================
# CHANNEL C: PSYCHOLINGUISTIC FEATURES 
# =============================================================================

class ChannelC_PsychoLinguistic(BaseEstimator, TransformerMixin):
    """
    Channel C: Psycholinguistic features with LENGTH NORMALIZATION.
    """

    def __init__(
        self,
        use_extended_features: bool = True,
        use_svd: bool = False,
        svd_components: int = 0,
        transfer_mode: bool = True,
    ) -> None:
        self.use_extended_features = use_extended_features
        self.transfer_mode = transfer_mode

        if transfer_mode:
            self.use_svd = False
            self.svd_components = 0
        else:
            self.use_svd = use_svd
            self.svd_components = svd_components

        self.tfidf_: Optional[TfidfVectorizer] = None
        self.svd_: Optional[TruncatedSVD] = None

        if use_extended_features:
            self.expected_cols_ = EXPECTED_PSYCHO_COLS_EXTENDED
        else:
            self.expected_cols_ = EXPECTED_PSYCHO_COLS_BASIC

    def _enforce_schema(self, df: pd.DataFrame) -> np.ndarray:
        """Enforce fixed schema - ensures exact feature count."""
        result = pd.DataFrame(0.0, index=range(len(df)), columns=list(self.expected_cols_))

        df_reset = df.reset_index(drop=True)

        for col in self.expected_cols_:
            if col in df_reset.columns:
                result[col] = pd.to_numeric(df_reset[col], errors='coerce').fillna(0.0).values

        return result.values.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ChannelC_PsychoLinguistic':
        X = X.reset_index(drop=True)

        if self.use_svd and "pos_sequence" in X.columns:
            pos_seq = X["pos_sequence"].fillna("").astype(str).values

            self.tfidf_ = TfidfVectorizer(
                ngram_range=(3, 3),
                max_features=1000,
                min_df=max(1, int(len(X) * 0.01))
            )

            try:
                mat = self.tfidf_.fit_transform(pos_seq)
                n_comp = min(self.svd_components, max(1, mat.shape[1] - 1))
                self.svd_ = TruncatedSVD(n_components=n_comp, random_state=42)
                self.svd_.fit(mat)
            except Exception:
                self.use_svd = False

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        psycho = self._enforce_schema(X)

        if self.use_svd and self.tfidf_ is not None and self.svd_ is not None:
            pos_seq = X["pos_sequence"].fillna("").astype(str).values
            try:
                svd_feat = self.svd_.transform(self.tfidf_.transform(pos_seq))
                return np.hstack([psycho, svd_feat.astype(np.float32)])
            except Exception:
                pass

        return psycho

    def get_feature_names_out(self) -> List[str]:
        psycho_names = [f"C_{c}" for c in self.expected_cols_]

        if self.use_svd and self.svd_ is not None:
            svd_names = [f"C_svd_{i}" for i in range(self.svd_.n_components)]
            return psycho_names + svd_names

        return psycho_names


# =============================================================================
# CHANNEL C EXTENDED (WITH STRUCTURAL ENHANCEMENTS)
# =============================================================================

class ChannelC_Extended(BaseEstimator, TransformerMixin):
    """
    Extended Channel C with STRUCTURAL ENHANCEMENTS

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  Structural Enhancements:                    ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  1. Punctuation N-grams:                                                 ║
    ║     - Captures "emotional intensity" through punctuation patterns        ║
    ║     - (..) vs (.) vs (?!) - different psychological signals              ║
    ║     - "!!" = emotional outburst, "..." = hesitation/depression           ║
    ║                                                                          ║
    ║  2. Structural Negation Scope:                                           ║
    ║     - NEG_VERB ("can't do") = helplessness                               ║
    ║     - NEG_ADJ ("not happy") = self-evaluation                            ║
    ║     - "How we negate things matters more than what we negate."                                  ║
    ║                                                                          ║
    ║  3. Sentence Length Rhythm:                                              ║
    ║                                                                          ║
    ║     - short_burst_ratio = fragmented thinking indicator                  ║
    ║     - Stressed speech often has erratic rhythm                           ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        use_svd: bool = False,
        svd_components: int = 0,
        transfer_mode: bool = True,
        use_open_lexicons: bool = True,
        use_length_robust_transform: bool = True,
        use_structural_features: bool = True,
        use_raw_signals: bool = True,
    ):
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.transfer_mode = transfer_mode
        self.use_open_lexicons = use_open_lexicons
        self.use_length_robust_transform = use_length_robust_transform
        self.use_structural_features = use_structural_features
        self.use_raw_signals = use_raw_signals

        self.psycho_base_ = ChannelC_PsychoLinguistic(
            use_extended_features=True,
            use_svd=False if transfer_mode else use_svd,
            svd_components=0 if transfer_mode else svd_components,
            transfer_mode=transfer_mode,
        )

        self.open_lexicon_names_ = get_open_lexicon_names(prefix="open_cat")

        # structural feature names
        self.structural_feature_names_ = [
            # Punctuation patterns
            'punct_double_period',    # ".."
            'punct_triple_period',    # "..."
            'punct_double_exclaim',   # "!!"
            'punct_question_exclaim', # "?!"
            'punct_exclaim_question', # "!?"
            # Negation scope
            'neg_verb_ratio',         # NEG + VERB patterns
            'neg_adj_ratio',          # NEG + ADJ patterns
            'neg_scope_balance',      # (neg_verb - neg_adj) / total
            # Sentence rhythm
            'std_sentence_length',    # Standard deviation of sentence lengths
            'short_burst_ratio',      # Ratio of very short sentences (< 5 words)
            'rhythm_coefficient',     # CV (coefficient of variation) of sentence lengths
        ]

        # raw signal feature names
        self.raw_signal_feature_names_ = [
            'raw_caps_ratio',           # Shouting indicator (uppercase ratio)
            'raw_elongated_count',      # Elongated words ("soooo", "nooooo")
            'raw_punctuation_density',  # Punctuation marks / word count
            'yule_i',                   # Length-independent lexical diversity (Yule's I)
        ]

    def _extract_raw_signals(self, text: str) -> List[float]:
        """
        Extract raw text signals from ORIGINAL (pre-cleaned) text.

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ : RAW SIGNALS (BEFORE TEXT CLEANING)                     ║
        ╠══════════════════════════════════════════════════════════════════════════╣
        ║  These features capture EMOTIONAL INTENSITY signals that are typically   ║
        ║  lost during text cleaning:                                              ║
        ║                                                                          ║
        ║  1. caps_ratio: Uppercase ratio → "shouting" indicator                   ║
        ║     - "I HATE THIS" → high caps_ratio → anger signal                     ║
        ║                                                                          ║
        ║  2. elongated_count: Words with repeated chars ("soooo", "nooooo")       ║
        ║     - "I feel soooo bad" → emotional emphasis                            ║
        ║                                                                          ║
        ║  3. punctuation_density: Punctuation marks / word count                  ║
        ║     - "Why??? What!!! No..." → high density → emotional intensity        ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        Args:
            text: Raw or cleaned text string

        Returns:
            List of [caps_ratio, elongated_count, punctuation_density]
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return [0.0, 0.0, 0.0]

        # 1. Caps ratio: uppercase letters / total letters
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 0:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        else:
            caps_ratio = 0.0

        # 2. Elongated word count: words with 3+ repeated characters
        # Regex: word with any character repeated 3+ times (e.g., "soooo", "nooooo")
        elongated_pattern = r'\b\w*(.)\1{2,}\w*\b'
        elongated_count = len(re.findall(elongated_pattern, text, re.IGNORECASE))

        # 3. Punctuation density: punctuation marks / word count
        words = text.split()
        word_count = max(len(words), 1)
        punct_marks = len(re.findall(r'[^\w\s]', text))
        punctuation_density = punct_marks / word_count

        return [caps_ratio, float(elongated_count), punctuation_density]

    def _calc_yule_i(self, tokens: List[str]) -> float:
        """
        Calculate Yule's I (Characteristic K) for lexical diversity.

        ╔══════════════════════════════════════════════════════════════════════════╗
        ║ : YULE'S I - LENGTH-INDEPENDENT DIVERSITY               ║
        ╠══════════════════════════════════════════════════════════════════════════╣
        ║  Unlike Type-Token Ratio (TTR), Yule's I is relatively stable across    ║
        ║  different text lengths, making it CRITICAL for cross-platform          ║
        ║  comparison (Reddit posts vs Twitter tweets).                            ║
        ║                                                                          ║
        ║  Formula: I = M1² / (M2 - M1)                                            ║
        ║  Where:                                                                  ║
        ║    M1 = Vocabulary size (number of unique words)                         ║
        ║    M2 = Σ(freq²) - sum of squared frequencies                            ║
        ║                                                                          ║
        ║  Interpretation:                                                         ║
        ║    - Higher I → More diverse vocabulary                                  ║
        ║    - Lower I → More repetitive vocabulary (potential stress signal)      ║
        ║    - Rumination often shows low lexical diversity                        ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        Args:
            tokens: List of word tokens (lowercased)

        Returns:
            Yule's I value (float), or 0.0 if calculation not possible
        """
        if not tokens or len(tokens) < 2:
            return 0.0

        # Count word frequencies
        freq_counter = Counter(tokens)

        # M1 = vocabulary size (unique words)
        M1 = len(freq_counter)

        if M1 == 0:
            return 0.0

        # M2 = sum of squared frequencies
        M2 = sum(f ** 2 for f in freq_counter.values())

        # Yule's I formula: I = M1² / (M2 - M1)
        denominator = M2 - M1
        if denominator <= 0:
            return 0.0  # Avoid division by zero

        yule_i = (M1 ** 2) / denominator

        # Normalize to reasonable range (typical values: 0-200)
        # Cap at 1000 to avoid extreme outliers
        return min(yule_i, 1000.0)

    def _extract_punctuation_ngrams(self, text: str) -> Dict[str, float]:
        """
        Extract punctuation n-gram features.

        Punctuation patterns reveal emotional state:
        - "..." = hesitation, trailing off (depression signal)
        - "!!" = emotional intensity (anger/frustration)
        - "?!" = confusion + frustration
        """
        if not isinstance(text, str) or len(text) == 0:
            return {name: 0.0 for name in [
                'punct_double_period', 'punct_triple_period',
                'punct_double_exclaim', 'punct_question_exclaim',
                'punct_exclaim_question'
            ]}

        # Count punctuation patterns
        features = {
            'punct_double_period': len(re.findall(r'\.\.(?!\.)', text)),
            'punct_triple_period': len(re.findall(r'\.{3,}', text)),
            'punct_double_exclaim': len(re.findall(r'!{2,}', text)),
            'punct_question_exclaim': len(re.findall(r'\?\s*!', text)),
            'punct_exclaim_question': len(re.findall(r'!\s*\?', text)),
        }

        # Normalize by text length (characters / 100)
        text_length = max(len(text), 1)
        normalizer = text_length / 100.0

        return {k: v / normalizer for k, v in features.items()}

    def _extract_negation_scope(self, token_pos_pairs: List) -> Dict[str, float]:
        """
        Extract structural negation scope features.

        "How we negate things matters more than what we negate."

        NEG + VERB ("can't do", "don't want") = helplessness, inability
        NEG + ADJ ("not happy", "not okay") = self-evaluation, emotional state
        """
        if not token_pos_pairs or len(token_pos_pairs) < 2:
            return {
                'neg_verb_ratio': 0.0,
                'neg_adj_ratio': 0.0,
                'neg_scope_balance': 0.0,
            }

        neg_verb_count = 0
        neg_adj_count = 0
        total_negations = 0

        # Look for negation followed by verb/adjective
        prev_is_negation = False

        for item in token_pos_pairs:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            tok, pos = str(item[0]).lower(), str(item[1])

            # Check if current token follows a negation
            if prev_is_negation:
                if pos.startswith('VB'):  # Any verb tag
                    neg_verb_count += 1
                elif pos.startswith('JJ'):  # Any adjective tag
                    neg_adj_count += 1

            # Check if this token is a negation
            prev_is_negation = tok in NEGATION_TOKENS or tok in {"n't", "not", "no", "never"}
            if prev_is_negation:
                total_negations += 1

        total_neg_scope = neg_verb_count + neg_adj_count

        return {
            'neg_verb_ratio': neg_verb_count / max(total_negations, 1),
            'neg_adj_ratio': neg_adj_count / max(total_negations, 1),
            'neg_scope_balance': (neg_verb_count - neg_adj_count) / max(total_neg_scope, 1),
        }

    def _extract_sentence_rhythm(self, text: str) -> Dict[str, float]:
        """
        Extract sentence length rhythm features.

        "A stressed person often speaks in short, rapid bursts."

        Stressed speech often shows:
        - High variance in sentence length (erratic thinking)
        - Many short "bursts" (fragmented thoughts)
        - Irregular rhythm (high coefficient of variation)
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'std_sentence_length': 0.0,
                'short_burst_ratio': 0.0,
                'rhythm_coefficient': 0.0,
            }

        # Split into sentences (simple heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return {
                'std_sentence_length': 0.0,
                'short_burst_ratio': 0.0,
                'rhythm_coefficient': 0.0,
            }

        # Calculate word counts per sentence
        sentence_lengths = [len(s.split()) for s in sentences]

        if len(sentence_lengths) == 0:
            return {
                'std_sentence_length': 0.0,
                'short_burst_ratio': 0.0,
                'rhythm_coefficient': 0.0,
            }

        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

        # Short burst ratio: sentences with < 5 words
        short_bursts = sum(1 for l in sentence_lengths if l < 5)
        short_burst_ratio = short_bursts / len(sentence_lengths)

        # Coefficient of variation (normalized std)
        cv = std_length / max(mean_length, 1.0)

        return {
            'std_sentence_length': std_length,
            'short_burst_ratio': short_burst_ratio,
            'rhythm_coefficient': cv,
        }

    def _length_robust_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply length-robust transformation to reduce variance in short texts.
        """
        return np.tanh(features * 2.0)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ChannelC_Extended':
        X = X.reset_index(drop=True)
        self.psycho_base_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        base = self.psycho_base_.transform(X)

        # Open lexicon features
        if self.use_open_lexicons:
            open_features = []
            for _, row in X.iterrows():
                text = row.get('cleaned_text', '')
                if not isinstance(text, str):
                    text = ''

                feat_dict = extract_open_lexicon_features(text, prefix="open_cat")
                feat_values = [feat_dict.get(name, 0.0) for name in self.open_lexicon_names_]
                open_features.append(feat_values)

            open_array = np.array(open_features, dtype=np.float32)
            combined = np.hstack([base, open_array])
        else:
            combined = base

        # ═══════════════════════════════════════════════════════════════════════
        # NEW STRUCTURAL FEATURES
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_structural_features:
            structural_features = []

            for _, row in X.iterrows():
                text = row.get('cleaned_text', '')
                token_pos_pairs = row.get('token_pos_pairs', [])

                if isinstance(token_pos_pairs, str):
                    try:
                        import ast
                        token_pos_pairs = ast.literal_eval(token_pos_pairs)
                    except:
                        token_pos_pairs = []

                # Extract all structural features
                punct_feats = self._extract_punctuation_ngrams(text)
                neg_feats = self._extract_negation_scope(token_pos_pairs)
                rhythm_feats = self._extract_sentence_rhythm(text)

                # Combine in fixed order
                feat_row = [
                    punct_feats['punct_double_period'],
                    punct_feats['punct_triple_period'],
                    punct_feats['punct_double_exclaim'],
                    punct_feats['punct_question_exclaim'],
                    punct_feats['punct_exclaim_question'],
                    neg_feats['neg_verb_ratio'],
                    neg_feats['neg_adj_ratio'],
                    neg_feats['neg_scope_balance'],
                    rhythm_feats['std_sentence_length'],
                    rhythm_feats['short_burst_ratio'],
                    rhythm_feats['rhythm_coefficient'],
                ]
                structural_features.append(feat_row)

            structural_array = np.array(structural_features, dtype=np.float32)
            combined = np.hstack([combined, structural_array])

            logger.debug(f"[Channel C] Added {len(self.structural_feature_names_)} structural features")

        # ═══════════════════════════════════════════════════════════════════════
        # RAW SIGNALS + YULE'S I
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_raw_signals:
            raw_features = []

            for _, row in X.iterrows():
                # Use raw text if available, otherwise fall back to cleaned text
                raw_text = row.get('text', row.get('cleaned_text', ''))
                if not isinstance(raw_text, str):
                    raw_text = ''

                # Get cleaned text for Yule's I calculation
                cleaned_text = row.get('cleaned_text', '')
                if not isinstance(cleaned_text, str):
                    cleaned_text = ''

                # Extract raw signals (caps_ratio, elongated_count, punctuation_density)
                raw_sigs = self._extract_raw_signals(raw_text)

                # Calculate Yule's I for lexical diversity
                tokens = cleaned_text.lower().split()
                yule_i = self._calc_yule_i(tokens)

                # Combine: [caps_ratio, elongated_count, punct_density, yule_i]
                raw_features.append(raw_sigs + [yule_i])

            raw_array = np.array(raw_features, dtype=np.float32)
            combined = np.hstack([combined, raw_array])

            logger.debug(f"[Channel C] Added {len(self.raw_signal_feature_names_)} raw signal features")

        # Length-robust transformation
        if self.use_length_robust_transform:
            combined = self._length_robust_transform(combined)

        return combined

    def get_feature_names_out(self) -> List[str]:
        names = list(self.psycho_base_.get_feature_names_out())

        if self.use_open_lexicons:
            names += [f"C_{n}" for n in self.open_lexicon_names_]

        if self.use_structural_features:
            names += [f"C_{n}" for n in self.structural_feature_names_]

        if self.use_raw_signals:
            names += [f"C_{n}" for n in self.raw_signal_feature_names_]

        return names


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ChannelA_CharTFIDF',
    'ChannelB_LexicalPolarity',
    'ChannelC_PsychoLinguistic',
    'ChannelC_Extended',
    'FORBIDDEN_COLS',
    'EXPECTED_PSYCHO_COLS_BASIC',
    'EXPECTED_PSYCHO_COLS_EXTENDED',
]
