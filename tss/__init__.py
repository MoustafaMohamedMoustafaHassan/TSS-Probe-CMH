"""
TSS: Transferable Stress Signals
================================
Cross-platform stress detection with interpretable feature channels.

Channels:
    A — Character TF-IDF (lexical surface patterns)
    B — POS Bigrams + Abstract SVO Triples (structural synergy)
    C — Psycholinguistic features (style, readability, emotion density)
"""

from .config import (
    __version__, SYSTEM_NAME, SYSTEM_FULL,
    ALL_DOMAINS, DOMAIN_MAP,
    ALLOWED_POS_TAGS,
    NEGATION_TOKENS, ABSOLUTIST_TERMS,
    FEAR_WORDS, ANGER_WORDS, SADNESS_WORDS, JOY_WORDS,
    ANTICIPATION_WORDS, STRESS_VERBS, FIRST_PERSON,
    VALENCE_LEXICON, AROUSAL_LEXICON,
    MENTAL_HEALTH_KEYWORDS,
)

from .features import (
    ChannelA_CharTFIDF,
    ChannelB_LexicalPolarity,
    ChannelC_PsychoLinguistic,
    ChannelC_Extended,
    FORBIDDEN_COLS,
    EXPECTED_PSYCHO_COLS_BASIC,
    EXPECTED_PSYCHO_COLS_EXTENDED,
)

from .open_lexicons import (
    OPEN_LEXICONS,
    load_open_lexicons,
    extract_open_lexicon_features,
    category_feature_names,
    get_all_feature_names,
    TOTAL_OPEN_FEATURES,
)

from .pipeline import (
    TSSTransformer,
    TSSClassifier,
    ThresholdOptimizer,
    TrainingArtifact,
    train_channel,
    evaluate_model,
    optimize_threshold,
    optimize_domain_thresholds,
    compute_dod,
    bootstrap_metric,
    json_default,
)

__all__ = [
    # Config
    '__version__', 'SYSTEM_NAME', 'SYSTEM_FULL',
    'ALL_DOMAINS', 'DOMAIN_MAP', 'ALLOWED_POS_TAGS',
    'NEGATION_TOKENS', 'ABSOLUTIST_TERMS',
    'FEAR_WORDS', 'ANGER_WORDS', 'SADNESS_WORDS', 'JOY_WORDS',
    'ANTICIPATION_WORDS', 'STRESS_VERBS', 'FIRST_PERSON',
    'VALENCE_LEXICON', 'AROUSAL_LEXICON', 'MENTAL_HEALTH_KEYWORDS',
    # Features
    'ChannelA_CharTFIDF', 'ChannelB_LexicalPolarity',
    'ChannelC_PsychoLinguistic', 'ChannelC_Extended',
    'FORBIDDEN_COLS', 'EXPECTED_PSYCHO_COLS_BASIC', 'EXPECTED_PSYCHO_COLS_EXTENDED',
    # Lexicons
    'OPEN_LEXICONS', 'load_open_lexicons', 'extract_open_lexicon_features',
    'category_feature_names', 'get_all_feature_names', 'TOTAL_OPEN_FEATURES',
    # Pipeline
    'TSSTransformer', 'TSSClassifier',
    'ThresholdOptimizer', 'TrainingArtifact',
    'train_channel', 'evaluate_model',
    'optimize_threshold', 'optimize_domain_thresholds',
    'compute_dod', 'bootstrap_metric', 'json_default',
]
