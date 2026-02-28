"""
TSS: Transferable Stress Signals — Configuration
=================================================
Centralized configuration for all TSS components.
All word sets use FrozenSet for immutability at runtime.
"""

from typing import Dict, FrozenSet

# =============================================================================
# VERSION — single source of truth
# =============================================================================
__version__ = "1.0.0"
SYSTEM_NAME = "TSS"
SYSTEM_FULL = "Transferable Stress Signals"

# =============================================================================
# POS TAGS — allowed for Channel B bigram extraction
# =============================================================================
ALLOWED_POS_TAGS: FrozenSet[str] = frozenset({
    'NN', 'NNS', 'NNP', 'NNPS',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
    'JJ', 'JJR', 'JJS',
    'RB', 'RBR', 'RBS',
    'PRP', 'PRP$',
    'MD',
    'IN', 'TO', 'CC',
    'DT', 'WDT', 'WP', 'WRB',
})

# =============================================================================
# DOMAIN CONFIGURATION — for LODO cross-validation on Dreaddit
# =============================================================================
ALL_DOMAINS = ['anxiety', 'abuse', 'ptsd', 'social', 'financial']

DOMAIN_MAP: Dict[str, str] = {
    'anxiety': 'anxiety',
    'ptsd': 'ptsd',
    'abuse': 'abuse',
    'social': 'social',
    'financial': 'financial',
    'domesticviolence': 'abuse',
    'survivorsofabuse': 'abuse',
    'almosthomeless': 'financial',
    'assistance': 'financial',
    'food_pantry': 'financial',
    'homeless': 'financial',
    'relationships': 'social',
    'stress': 'social',
}

# =============================================================================
# LINGUISTIC RESOURCES — Negation (expanded: formal + contracted + informal)
# =============================================================================
NEGATION_TOKENS: FrozenSet[str] = frozenset({
    # Standard negation
    'no', 'not', 'none', 'never', 'neither', 'nor', 'nobody', 'nothing',
    'nowhere', 'noone',
    # Contracted negation (apostrophe forms)
    "can't", "cannot", "won't", "wouldn't", "shouldn't", "couldn't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "mustn't", "needn't", "shan't",
    "mightn't", "daren't",
    # Suffix form (spaCy tokenization)
    "n't",
    # Adverbial negation
    'hardly', 'scarcely', 'barely', 'seldom', 'rarely',
    # Informal / social-media forms (no apostrophe)
    'cant', 'wont', 'dont', 'doesnt', 'didnt', 'isnt', 'arent',
    'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'wouldnt',
    'shouldnt', 'couldnt', 'aint',
})

# =============================================================================
# ABSOLUTIST TERMS — associated with cognitive distortions in stress/depression
# =============================================================================
ABSOLUTIST_TERMS: FrozenSet[str] = frozenset({
    'always', 'never', 'completely', 'totally', 'absolutely',
    'constantly', 'definitely', 'certainly', 'entirely', 'forever',
    'nothing', 'everything', 'everyone', 'nobody', 'nowhere',
    'all', 'none', 'every', 'only', 'just', 'must', 'impossible',
    'perfect', 'terrible', 'horrible', 'worst', 'best', 'ruined',
})

# =============================================================================
# EMOTION WORD SETS (all FrozenSet for immutability)
# =============================================================================
FEAR_WORDS: FrozenSet[str] = frozenset({
    'afraid', 'fear', 'scared', 'terrified', 'anxious', 'nervous',
    'worried', 'panic', 'dread', 'horror', 'fright', 'terror',
    'apprehensive', 'uneasy', 'tense', 'alarmed', 'phobia',
    'frightened', 'paranoid', 'trembling', 'shaking',
})

ANGER_WORDS: FrozenSet[str] = frozenset({
    'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated',
    'frustrated', 'outraged', 'hostile', 'bitter', 'resentful',
    'enraged', 'livid', 'aggravated', 'infuriated', 'pissed',
    'seething', 'fuming', 'wrathful',
})

SADNESS_WORDS: FrozenSet[str] = frozenset({
    'sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'grief',
    'sorrow', 'despair', 'heartbroken', 'devastated', 'lonely',
    'melancholy', 'gloomy', 'dejected', 'down', 'blue', 'crying',
    'mourning', 'bereft', 'despondent', 'forlorn',
})

ANTICIPATION_WORDS: FrozenSet[str] = frozenset({
    'hope', 'expect', 'anticipate', 'await', 'eager', 'excited',
    'optimistic', 'curious', 'interested', 'expecting', 'wondering',
})

JOY_WORDS: FrozenSet[str] = frozenset({
    'happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted',
    'cheerful', 'content', 'satisfied', 'grateful', 'thankful',
    'blessed', 'wonderful', 'amazing', 'great', 'fantastic',
    'thrilled', 'ecstatic', 'elated',
})

STRESS_VERBS: FrozenSet[str] = frozenset({
    'worry', 'stress', 'fear', 'dread', 'panic', 'struggle',
    'suffer', 'hurt', 'hate', 'despair', 'overwhelm', 'exhaust',
    'frustrate', 'depress', 'terrify', 'scare',
    'agonize', 'obsess', 'ruminate',
})

FIRST_PERSON: FrozenSet[str] = frozenset({
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
})

# =============================================================================
# VALENCE / AROUSAL LEXICONS (expanded, grounded in Russell's Circumplex)
# =============================================================================

# Hardcoded fallback (used if NRC-VAD.txt is not found)
_VALENCE_FALLBACK: Dict[str, float] = {
    'happy': 0.90, 'love': 0.95, 'joy': 0.95, 'hope': 0.80,
    'peace': 0.85, 'grateful': 0.88, 'excited': 0.85, 'wonderful': 0.90,
    'amazing': 0.87, 'great': 0.82, 'good': 0.75, 'beautiful': 0.88,
    'delighted': 0.90, 'cheerful': 0.85, 'content': 0.78,
    'pleased': 0.80, 'blessed': 0.85, 'thankful': 0.82,
    'sad': 0.10, 'angry': 0.20, 'fear': 0.15, 'hate': 0.10,
    'anxiety': 0.20, 'despair': 0.05, 'stress': 0.15, 'depressed': 0.08,
    'miserable': 0.07, 'hopeless': 0.05, 'lonely': 0.15, 'terrified': 0.10,
    'frustrated': 0.18, 'overwhelmed': 0.12, 'exhausted': 0.20,
    'devastated': 0.06, 'worthless': 0.05, 'helpless': 0.08,
    'okay': 0.50, 'fine': 0.55, 'normal': 0.50, 'alright': 0.52,
}

_AROUSAL_FALLBACK: Dict[str, float] = {
    'angry': 0.90, 'fear': 0.85, 'excited': 0.90, 'panic': 0.95,
    'rage': 0.95, 'anxious': 0.80, 'stressed': 0.85, 'terrified': 0.90,
    'furious': 0.92, 'overwhelmed': 0.82, 'frustrated': 0.75,
    'ecstatic': 0.88, 'thrilled': 0.87, 'alarmed': 0.85,
    'agitated': 0.80, 'frantic': 0.90,
    'calm': 0.20, 'relaxed': 0.15, 'peaceful': 0.20, 'serene': 0.10,
    'content': 0.30, 'tired': 0.25, 'exhausted': 0.30, 'sleepy': 0.15,
    'bored': 0.20, 'numb': 0.15, 'depressed': 0.35, 'lonely': 0.30,
    'sad': 0.35, 'melancholy': 0.25, 'quiet': 0.18,
}


def _load_nrc_vad() -> tuple:
    """
    Load VALENCE and AROUSAL lexicons from NRC-VAD.txt (20K entries).
    Format: word<TAB>valence<TAB>arousal<TAB>dominance
    Falls back to hardcoded dictionaries if file not found.
    """
    import os
    # Search common locations for the NRC-VAD file
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'lexicons', 'NRC-VAD.txt'),
        os.path.join(os.path.dirname(__file__), 'data', 'lexicons', 'NRC-VAD.txt'),
        'data/lexicons/NRC-VAD.txt',
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            valence, arousal = {}, {}
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            word = parts[0].lower()
                            try:
                                valence[word] = float(parts[1])
                                arousal[word] = float(parts[2])
                            except ValueError:
                                continue
                if len(valence) > 100:
                    return valence, arousal
            except Exception:
                continue
    return _VALENCE_FALLBACK.copy(), _AROUSAL_FALLBACK.copy()


VALENCE_LEXICON, AROUSAL_LEXICON = _load_nrc_vad()

# =============================================================================
# MENTAL HEALTH KEYWORDS (for optional removal during preprocessing)
# =============================================================================
MENTAL_HEALTH_KEYWORDS: FrozenSet[str] = frozenset({
    'depression', 'depressed', 'anxiety', 'anxious', 'ptsd',
    'trauma', 'suicidal', 'suicide', 'therapy', 'therapist',
    'psychiatrist', 'medication', 'antidepressant', 'mental health',
    'bipolar', 'schizophrenia', 'ocd', 'adhd', 'panic attack',
    'self harm', 'cutting', 'eating disorder', 'anorexia', 'bulimia',
})
