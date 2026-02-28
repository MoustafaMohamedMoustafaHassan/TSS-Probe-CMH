"""
================================================================================
TSS Open Lexicons (Reproducible, Non-Proprietary) - FIXED
================================================================================
56 open lexical categories × 2 (count + ratio) = 112 features

Changes:
- [CRITICAL] Now counts FREQUENCY not unique matches
- [CRITICAL] Supports multi-word phrases (e.g., "kind of", "a bit")
- All counts are properly normalized by token count
================================================================================
"""

from __future__ import annotations
from typing import Dict, Set, List, Optional, Tuple
import re

# =============================================================================
# OPEN LEXICON CATEGORIES (56 categories)
# Single words and multi-word phrases are stored separately for efficient matching
# =============================================================================

# Multi-word phrases that need special handling
MULTI_WORD_PHRASES: Dict[str, List[str]] = {
    "hedging": ["kind of", "sort of", "a bit", "a little"],
}

# Single-word lexicons (will use word tokenization)
OPEN_LEXICONS: Dict[str, Set[str]] = {
    # Emotion categories
    "anger": {"angry", "furious", "rage", "hate", "mad", "irritated", "annoyed", "frustrated", "hostile", "bitter"},
    "fear": {"afraid", "scared", "terrified", "anxious", "nervous", "worried", "panic", "dread", "horror", "frightened"},
    "sadness": {"sad", "depressed", "unhappy", "miserable", "hopeless", "grief", "sorrow", "despair", "heartbroken", "lonely"},
    "joy": {"happy", "joyful", "glad", "pleased", "delighted", "cheerful", "content", "satisfied", "grateful", "blessed"},
    "surprise": {"surprised", "amazed", "astonished", "shocked", "stunned", "unexpected", "startled", "wonder"},
    "disgust": {"disgusted", "revolted", "sick", "nauseated", "repulsed", "gross", "awful", "horrible"},

    # Cognitive categories
    "certainty": {"always", "never", "definitely", "certainly", "absolutely", "completely", "totally", "undoubtedly"},
    "uncertainty": {"maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain", "doubtful", "probably"},
    "insight": {"think", "know", "realize", "understand", "feel", "believe", "consider", "recognize", "aware"},
    "causation": {"because", "cause", "effect", "therefore", "hence", "thus", "reason", "result", "consequence"},
    "discrepancy": {"should", "would", "could", "ought", "need", "want", "wish", "hope", "expect"},
    "tentative": {"maybe", "perhaps", "guess", "seem", "appear", "somewhat", "somehow", "partially"},

    # Social categories
    "family": {"family", "mother", "father", "parent", "child", "son", "daughter", "brother", "sister", "husband", "wife"},
    "friend": {"friend", "buddy", "pal", "companion", "mate", "colleague", "acquaintance", "neighbor"},
    "social": {"people", "person", "group", "community", "society", "public", "others", "everyone", "someone"},

    # Personal pronouns
    "i": {"i", "me", "my", "mine", "myself"},
    "we": {"we", "us", "our", "ours", "ourselves"},
    "you": {"you", "your", "yours", "yourself", "yourselves"},
    "they": {"they", "them", "their", "theirs", "themselves"},

    # Time categories
    "past": {"was", "were", "had", "did", "used", "ago", "yesterday", "before", "previously", "earlier"},
    "present": {"is", "are", "am", "now", "today", "currently", "presently", "ongoing"},
    "future": {"will", "going", "tomorrow", "soon", "later", "eventually", "someday", "upcoming"},

    # Work/achievement
    "work": {"work", "job", "career", "office", "boss", "employee", "coworker", "workplace", "professional"},
    "achieve": {"achieve", "accomplish", "success", "win", "goal", "complete", "finish", "progress", "improve"},
    "money": {"money", "pay", "salary", "income", "cost", "price", "expense", "afford", "financial", "debt", "broke"},

    # Health categories
    "health": {"health", "healthy", "sick", "ill", "disease", "doctor", "hospital", "medicine", "treatment", "pain"},
    "body": {"body", "head", "heart", "hand", "face", "eye", "stomach", "chest", "arm", "leg", "back"},
    "sleep": {"sleep", "tired", "exhausted", "rest", "awake", "insomnia", "nap", "bed", "dream", "nightmare"},
    "eating": {"eat", "food", "hungry", "meal", "appetite", "diet", "weight", "fat", "thin", "starving"},

    # Negative states
    "anxiety": {"anxious", "anxiety", "nervous", "worried", "stress", "stressed", "tense", "uneasy", "restless", "panic"},
    "depression": {"depressed", "depression", "hopeless", "worthless", "empty", "numb", "unmotivated", "isolated"},
    "stress": {"stress", "stressed", "overwhelmed", "pressure", "burden", "strain", "tension", "exhausted", "burnout"},
    "loneliness": {"lonely", "alone", "isolated", "abandoned", "rejected", "unwanted", "excluded", "disconnected"},

    # Positive states
    "positive_emotion": {"good", "great", "amazing", "wonderful", "fantastic", "excellent", "love", "like", "enjoy"},
    "optimism": {"hope", "hopeful", "optimistic", "positive", "confident", "believe", "faith", "trust", "expect"},
    "gratitude": {"thank", "thanks", "grateful", "appreciate", "thankful", "blessed", "fortunate", "lucky"},

    # Negation and intensifiers
    "negation": {"no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot", "without"},
    "intensifier": {"very", "really", "extremely", "incredibly", "absolutely", "completely", "totally", "utterly"},
    # Note: hedging single words only - multi-word handled separately
    "hedging": {"somewhat", "slightly", "rather", "fairly", "partially"},

    # Communication
    "communication": {"say", "said", "tell", "told", "talk", "speak", "ask", "answer", "explain", "discuss"},
    "question": {"what", "why", "how", "when", "where", "who", "which", "whose"},

    # Motion and space
    "motion": {"go", "come", "move", "walk", "run", "leave", "arrive", "return", "travel", "stay"},
    "space": {"here", "there", "where", "place", "room", "home", "house", "inside", "outside", "around"},

    # Perception
    "see": {"see", "saw", "look", "watch", "notice", "observe", "view", "appear", "visible", "sight"},
    "hear": {"hear", "heard", "listen", "sound", "voice", "noise", "loud", "quiet", "silence"},
    "feel": {"feel", "felt", "touch", "sense", "emotion", "feeling", "sensation", "experience"},

    # Cognitive processes
    "memory": {"remember", "forgot", "forget", "memory", "recall", "remind", "past", "memories"},
    "thinking": {"think", "thought", "believe", "consider", "wonder", "imagine", "assume", "guess"},
    "decision": {"decide", "decision", "choose", "choice", "option", "alternative", "select", "prefer"},

    # Relationship
    "affiliation": {"together", "join", "share", "connect", "belong", "include", "participate", "involve"},
    "conflict": {"fight", "argue", "conflict", "disagree", "problem", "issue", "trouble", "difficulty"},
    "support": {"help", "support", "care", "assist", "encourage", "comfort", "advice", "guide"},

    # Life events
    "death": {"die", "died", "death", "dead", "kill", "killed", "funeral", "loss", "grief", "mourn"},
    "life": {"life", "live", "living", "alive", "birth", "born", "exist", "survive", "lifetime"},
    "change": {"change", "different", "new", "transform", "shift", "alter", "modify", "adjust", "adapt"},
}


def get_all_feature_names(prefix: str = "open_cat") -> List[str]:
    """Return all 112 feature names."""
    names = []
    for cat in sorted(OPEN_LEXICONS.keys()):
        names.append(f"{prefix}_{cat}_count")
        names.append(f"{prefix}_{cat}_ratio")
    return names


def load_open_lexicons() -> Dict[str, Set[str]]:
    """Return a deep copy of the open lexicons dictionary."""
    return {k: v.copy() for k, v in OPEN_LEXICONS.items()}


def extract_open_lexicon_features(
    text: str,
    lexicons: Optional[Dict[str, Set[str]]] = None,
    prefix: str = "open_cat"
) -> Dict[str, float]:
    """
    Extract open lexicon features from text.

    Note:
    - Counts FREQUENCY (occurrences) not unique matches
    - Handles multi-word phrases correctly
    - Properly normalizes by token count

    Returns 112 features: 56 categories × (count + ratio)
    """
    if lexicons is None:
        lexicons = OPEN_LEXICONS

    features = {}

    if not text or not isinstance(text, str):
        text = ""

    text_lower = text.lower()

    # Tokenize for single-word matching
    words = re.findall(r'\b[a-z]+\b', text_lower)
    n_tokens = max(len(words), 1)

    # Count word frequencies for efficient lookup
    from collections import Counter
    word_freq = Counter(words)

    # Count matches for each category
    for cat in sorted(lexicons.keys()):
        vocab = lexicons[cat]

        # ✅ FIX: Count FREQUENCY not unique matches
        count = sum(word_freq.get(word, 0) for word in vocab)

        # Add multi-word phrase counts (non-overlapping, word-boundary)
        if cat in MULTI_WORD_PHRASES:
            for phrase in MULTI_WORD_PHRASES[cat]:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                count += len(re.findall(pattern, text_lower))

        features[f"{prefix}_{cat}_count"] = float(count)
        features[f"{prefix}_{cat}_ratio"] = float(count) / float(n_tokens)

    return features


def category_feature_names(prefix: str = "open_cat", categories: Optional[List[str]] = None) -> List[str]:
    """Get feature names for specified categories."""
    if categories is None:
        categories = sorted(OPEN_LEXICONS.keys())

    names = []
    for cat in categories:
        names.append(f"{prefix}_{cat}_count")
        names.append(f"{prefix}_{cat}_ratio")
    return names


# Total: 56 categories × 2 = 112 features
TOTAL_OPEN_FEATURES = len(OPEN_LEXICONS) * 2  # 112

__all__ = [
    'OPEN_LEXICONS',
    'MULTI_WORD_PHRASES',
    'load_open_lexicons',
    'extract_open_lexicon_features',
    'category_feature_names',
    'get_all_feature_names',
    'TOTAL_OPEN_FEATURES',
]
