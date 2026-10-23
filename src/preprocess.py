"""Text cleaning utilities shared by training, prediction and the Streamlit app."""
import re

import nltk

_STOPWORDS = None


# Used only if the NLTK stop-word list cannot be downloaded (e.g. offline).
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves
""".split())


def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            print("[warn] NLTK stopwords unavailable - using built-in list.")
            _STOPWORDS = set(_FALLBACK_STOPWORDS)
        # Keep negations - they flip meaning and matter for sequence models.
        _STOPWORDS -= {"no", "not", "nor", "never"}
    return _STOPWORDS


# Leakage guard: many "real" articles start with an agency dateline like
# "WASHINGTON (Reuters) -". A model can learn this shortcut instead of
# language patterns, so we strip it.
_DATELINE_RE = re.compile(r"^.{0,80}?\((reuters|ap|afp)\)\s*-?\s*", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_MENTION_RE = re.compile(r"[@#]\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")


def strip_dateline(text: str) -> str:
    """Remove a leading agency dateline such as "WASHINGTON (Reuters) - "."""
    if not isinstance(text, str):
        return ""
    return _DATELINE_RE.sub("", text.strip())


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Lowercase, strip leakage/URLs/HTML/punctuation and optionally stop-words."""
    if not isinstance(text, str):
        return ""
    text = strip_dateline(text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("reuters", " ")
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        sw = _get_stopwords()
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    return _SPACES_RE.sub(" ", " ".join(tokens)).strip()
