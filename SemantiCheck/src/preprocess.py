import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalizes, tokenizes, filters stopwords, and lemmatizes input text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [_LEMMATIZER.lemmatize(w) for w in tokens if w not in _STOP_WORDS]
    else:
        tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)
