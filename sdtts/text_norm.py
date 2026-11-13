"""Arabic text normalisation for scoring (CER/WER), so spelling habits are not counted as speech errors.

Dialect is written without a fixed spelling: the same word appears with or without hamza, with ة or ه, ى or ي.
Before comparing an ASR transcript with the reference text we therefore remove diacritics and tatweel, unify the
alef, ya, ta-marbuta and hamza-seat forms, map Arabic-Indic digits to Western ones, lowercase Latin letters and
drop punctuation. This is for scoring only: training and benchmark text are never changed by it.
"""
import re
import unicodedata

_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه", "ء": "",
    "گ": "ق", "ڤ": "ف", "پ": "ب", "چ": "ج",
    **{chr(0x0660 + i): str(i) for i in range(10)}, **{chr(0x06F0 + i): str(i) for i in range(10)},
})
_NOT_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)


def strip_diacritics(text):
    """Remove tashkeel and tatweel only (what a typed, undiacritised sentence looks like)."""
    return _DIACRITICS.sub("", text)


def normalize(text):
    """Text as compared in CER/WER."""
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text).translate(_MAP).lower()
    text = re.sub(r"<[^>]*>", " ", text)  # transcript tags such as <hesitation>
    text = _NOT_WORD.sub(" ", text)
    return " ".join(text.split())
