"""Cheap text screen for Sudanese Arabic: counts dialect marker words in a transcript.

Used to pick candidates from large Arabic corpora before any audio is downloaded or listened to. A text screen only
finds candidates: transcripts of Sudanese speech are often written in MSA spelling, and other dialects share some
words, so every candidate still needs an audio dialect-ID pass and a native listener.

STRONG markers are (nearly) unique to Sudanese Arabic; WEAK ones are also common in Gulf, Iraqi or Levantine speech
and only add support.
"""
import re

try:
    from .text_norm import normalize
except ImportError:
    from text_norm import normalize

# normalised spellings (see text_norm.normalize: ة->ه, ى->ي, hamza seats unified, no diacritics)
STRONG = {
    "زول", "الزول", "زوله", "الزوله", "هسع", "ياخ", "ساي", "عديل", "براهو", "براها", "براهم", "زاتو", "زاتي",
    "زاتها", "زاتك", "ياتو", "ياتي", "بالحيل", "حبوبه", "حبوبتي", "كيفنك", "كيفنكم", "جنا", "الجنا", "ملاح", "الملاح",
    "كسره", "الكسره",
}
WEAK = {"شنو", "منو", "داير", "دايره", "دايرين", "دي", "ديل", "سمح", "كدا", "كده", "هسه", "بتاع", "بتاعت", "تب", "نان", "قروش"}
SUDAN = {"السودان", "سوداني", "سودانيه", "سودانيين", "الخرطوم", "امدرمان", "ام درمان", "بحري", "كسلا", "الابيض",
         "بورتسودان", "دارفور", "عطبره", "مدني", "الجزيره", "شندي", "دنقلا", "النيل الازرق", "كردفان"}


def score(text):
    """Marker counts for one transcript: {"words", "strong", "strong_types", "weak", "places", "per100"}."""
    t = normalize(text)
    words = t.split()
    strong = [w for w in words if w in STRONG]
    weak = [w for w in words if w in WEAK]
    places = sum(t.count(p) for p in SUDAN)
    n = max(1, len(words))
    return {"words": len(words), "strong": len(strong), "strong_types": len(set(strong)), "weak": len(weak),
            "places": places, "per100": round(100 * (len(strong) + 0.3 * len(weak)) / n, 2)}


def is_candidate(s, min_words=60):
    """A long enough transcript with at least two different strong markers and a marker rate above 0.5 per 100 words."""
    return s["words"] >= min_words and s["strong_types"] >= 2 and s["per100"] >= 0.5
