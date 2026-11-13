"""Error rates between a reference text and an ASR transcript (pure Python, no dependencies)."""
try:
    from .text_norm import normalize
except ImportError:  # inside a Kaggle kernel the files sit flat next to each other
    from text_norm import normalize


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character error rate on normalised text, spaces removed. Can exceed 1 when the hypothesis is much longer."""
    r, h = normalize(reference).replace(" ", ""), normalize(hypothesis).replace(" ", "")
    return edit_distance(r, h) / max(1, len(r))


def wer(reference, hypothesis):
    """Word error rate on normalised text."""
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(r, h) / max(1, len(r))


def corpus_rate(pairs, unit="char"):
    """Micro-averaged error rate over (reference, hypothesis) pairs: total edits / total reference length."""
    edits = total = 0
    for ref, hyp in pairs:
        r, h = normalize(ref), normalize(hyp)
        r, h = (r.replace(" ", ""), h.replace(" ", "")) if unit == "char" else (r.split(), h.split())
        edits += edit_distance(r, h)
        total += len(r)
    return edits / max(1, total)
