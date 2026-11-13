"""Cut a long spontaneous take into TTS-sized clips, given word timings from forced alignment.

Pure Python (no torch), so the same file runs inside a Kaggle kernel and in the local tests.
A transcript token is a word ("w"), a tag such as <hesitation> or <noise> ("tag"), or a cut-off word like "bed-"
("frag"). Tags and fragments are never spoken in a clip: they end a fluent run, and clips are cut inside runs only.
"""
import re

PUNCT = ".,!?;:،؛؟…\"'«»()[]{}“”‘’ـ"
STRONG = ".!?؟…"
WEAK = "،,;؛:"
CLOSERS = "\"'»)]}”’"
_TAG = re.compile(r"(<[^>]*>)")


def tokens_of(text):
    """Split a transcript into tokens. Punctuation standing alone is attached to the word before it."""
    toks = []
    for part in _TAG.split(text):
        if _TAG.fullmatch(part):
            toks.append({"t": part, "kind": "tag"})
            continue
        for raw in part.split():
            core = raw.strip(PUNCT)
            if not core:
                if toks and toks[-1]["kind"] == "w":
                    toks[-1]["t"] += raw
            elif core.endswith("-") or core.startswith("-"):
                toks.append({"t": raw, "kind": "frag"})
            else:
                toks.append({"t": raw, "kind": "w"})
    for i, t in enumerate(toks):
        t["i"] = i
    return toks


def runs_of(toks):
    """Maximal stretches of consecutive words (no tag or fragment inside)."""
    runs, cur = [], []
    for t in toks:
        if t["kind"] == "w":
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _ending(word):
    last = word.rstrip(CLOSERS)[-1:]
    return 0.5 if last in STRONG else 0.2 if last and last in WEAK else 0.0


def cut_points(run, min_s=3.0, max_s=14.0, good_s=5.0, pause=0.18):
    """Greedy cuts of one run into clips of min_s..max_s seconds, each time at the best pause.

    Every word needs "s" and "e" (seconds). Returns [(first, last, hard)] with inclusive word indexes into `run`;
    hard=True means no pause of at least `pause` seconds was available and the cut fell on the widest gap.
    """
    clips, a, n = [], 0, len(run)
    while a < n:
        best = forced = None
        last_fit = a
        for b in range(a, n):
            d = run[b]["e"] - run[a]["s"]
            if d > max_s:
                break
            last_fit = b
            if d < min_s:
                continue
            at_end = b + 1 == n
            gap = 9.0 if at_end else run[b + 1]["s"] - run[b]["e"]
            left = 0.0 if at_end else run[-1]["e"] - run[b + 1]["s"]
            if gap >= pause:
                score = gap + _ending(run[b]["t"]) + (0.3 if d >= good_s else 0.0) - (0.5 if 0 < left < min_s else 0.0)
                if best is None or score > best[0]:
                    best = (score, b)
            if forced is None or gap > forced[0]:
                forced = (gap, b)
        if best is not None:
            b, hard = best[1], False
        elif forced is not None:
            b, hard = forced[1], True
        else:  # what is left is shorter than min_s, or a single word is longer than max_s
            b, hard = last_fit, last_fit + 1 < n
        clips.append((a, b, hard))
        a = b + 1
    return clips


def clip_bounds(toks, first, last, total, lead=0.15, tail=0.25):
    """Start and end time of a clip: the words plus up to lead/tail seconds of the silence around them (half the gap)."""
    before = first["s"] - (toks[first["i"] - 1]["e"] if first["i"] > 0 else 0.0)
    after = (toks[last["i"] + 1]["s"] if last["i"] + 1 < len(toks) else total) - last["e"]
    start = max(0.0, first["s"] - min(lead, max(0.0, before) / 2))
    end = min(total, last["e"] + min(tail, max(0.0, after) / 2))
    return start, end


def next_to_cut(toks, first, last):
    """True when the clip touches a tag or fragment (its edge may carry a trace of the sound that was cut out)."""
    return (first["i"] > 0 and toks[first["i"] - 1]["kind"] != "w") or \
           (last["i"] + 1 < len(toks) and toks[last["i"] + 1]["kind"] != "w")


def letters(text):
    """Number of letters in a text, ignoring spaces and punctuation (for characters-per-second)."""
    return len(re.sub(r"[\s%s]" % re.escape(PUNCT), "", text))
