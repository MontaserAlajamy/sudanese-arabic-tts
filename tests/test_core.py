"""Plain-assert tests for the pure-Python core:  python3 tests/test_core.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdtts.metrics import cer, corpus_rate, edit_distance, wer
from sdtts.segment import clip_bounds, cut_points, letters, next_to_cut, runs_of, tokens_of
from sdtts.text_norm import normalize, strip_diacritics

# --- text normalisation
assert strip_diacritics("حَبُّوبَه") == "حبوبه"
assert normalize("أَسْمَعْ بَسّ… زَمَانْ،") == "اسمع بس زمان"
assert normalize("الحديقة") == normalize("الحديقه") and normalize("إلى") == normalize("الي")
assert normalize("٢٥ <hesitation> Link!") == "25 link"
assert normalize("مسؤول") == "مسوول" and normalize("شيء") == "شي"

# --- error rates
assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3
assert cer("هسع ماشي وين", "هسع ماشي وين؟") == 0 and wer("a b c", "a x c") == 1 / 3
assert abs(cer("abcd", "abxd") - 0.25) < 1e-9
assert corpus_rate([("ab", "ab"), ("abcd", "abxx")]) == 2 / 6

# --- tokens
t = tokens_of("يعني <hesitation> أنا كنت في البي- البيت ، وبعدين مشيت.<noise>")
assert [(x["t"], x["kind"]) for x in t] == [("يعني", "w"), ("<hesitation>", "tag"), ("أنا", "w"), ("كنت", "w"), ("في", "w"),
                                            ("البي-", "frag"), ("البيت،", "w"), ("وبعدين", "w"), ("مشيت.", "w"), ("<noise>", "tag")]
assert [x["i"] for x in t] == list(range(10))
assert [[w["t"] for w in r] for r in runs_of(t)] == [["يعني"], ["أنا", "كنت", "في"], ["البيت،", "وبعدين", "مشيت."]]
assert tokens_of("<two words> كلمة")[0] == {"t": "<two words>", "kind": "tag", "i": 0}
assert letters("أنا، كنت.") == 6


def words(spec):  # spec: list of (start, end, text)
    return [{"s": s, "e": e, "t": w, "kind": "w", "i": i} for i, (s, e, w) in enumerate(spec)]


# one word per second, a long pause after word 5 (index), sentence end at index 9
run = words([(i * 1.0 + (0.6 if i > 5 else 0), i * 1.0 + 0.8 + (0.6 if i > 5 else 0), "كلمة." if i == 9 else "كلمة") for i in range(10)])
assert cut_points(run) == [(0, 9, False)]                  # 10.4 s fits: keep the whole run
assert cut_points(run, max_s=8.0) == [(0, 5, False), (6, 9, False)]   # must split: at the long pause
# no pause anywhere and too long: forced cuts on the widest gap, flagged hard
flat = words([(i * 1.0, i * 1.0 + 0.95, "كلمة") for i in range(20)])
cuts = cut_points(flat, max_s=8.0)
assert all(h for _, _, h in cuts[:-1]) and cuts[0][0] == 0 and cuts[-1][1] == 19
assert all(flat[b]["e"] - flat[a]["s"] <= 8.0 for a, b, _ in cuts)
# a short leftover is avoided when a slightly earlier cut leaves a usable tail
tail = words([(0, 1, "a"), (1.5, 2.5, "b"), (3, 4, "c"), (4.5, 5.5, "d"), (6, 7, "e"), (7.5, 8.5, "f"), (9, 10, "g"), (10.5, 11.5, "h")])
for a, b, _ in cut_points(tail, max_s=8.0):
    assert tail[b]["e"] - tail[a]["s"] >= 3.0
# a run shorter than min_s comes back whole (the caller filters it by duration)
assert cut_points(words([(0, 0.5, "a"), (0.6, 1.2, "b")])) == [(0, 1, False)]

# --- clip bounds and edges
toks = tokens_of("<noise> أنا كنت هناك")
for x, (s, e) in zip(toks, [(0.0, 1.0), (1.4, 1.8), (1.9, 2.3), (2.5, 3.0)]):
    x["s"], x["e"] = s, e
s, e = clip_bounds(toks, toks[1], toks[3], total=4.0)
assert abs(s - 1.25) < 1e-9 and abs(e - 3.25) < 1e-9      # 0.15 s lead (gap 0.4), 0.25 s tail (1.0 s of silence left)
assert next_to_cut(toks, toks[1], toks[3]) and not next_to_cut(toks, toks[2], toks[3])
print("all core tests passed")
