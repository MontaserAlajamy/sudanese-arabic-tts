# Results (November 2026)

Everything here can be rerun with the code in this repository. Two commercial TTS services were included as private
reference points; they are anonymised as **Commercial A** and **Commercial B**, and their audio is not published.

## Setup
- **Benchmark**: 50 Sudanese sentences (`bench/sentences.jsonl`, checked by a native speaker): everyday words
  (هسع، شنو، زول، داير), ق said as [g], numbers, English code-switching, questions, proverbs.
- **Voices**: every system clones one of 4 real Sudanese speakers who are *not* in any training data (the dev/test
  speakers of the Omnilingual ASR corpus, `apd_Arab`). "Human" = 42 real clips of the same 4 speakers.
- **Machine scores** (`jobs/eval_job.py`): Whisper large-v3 character error rate (CER), WavLM speaker similarity,
  UTMOS22.
- **Blind listening**: one native Sudanese listener, 12 sentences, every system's clip under a hidden name in a
  shuffled order, loudness matched, edge silence trimmed. Two 1-5 scales ("Sudanese?", "natural?"), a flag for a
  wrong or missing word, and whether ق was said as [g]. Rounds 2 and 3 replay 3 clips to check the listener.

## 1. Zero-shot: which open model already speaks Sudanese?
| system | CER ↓ | median CER | UTMOS | blind: Sudanese | blind: natural | wrong-word flags | licence |
|---|---|---|---|---|---|---|---|
| Commercial A | – | – | – | **4.08** | **3.58** | 2/12 | paid |
| lahgtna-omnivoice (`sd`) | 0.139 | 0.100 | 3.26 | 3.33 | 2.83 | 1/12 | CC-BY-NC |
| lahgtna-chatterbox (`sd`) | 0.486 | 0.437 | 3.35 | 3.25 | 2.92 | 11/12 | MIT |
| OmniVoice (`apd`) | 0.128 | 0.107 | 3.16 | 2.67 | 2.33 | 0/12 | CC-BY-NC |
| Habibi-TTS (`SDN`) | 0.099 | 0.058 | 3.16 | 2.00 | 2.42 | 1/12 | CC-BY-NC-SA |
| Commercial B | – | – | – | 1.50 | 2.00 | 2/12 | paid |
| Chatterbox Multilingual (`ar`) | **0.081** | 0.064 | 3.58 | 1.17 | 1.08 | 1/12 | MIT |
| *human (real Sudanese)* | 0.134 | 0.093 | 2.56 | – | – | – | – |

**Finding: speech recognition ranks dialect TTS upside down.** Over the 60 open-model clips, the listener's
Sudanese score correlates *positively* with Whisper's error rate (r = +0.29). The model Whisper understands best
(Chatterbox, CER 0.081) sounds least Sudanese (1.17), because Whisper reads MSA-like speech most easily. A low CER
shows the words are there; only a native listener can say whether the dialect is.

Only the two lahgtna models (community fine-tunes for Arabic dialects) carry the accent. lahgtna-chatterbox has it
but says a wrong or missing word in 11 of 12 clips. Both became fine-tuning bases.

## 2. Fine-tuning on 4.7 hours of open Sudanese speech
Training data: the 8 Omnilingual train speakers, 1,956 clips, 4.72 h after filtering (`scripts/train.py`), on a
free Kaggle GPU.

| model | base | licence | training | CER ↓ (capped at 1 per clip) | median CER |
|---|---|---|---|---|---|
| **ft2** | lahgtna-chatterbox | **MIT** | 3 epochs, 1 × T4, 65 min | 0.474 → **0.264** | 0.437 → **0.216** |
| ft1 | lahgtna-omnivoice | CC-BY-NC | 1,200 steps, 2 × T4, 37 min | 0.139 → **0.121** | 0.100 → **0.093** |

ft2 got better than its base on 38 of the 50 sentences and worse on 4. The uncapped ft2 mean (0.507) is dominated by one clip where
Whisper looped a phrase dozens of times (CER 9.7); capping each clip at 1 removes that artefact.

## 3. Blind rounds 2 and 3: does fine-tuning make it more Sudanese?
Scores are compared only within a round: the same clips heard in two rounds scored about 0.2-0.4 higher the second time, since a rating
depends on the clips around it.

**Round 2** (ft1 vs its base)

| system | Sudanese | natural | wrong-word flags |
|---|---|---|---|
| Commercial A | 4.25 | 4.25 | 0 |
| **ft1** | **4.00** | **3.67** | 0 |
| lahgtna-omnivoice (base) | 3.75 | 3.33 | 0 |

**Round 3** (ft2 vs its base, with ft1 and Commercial A)

| system | Sudanese | natural | wrong-word flags | ق as [g] |
|---|---|---|---|---|
| Commercial A | 4.33 | 4.42 | 0/12 | 4/4 |
| ft1 | 3.67 | 3.42 | 0/12 | 5/5 |
| **ft2** | **3.62** | **3.46** | **1/13** | 5/5 |
| lahgtna-chatterbox (base) | 3.50 | 3.21 | 12/14 | 5/5 |

- **Fine-tuning fixed word accuracy**: wrong or missing words in lahgtna-chatterbox fell from 12 of 14 clips to
  1 of 13, matching the CER drop.
- **ft2 (MIT) ties ft1 (non-commercial)**: better on 4 sentences, worse on 4, equal on 4.
- **The gap to the best paid service is about 0.7 (Sudanese) and 1.0 (natural)** on a 5-point scale.
- Listener consistency: 5 of the 6 replayed clips scored within 1 point of their first play.

## Limits
One native listener, 12 listening sentences and 4 voices: the direction is consistent across three rounds, but
the differences between ft1, ft2 and their bases are not statistically solid. The training data is spontaneous
speech from 8 speakers; the models have not been tested on long passages.
