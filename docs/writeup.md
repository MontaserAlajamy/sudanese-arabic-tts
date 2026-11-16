# Teaching open text-to-speech to speak Sudanese, on free GPUs

*November 2026.*

Sudanese Arabic is spoken by about 40 million people, yet when I went looking for a voice model that sounds
Sudanese, I found plenty that *claim* Arabic dialects and no measurement of whether any of them actually speak
ours. So I built one: a benchmark, an open dataset and two fine-tuned models, all on free Kaggle GPUs.

## 1. The benchmark
50 sentences a Sudanese person would actually say: هسع، شنو، زول، داير, a ق that becomes [g], prices and times,
English words dropped mid-sentence ("الـ laptop دا قديم"), questions and proverbs. Each model clones four real
Sudanese voices that no model was trained on, and is scored two ways: by machines (Whisper error rate, speaker
similarity, a naturalness predictor) and by a native listener in a blind test, where every clip is unlabelled and
shuffled.

## 2. The surprise: machines rank dialects upside down
The model Whisper understood best sounded the *least* Sudanese of all (1.17 out of 5). Across all open models,
the more Sudanese a clip sounded, the *worse* Whisper read it (correlation +0.29). Whisper understands MSA-like
speech best, so a low error rate rewards exactly the accent we were trying to get away from.

Lesson: for dialect speech, an ASR score tells you the words are there, not that the dialect is. You need a
native ear, and you need it blind.

Of seven systems (five open, two commercial), only two open models carried a real Sudanese accent, both
community fine-tunes from the lahgtna project. One of them (MIT-licensed) had the accent but got a word wrong in
11 of 12 sentences.

## 3. The data
Other Sudanese speech sets exist on Hugging Face (Lisan, arbml), but they state no licence, and one more is
non-commercial only. The only one I found with a clear open licence that allows any use is the Sudanese part of
Meta's Omnilingual ASR Corpus (CC-BY-4.0): 12 speakers, about 10 hours of spontaneous talk in long takes. I force-aligned every transcript
word by word and cut it at natural pauses into 3,807 clips (8.8 h), cutting out hesitations and half-words
rather than leaving them in. The train speakers and the benchmark speakers never overlap.

Released as **sudanese-arabic-speech** (https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech), CC-BY-4.0, with a data card that states what was and was not
checked.

## 4. Fine-tuning on a free GPU
Both accent-carrying models were fine-tuned on the 4.7 hours of clips that pass the quality filter:

- **ft1** (lahgtna-omnivoice, non-commercial licence): 37 minutes on 2 × T4.
- **ft2** (lahgtna-chatterbox, MIT): 65 minutes on one T4.

Fitting them on a 15 GB T4 took an 8-bit optimiser and frozen text embeddings (ft1) and small batches (ft2). One failure was not about the model at all: the job platform served my previous input files to the new
run, so a "fixed" setting never reached the GPU. A file stamped with a unique ID for each run now catches that.

## 5. Results
| | wrong-word clips (native listener) | "Sudanese?" (1-5) | "natural?" (1-5) |
|---|---|---|---|
| best commercial service | 0 of 12 | 4.33 | 4.42 |
| ft1 (non-commercial) | 0 of 12 | 3.67 | 3.42 |
| **ft2 (MIT)** | **1 of 13** | **3.62** | **3.46** |
| ft2's base model | 12 of 14 | 3.50 | 3.21 |

Fine-tuning fixed the MIT model's main flaw: wrong words fell from 12 of 14 clips to 1 of 13, and its error rate
fell from 0.47 to 0.26. It now ties the non-commercial model, so there is a free, commercially usable Sudanese
voice model. The best paid service is still about 0.7 points more Sudanese and 1 point more natural.

## 6. What's honest to say
One listener, 12 listening sentences, four voices: the direction is the same in three rounds, but the gaps
between the open models are small and not statistically solid. The next step is more listeners and more,
read-style Sudanese speech recorded with consent.

## Everything is open
- Code and benchmark: https://github.com/MontaserAlajamy/sudanese-arabic-tts (MIT)
- Dataset: https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech (CC-BY-4.0)
- Model: https://huggingface.co/MuntasirElagami/sudanese-chatterbox (MIT), with 12 unselected samples and their blind ratings
- Full results and method: `results/README.md`

Thanks to Meta FAIR's Omnilingual ASR team for releasing Sudanese speech under an open licence, and to the
lahgtna and Chatterbox authors for open dialect models.
