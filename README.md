# Sudanese Arabic TTS: benchmark, open data and fine-tuning on free GPUs

Work in progress (started November 2026). Sudanese Arabic is spoken by about 40 million people, yet open
text-to-speech models barely cover it, and nobody had measured how well the ones that claim it actually speak it.
This project does that in four steps, each with code, data and numbers you can rerun:

1. **Benchmark**: 50 Sudanese sentences (ق → [g], everyday words such as هسع / شنو / زول / داير, numbers, English
   code-switching, questions), spoken zero-shot by five open models in four unseen real Sudanese voices, scored by
   machines and by a native listener in a blind test.
2. **Data**: Sudanese speech we are allowed to republish, cut into TTS-ready clips by forced alignment, with a data
   card that says where every clip comes from and how much of it a native speaker checked.
3. **Fine-tuning** the best base model on that data with free Kaggle GPUs (2 × T4).
4. **Release** of the dataset, the models and the results.

## Release v0.1
- **Dataset**: [sudanese-arabic-speech](https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech): 3,807 clips, 8.8 h, 12 speakers, CC-BY-4.0
- **Model**: [sudanese-chatterbox](https://huggingface.co/MuntasirElagami/sudanese-chatterbox) (ft2): MIT, with 12 unselected samples and their blind ratings
- **Results**: [results/README.md](results/README.md) · **Write-up**: [docs/writeup.md](docs/writeup.md)

## Models in the benchmark
| Engine | Model | How it is asked for Sudanese | Licence of the weights |
|---|---|---|---|
| `chatterbox` | Resemble AI Chatterbox Multilingual (t3 v3) | Arabic (`ar`) | MIT |
| `lahgtna` | oddadmix/lahgtna-chatterbox-v1 | `sd` (Chatterbox's `pt` slot) | MIT |
| `habibi` | SWivid/Habibi-TTS Unified (F5-TTS) | dialect ID `SDN` | CC-BY-NC-SA-4.0 |
| `omnivoice` | k2-fsa/OmniVoice | language `apd` | CC-BY-NC |
| `lahgtna-omnivoice` | oddadmix/lahgtna-omnivoice-v2 | `sd` | follows OmniVoice (CC-BY-NC) |

Upstream code is pinned to exact commits in `jobs/bake_job.py`.

## Data
- **Omnilingual ASR Corpus, Sudanese Arabic (`apd_Arab`)** by Meta FAIR, CC-BY-4.0: 12 speakers, 1,097 spontaneous
  recordings, 10.05 hours, 48 kHz, human transcripts. The benchmark's reference voices and its "human" topline come
  from the 4 dev/test speakers only, so models fine-tuned on the 8 train speakers are still tested on unseen voices.
- No audio of any identifiable public person is cloned or imitated.

## Layout
```
jobs/          scripts that run on a Kaggle GPU (bake = speak the benchmark, eval = score clips, prep_apd = align + cut)
scripts/       local drivers: bake.sh, make_bench.py, eval.py, make_page.py
sdtts/         small pure-Python core: Arabic text normalisation, CER/WER, clip segmentation, the blind-test page
kaggle/run.sh  push any script as a private Kaggle kernel, with inputs, and pull its outputs back
bench/         the 50 sentences and the chosen reference clips
tests/         python3 tests/test_core.py
results/       benchmark, fine-tuning and blind-test results (README.md)
release/ft2/   model card + 12 samples of the MIT fine-tune (Sudanese Chatterbox v0.1)
docs/          writeup.md
```

## Run it
Needs a Kaggle account with phone verification (for GPU + internet), the `kaggle` CLI and `KAGGLE_USER` set.
```
SDTTS_APD_DIR=<parquet dir> SDTTS_IN=<dir with segment.py + job.json> SDTTS_OUTDIR=out/apd python3 jobs/prep_apd_job.py
python3 scripts/make_bench.py --apd out/apd
scripts/bake.sh habibi            # one engine per Kaggle kernel; --smoke for a 2-sentence check
python3 scripts/eval.py           # Whisper large-v3 CER, WavLM speaker similarity, UTMOS, signal checks
python3 scripts/train.py --engine lahgtna --apd-train out/apd --apd-heldout out/apd   # fine-tune on Kaggle (detached)
python3 scripts/make_dataset.py --apd out/apd   # Hugging Face dataset: parquet shards + data card
```

## How the scores are read
Whisper is weak on Sudanese, so a CER is only meaningful next to the CER of real Sudanese speech, scored by the same
job (the `human` row). UTMOS was trained on English and Chinese ratings and is only a rough naturalness proxy. The
blind native-listener test decides; the machine scores catch broken output.

## Licence
Code: MIT. Benchmark sentences: CC-BY-4.0. Audio derived from the Omnilingual ASR Corpus: CC-BY-4.0, credit
"Omnilingual ASR Team, Meta FAIR (2025)". Each model keeps its own licence (table above).
