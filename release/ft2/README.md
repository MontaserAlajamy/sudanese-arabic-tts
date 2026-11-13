---
license: mit
language:
- apd
- ar
base_model: oddadmix/lahgtna-chatterbox-v1
pipeline_tag: text-to-speech
tags:
- sudanese-arabic
- apd
- chatterbox
- voice-cloning
datasets:
- MuntasirElagami/sudanese-arabic-speech
---

# Sudanese Chatterbox v0.1 (ft2)

A text-to-speech model for **Sudanese Arabic** that clones a voice from a short reference clip. It is
[lahgtna-chatterbox-v1](https://huggingface.co/oddadmix/lahgtna-chatterbox-v1) (Chatterbox Multilingual,
fine-tuned for Arabic dialects), fine-tuned again on 4.7 hours of openly licensed Sudanese speech. MIT licence:
free for research and commercial use.

## What fine-tuning changed
Tested on 50 Sudanese sentences in 4 real Sudanese voices that were never in training (full method:
[results/README.md](https://github.com/MontaserAlajamy/sudanese-arabic-tts/blob/main/results/README.md) in the [project repository](https://github.com/MontaserAlajamy/sudanese-arabic-tts)).

| | base (lahgtna-chatterbox) | **this model** | real Sudanese speech |
|---|---|---|---|
| Whisper CER, capped at 1 per clip ↓ | 0.474 | **0.264** | 0.126 |
| clips flagged "wrong or missing word" by a native listener | 12 of 14 | **1 of 13** | – |
| native listener: "Sudanese?" (1-5) | 3.50 | **3.62** | – |
| native listener: "natural?" (1-5) | 3.21 | **3.46** | – |
| ق said as [g] where the dialect wants it | 5 of 5 | 5 of 5 | – |

In the same blind round the best commercial service scored 4.33 / 4.42, so a clear gap remains.

## Samples
All 12 listening sentences of the benchmark, not hand-picked, each with the native listener's blind rating
(Sudanese / natural, 1-5). Voices are cloned from speakers of the Omnilingual ASR corpus (CC-BY-4.0).

| # | sentence | voice | rating | audio |
|---|---|---|---|---|
| 1 | يا جدو… دي فرصة، خليهو يمشي. | spk05 | 4 / 4 | [o03.mp3](samples/o03.mp3) |
| 2 | هسع ببني المشهد تلاتي الأبعاد أول: الأوضة، وكل زول في محلو. | spk08 | 2 / 2 | [o07.mp3](samples/o07.mp3) |
| 3 | التسوي كريت في القرض تلقاه في جلدها. | spk09 | 3 / 3 | [p02.mp3](samples/p02.mp3) |
| 4 | قال لي الزول دا قاعد في القهوة من الصباح، ما قام لي هسع. | spk12 | 3 / 3 | [d01.mp3](samples/d01.mp3) |
| 5 | القطر بقوم الساعة تمانية، قوم بدري عشان ما يفوتك. | spk05 | 4 / 4 | [d02.mp3](samples/d02.mp3) |
| 6 | شنو الحاصل؟ الناس دي ماشة وين بالحيل كدا؟ | spk08 | 4 / 4 | [d06.mp3](samples/d06.mp3) |
| 7 | يا زول، إنت جيت متين من الخرطوم؟ | spk09 | 4 / 4 | [d08.mp3](samples/d08.mp3) |
| 8 | أديني موية باردة، الحر دا كتلني عديل. | spk12 | 3 / 3 | [d10.mp3](samples/d10.mp3) |
| 9 | أمبارح مشينا عرس في الحلة، والدلوكة ما وقفت لي الصبح. | spk05 | 4 / 3 | [d14.mp3](samples/d14.mp3) |
| 10 | الرغيفة بقت بخمسين جنيه، والكيلو بتلاتة آلاف. | spk08 | 4 / 4 (heard twice: 4 / 3 with a wrong-word flag) | [d17.mp3](samples/d17.mp3) |
| 11 | الـ laptop دا قديم، داير أشتري واحد جديد. | spk09 | 4 / 4 | [d22.mp3](samples/d22.mp3) |
| 12 | حبوبتي كانت بتحكي لينا قصص بالليل، وكنا بننوم قبل ما تكمل. | spk12 | 4 / 4 | [d31.mp3](samples/d31.mp3) |

## Use
Needs the lahgtna-chatterbox code (pinned commit below) and a 6-10 s reference clip of the voice, used with the
speaker's consent.
```bash
git clone https://github.com/Oddadmix/lahgtna-chatterbox && cd lahgtna-chatterbox
git checkout 433cb74200b55457bffa8ee6965a02ecab546a1c && pip install -e .
# its alignment analyzer needs attention maps: in src/chatterbox/models/t3/llama_configs.py
# change attn_implementation="sdpa" to "eager"
```
```python
import torch, soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
t3 = hf_hub_download("MuntasirElagami/sudanese-chatterbox", "t3_mtl23ls_v2.safetensors")
model.t3.load_state_dict(load_file(t3, device="cuda"))
model.t3.eval()
wav = model.generate("يا زول، إنت جيت متين من الخرطوم؟", language_id="sd", audio_prompt_path="reference.wav",
                     exaggeration=0.5, temperature=0.8, cfg_weight=0.5)
sf.write("out.wav", wav.squeeze().cpu().numpy(), model.sr)
```
`language_id="sd"` is lahgtna's Sudanese slot. Plain Arabic text works best; tashkeel is not needed.

## Training
- Data: [sudanese-arabic-speech](https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech), train split
  (8 speakers), `tts_ready` clips only: 1,956 clips, 4.72 h. Derived from the Omnilingual ASR Corpus (`apd_Arab`),
  Omnilingual ASR Team, Meta FAIR (2025), CC-BY-4.0.
- Only the T3 (text-to-speech-token) model is trained; the speech decoder and voice encoder are Chatterbox's own.
- 3 epochs, 735 optimizer steps, batch 2 × gradient accumulation 4, lr 2e-5, warmup 30; one free Kaggle T4,
  65 minutes. Training loss 6.4 → 4.1. Code: `jobs/train_chatterbox_job.py`, `scripts/train.py`.

## Limits
- Speaks fast: about 9.4 letters per second against 8.5 for the real speakers, and it still drops a word now and then.
- Evaluated by one native listener on 12 sentences; the gain over the base model is clear on word accuracy and
  small on accent.
- Trained on spontaneous speech of 8 adults; not tested on long passages, children's voices or other Sudanese
  regional accents.

## Responsible use
Clone only a voice whose owner has agreed. Do not use the model to impersonate anyone or to make speech that
someone did not say look real. Note: the lahgtna-chatterbox code at this commit does not watermark its output
(the Perth watermark call is commented out); if you publish generated speech, add it back:
`wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)` on the NumPy waveform.

## Credits and licence
MIT, like Chatterbox (Resemble AI) and lahgtna-chatterbox (oddadmix). Training audio: Omnilingual ASR Team,
Meta FAIR (2025), CC-BY-4.0.
