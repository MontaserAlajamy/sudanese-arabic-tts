---
title: Sudanese Chatterbox
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
- MuntasirElagami/sudanese-chatterbox
datasets:
- MuntasirElagami/sudanese-arabic-speech
short_description: Type Sudanese Arabic, hear it spoken (open TTS, v0.1)
---

# سمّعني سوداني · Sudanese Chatterbox v0.1

Type a sentence in Sudanese Arabic, pick one of four voices, and hear it from the model before and after our
Sudanese fine-tuning, side by side. A few seconds on a GPU; about a minute for both on a CPU.

- Model: [MuntasirElagami/sudanese-chatterbox](https://huggingface.co/MuntasirElagami/sudanese-chatterbox) (MIT),
  lahgtna-chatterbox fine-tuned on 4.7 h of open Sudanese speech.
- Data: [sudanese-arabic-speech](https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech) (CC-BY-4.0).
- Code, benchmark and results: [github.com/MontaserAlajamy/sudanese-arabic-tts](https://github.com/MontaserAlajamy/sudanese-arabic-tts).

Only four preset voices (speakers of Meta FAIR's Omnilingual ASR corpus, CC-BY-4.0); no voice upload, so the page
cannot clone anyone. Every clip carries an inaudible Perth watermark.
