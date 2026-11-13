"""Sudanese Chatterbox v0.1 demo: type Sudanese Arabic, pick one of four voices, hear it.

Runs on a free CPU Space. The model is lahgtna-chatterbox (MIT) fine-tuned on open Sudanese speech
(MuntasirElagami/sudanese-chatterbox). Only the four preset voices are offered: no voice upload, so the page
cannot be used to clone someone. Every output is watermarked (Resemble AI Perth).
"""
import os, subprocess, sys, time

LAHGTNA_REPO = "https://github.com/Oddadmix/lahgtna-chatterbox"
LAHGTNA_SHA = "433cb74200b55457bffa8ee6965a02ecab546a1c"
MODEL_REPO = "MuntasirElagami/sudanese-chatterbox"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.environ.get("LAHGTNA_DIR", "/tmp/lahgtna"))
MAX_CHARS = 220


def fetch_lahgtna():
    """lahgtna ships its own copy of chatterbox (dialect text rules; "sd" rides the "pt" slot), pinned."""
    if not os.path.isdir(os.path.join(SRC, ".git")):
        subprocess.run(["git", "clone", "-q", LAHGTNA_REPO, SRC], check=True)
    subprocess.run(["git", "-C", SRC, "checkout", "-q", LAHGTNA_SHA], check=True)
    cfg = os.path.join(SRC, "src/chatterbox/models/t3/llama_configs.py")
    text = open(cfg).read()
    if 'attn_implementation="sdpa"' in text:  # its alignment analyzer needs attention maps, which sdpa cannot return
        open(cfg, "w").write(text.replace('attn_implementation="sdpa"', 'attn_implementation="eager"'))
    sys.path.insert(0, os.path.join(SRC, "src"))


fetch_lahgtna()
import gradio as gr
import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"map_location": DEVICE, **k})  # checkpoints saved on GPU, loaded on CPU
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
torch.load = _load
model.t3.load_state_dict(load_file(hf_hub_download(MODEL_REPO, "t3_mtl23ls_v2.safetensors"), device=DEVICE))
model.t3.to(DEVICE).eval()

VOICES = {  # speakers of the Omnilingual ASR corpus (apd_Arab), CC-BY-4.0; not in the training data
    "صوت 1 (أعلى)": "voices/spk05.wav",
    "صوت 2 (أعلى)": "voices/spk08.wav",
    "صوت 3 (أعلى)": "voices/spk12.wav",
    "صوت 4 (أخفض)": "voices/spk09.wav",
}
EXAMPLES = [
    ["يا زول، إنت جيت متين من الخرطوم؟", "صوت 4 (أخفض)"],
    ["القطر بقوم الساعة تمانية، قوم بدري عشان ما يفوتك.", "صوت 1 (أعلى)"],
    ["حبوبتي كانت بتحكي لينا قصص بالليل، وكنا بننوم قبل ما تكمل.", "صوت 3 (أعلى)"],
    ["شنو الحاصل؟ الناس دي ماشة وين بالحيل كدا؟", "صوت 2 (أعلى)"],
    ["الـ laptop دا قديم، داير أشتري واحد جديد.", "صوت 4 (أخفض)"],
]


def speak(text, voice, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("اكتب جملة أول.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"الجملة طويلة: الحد {MAX_CHARS} حرف. قسّمها لجمل أقصر.")
    t0 = time.time()
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        wav = model.generate(text, language_id="sd", audio_prompt_path=os.path.join(HERE, VOICES[voice]),
                             exaggeration=0.5, temperature=0.8, cfg_weight=0.5)
    wav = wav.squeeze().cpu().numpy().astype(np.float32)
    wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
    print(f"{len(text)} chars -> {len(wav) / model.sr:.1f} s audio in {time.time() - t0:.0f} s", flush=True)
    return model.sr, np.clip(wav, -1, 1)


INTRO = """
# سمّعني سوداني · Sudanese Chatterbox v0.1
اكتب جملة باللهجة السودانية، اختار صوت، واضغط «اتكلم». · Type Sudanese Arabic, pick a voice, press the button.

النسخة المجانية بتشتغل على معالج عادي، فالجملة الواحدة بتاخد حوالي نص دقيقة. · Free CPU: about 30-40 s per sentence.
"""
ABOUT = """
**Model** [MuntasirElagami/sudanese-chatterbox](https://huggingface.co/MuntasirElagami/sudanese-chatterbox) (MIT) ·
**Data** [sudanese-arabic-speech](https://huggingface.co/datasets/MuntasirElagami/sudanese-arabic-speech) (CC-BY-4.0) ·
**Code & results** [GitHub](https://github.com/MontaserAlajamy/sudanese-arabic-tts)

v0.1: it speaks a little fast and can drop a word. Only these four voices are available: no one's voice can be
uploaded or cloned here. Every clip carries an inaudible AI watermark. Voices: speakers of Meta FAIR's Omnilingual
ASR corpus (CC-BY-4.0).
"""

with gr.Blocks(title="Sudanese Chatterbox") as demo:
    gr.Markdown(INTRO, rtl=True)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="الجملة", lines=3, max_length=MAX_CHARS, rtl=True,
                              placeholder="مثلاً: يا زول، إنت جيت متين من الخرطوم؟")
            voice = gr.Radio(list(VOICES), value="صوت 1 (أعلى)", label="الصوت")
            seed = gr.Number(value=7, precision=0, label="Seed (غيّرو عشان تسمع نسخة تانية)")
            go = gr.Button("اتكلم", variant="primary")
        with gr.Column():
            out = gr.Audio(label="الصوت", type="numpy", autoplay=True)
    gr.Examples(EXAMPLES, [text, voice], label="أمثلة")
    gr.Markdown(ABOUT)
    go.click(speak, [text, voice, seed], out, concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(theme=gr.themes.Soft(), share=os.environ.get("SHARE") == "1")
