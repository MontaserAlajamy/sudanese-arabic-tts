# Run the demo free on Google Colab (GPU, public link)
1. colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU** → Save.
2. Paste this into one cell and run it (about 3-4 minutes the first time):
```
!pip -q install transformers==5.20.2 diffusers==0.42.0 s3tokenizer==0.3.0 resemble-perth conformer==0.3.2 omegaconf librosa safetensors pykakasi spacy-pkuseg einops soundfile gradio
import os, shutil
from huggingface_hub import snapshot_download
d = snapshot_download("MuntasirElagami/sudanese-chatterbox", allow_patterns=["demo/*"])
shutil.copytree(d + "/demo", "/content/demo", dirs_exist_ok=True)
os.chdir("/content/demo"); os.environ["SHARE"] = "1"
!python app.py
```
3. Open the `https://….gradio.live` link it prints. Anyone with the link can use it for up to 72 hours,
   while the notebook stays open (Colab stops idle sessions after a while).
