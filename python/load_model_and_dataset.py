# !pip install -q transformers datasets pillow torch accelerate huggingface_hub

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
from huggingface_hub import login

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load processor and model
model_id = "google/paligemma-3b-pt-448"

processor = AutoProcessor.from_pretrained(
    model_id,
    use_fast=False
)

# Use AutoModelForImageTextToText instead of AutoModelForVision2Seq
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

# load first 100 rows from pixelprose dataset
dataset = load_dataset(
    "tomg-group-umd/pixelprose",
    split="train[:100]"
)

# image loader
def load_image(url):
    r = requests.get(url, timeout=10)
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img
