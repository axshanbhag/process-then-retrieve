#vision representations patching
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests

def load_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None

        if "image" not in r.headers.get("Content-Type", ""):
            return None

        img = Image.open(BytesIO(r.content))
        img.verify()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img

    except (UnidentifiedImageError, OSError, requests.RequestException):
        return None

valid_urls = []

for url in dataset["url"]:
    img = load_image(url)
    if img is not None:
        valid_urls.append(url)

    if len(valid_urls) >= 100:
        break


import torch
import numpy as np

prompt = "<image> How many objects are in the image?"
answer = "2"

vision_layers = model.vision_tower.vision_model.encoder.layers

# patching mid to late layers (excluding the last layer)
SOURCE_LAYERS = [8, 9, 10]
# uncomment the following line for front patching early layers into late layers
# SOURCE_LAYERS = [2, 3, 4]
TARGET_LAYERS = [-4, -3, -2]

clean_activations = {}

def save_activation_hook(layer_id):
    def hook(module, inp, out):
        clean_activations[layer_id] = out.detach()
    return hook

def patch_activation_hook(src_layer):
    def hook(module, inp, out):
        patched = clean_activations[src_layer]
        assert out.shape == patched.shape
        return patched
    return hook

results = []
MAX_IMAGES = 100
processed = 0

for url in dataset["url"]:
    if processed >= MAX_IMAGES:
        break

    img = load_image(url)

    if img is None:
        continue

    processed += 1

    inputs = processor(
        images=img,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    labels = inputs["input_ids"].clone()
    answer_token_id = processor.tokenizer(
        answer, add_special_tokens=False
    ).input_ids[0]

    labels[:, :] = -100
    labels[:, -1] = answer_token_id

    # clean run
    clean_activations.clear()
    save_handles = []

    for src in SOURCE_LAYERS:
        h = vision_layers[src].register_forward_hook(
            save_activation_hook(src)
        )
        save_handles.append(h)

    with torch.no_grad():
        clean_outputs = model(**inputs, labels=labels)

    for h in save_handles:
        h.remove()

    clean_logits = clean_outputs.logits[:, -1, :]
    clean_score = clean_logits[0, answer_token_id].item()
    clean_loss = clean_outputs.loss.item()

    # patched run
    patch_handles = []

    for src, tgt in zip(SOURCE_LAYERS, TARGET_LAYERS):
        h = vision_layers[tgt].register_forward_hook(
            patch_activation_hook(src)
        )
        patch_handles.append(h)

    with torch.no_grad():
        patched_outputs = model(**inputs, labels=labels)

    for h in patch_handles:
        h.remove()

    patched_logits = patched_outputs.logits[:, -1, :]
    patched_score = patched_logits[0, answer_token_id].item()
    patched_loss = patched_outputs.loss.item()

    results.append({
        "clean_score": clean_score,
        "patched_score": patched_score,
        "delta": patched_score - clean_score,
        "clean_loss": clean_loss,
        "patched_loss": patched_loss
    })

assert len(results) > 0, "No valid images processed"

avg_clean_score = np.mean([r["clean_score"] for r in results])
avg_patched_score = np.mean([r["patched_score"] for r in results])
avg_delta = np.mean([r["delta"] for r in results])
avg_clean_loss = np.mean([r["clean_loss"] for r in results])
avg_patched_loss = np.mean([r["patched_loss"] for r in results])

print("Averaged Front-Patching Reuslts: ")
print(f"Images processed:     {len(results)}")
print(f"Avg Clean score:      {avg_clean_score:.4f}")
print(f"Avg Patched score:    {avg_patched_score:.4f}")
print(f"Avg Δ logit:          {avg_delta:.4f}")
print(f"Avg Clean loss:       {avg_clean_loss:.4f}")
print(f"Avg Patched loss:     {avg_patched_loss:.4f}")
