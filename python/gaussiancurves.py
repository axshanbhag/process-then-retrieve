import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None

def get_decoder_layers(vlm_model):
    for name, module in vlm_model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            layer_name = module[0].__class__.__name__.lower()
            if "vision" not in name.lower() and ("decoder" in layer_name or "layer" in layer_name):
                return module
    raise AttributeError("Could not locate decoder layers.")

decoder_layers = get_decoder_layers(model)
activation_cache = {}

def get_cache_hook(layer_idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activation_cache[layer_idx] = hidden.detach()
    return hook

def get_patch_hook(layer_idx, vision_mask):
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        cached = activation_cache[layer_idx]
        if hidden.dim() == 3:
            hidden[vision_mask] = cached[vision_mask]
        return (hidden,) + output[1:] if is_tuple else hidden
    return hook

layer_groups = {
    "Layers 0-8": list(range(0, 9)),
    "Layers 9-17": list(range(9, 18))
}

results = {
    "Layers 0-8": [],
    "Layers 9-17": [],
    "Natural Baseline": []
}
num_pairs = len(dataset) // 2

for i in tqdm(range(num_pairs), desc="Running Grouped Causal Trace"):
    idx_a, idx_b = i * 2, i * 2 + 1
    img_a, img_b = fetch_image(dataset[idx_a]["url"]), fetch_image(dataset[idx_b]["url"])
    if img_a is None or img_b is None: continue

    caption_a = dataset[idx_a]["original_caption"]
    prompt = f"<image>question Is this a picture of {caption_a}?\n"
    inputs_a = processor(text=prompt, images=img_a, return_tensors="pt").to(device)
    inputs_b = processor(text=prompt, images=img_b, return_tensors="pt").to(device)
    vision_mask = (inputs_a["input_ids"] == image_token_id)

    with torch.no_grad():
        logits_a = model(**inputs_a).logits[0, -1, :]
        probs_a = F.softmax(logits_a, dim=-1)

    with torch.no_grad():
        logits_b = model(**inputs_b).logits[0, -1, :]
        log_probs_b = F.log_softmax(logits_b, dim=-1)
    kl_natural = F.kl_div(log_probs_b, probs_a, reduction="sum").item()
    results["Natural Baseline"].append(kl_natural)

    for group_name, layers in layer_groups.items():
        activation_cache.clear()
        handles = []
        for layer_idx in layers:
            handles.append(decoder_layers[layer_idx].register_forward_hook(get_cache_hook(layer_idx)))
        with torch.no_grad(): _ = model(**inputs_b)
        for h in handles: h.remove()
        handles = []
        for layer_idx in layers:
            handles.append(decoder_layers[layer_idx].register_forward_hook(get_patch_hook(layer_idx, vision_mask)))
        with torch.no_grad():
            logits_patch = model(**inputs_a).logits[0, -1, :]
            log_probs_patch = F.log_softmax(logits_patch, dim=-1)
        for h in handles: h.remove()

        kl_intervention = F.kl_div(log_probs_patch, probs_a, reduction="sum").item()
        results[group_name].append(kl_intervention)

plt.figure(figsize=(10, 6))

colors = {
    "Layers 0-8": "blue",
    "Layers 9-17": "red",
    "Natural Baseline": "green"
}

line_styles = {
    "Layers 0-8": "-",
    "Layers 9-17": "-",
    "Natural Baseline": "--"
}
for group, data in results.items():
    if not data:
        continue
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=2, linestyle=line_styles[group],
             label=f"{group} ($\mu={mu:.2f}, \sigma={std:.2f}$)", color=colors[group])
    plt.fill_between(x, p, alpha=0.15, color=colors[group])

plt.title("Output Distribution Shift: Intervention vs. Natural Difference")
plt.xlabel("KL Divergence $D_{KL}$(Clean A || Intervention / Clean B)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
