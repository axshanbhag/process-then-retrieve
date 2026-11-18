import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/paligemma-3b-pt-448"
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()

def clear_all_hooks(target_model):
    for module in target_model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()

clear_all_hooks(model)
dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True).take(100)

def load_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
    llm_layers = model.language_model.model.layers
    num_layers = model.language_model.config.num_hidden_layers
elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
    llm_layers = model.model.language_model.layers
    num_layers = model.config.text_config.num_hidden_layers
else:
    raise RuntimeError("Could not locate language model layers.")

layer_data = {i: {'pre': None, 'mid': None, 'post': None} for i in range(num_layers)}

def get_pre_hook(layer_idx):
    def hook(module, args):
        layer_data[layer_idx]['pre'] = args[0].detach()
    return hook

def get_mid_hook(layer_idx):
    def hook(module, args):
        layer_data[layer_idx]['mid'] = args[0].detach()
    return hook

def get_post_hook(layer_idx):
    def hook(module, args, output):
        layer_data[layer_idx]['post'] = output[0].detach() 
    return hook

for i, layer in enumerate(llm_layers):
    layer.register_forward_pre_hook(get_pre_hook(i))
    layer.post_attention_layernorm.register_forward_pre_hook(get_mid_hook(i))
    layer.register_forward_hook(get_post_hook(i))

metrics = {
    "vision": {"attn": np.zeros(num_layers), "mlp": np.zeros(num_layers), "total": np.zeros(num_layers)},
    "text":   {"attn": np.zeros(num_layers), "mlp": np.zeros(num_layers), "total": np.zeros(num_layers)}
}
valid_samples = 0
NUM_VISION_TOKENS = 1024 
PROMPT = "describe this image"

with torch.no_grad():
    for item in tqdm(dataset, total=100):
        image_url = item.get("url", "")
        if not image_url: continue
        
        image = load_image(image_url)
        if image is None: continue

        inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        for layer_idx in range(num_layers):
            pre = layer_data[layer_idx]['pre'].squeeze(0)   
            mid = layer_data[layer_idx]['mid'].squeeze(0)
            post = layer_data[layer_idx]['post'].squeeze(0)

            attn_update = torch.norm((mid - pre).float(), dim=-1).cpu().numpy()
            mlp_update  = torch.norm((post - mid).float(), dim=-1).cpu().numpy()
            total_update = torch.norm((post - pre).float(), dim=-1).cpu().numpy()

            metrics["vision"]["attn"][layer_idx] += np.mean(attn_update[:NUM_VISION_TOKENS])
            metrics["vision"]["mlp"][layer_idx]  += np.mean(mlp_update[:NUM_VISION_TOKENS])
            metrics["vision"]["total"][layer_idx] += np.mean(total_update[:NUM_VISION_TOKENS])

            metrics["text"]["attn"][layer_idx] += np.mean(attn_update[NUM_VISION_TOKENS:])
            metrics["text"]["mlp"][layer_idx]  += np.mean(mlp_update[NUM_VISION_TOKENS:])
            metrics["text"]["total"][layer_idx] += np.mean(total_update[NUM_VISION_TOKENS:])

        valid_samples += 1

print(f"\n Processed {valid_samples} valid images.")

for modality in ["vision", "text"]:
    for component in ["attn", "mlp", "total"]:
        metrics[modality][component] /= max(1, valid_samples)

relative_metrics = {
    "vision": {
        "attn": (metrics["vision"]["attn"] / (metrics["vision"]["attn"] + metrics["vision"]["mlp"])) * 100,
        "mlp":  (metrics["vision"]["mlp"] / (metrics["vision"]["attn"] + metrics["vision"]["mlp"])) * 100
    },
    "text": {
        "attn": (metrics["text"]["attn"] / (metrics["text"]["attn"] + metrics["text"]["mlp"])) * 100,
        "mlp":  (metrics["text"]["mlp"] / (metrics["text"]["attn"] + metrics["text"]["mlp"])) * 100
    }
}
print("Generating graphs...")

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'

layers = np.arange(num_layers)
x_ticks = np.arange(0, num_layers+1, 4)

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(layers, metrics["vision"]["total"], marker='o', markersize=4, color='#1b7837', linewidth=1.5, label='Vision Tokens')
ax1.plot(layers, metrics["text"]["total"], marker='o', markersize=4, color='#762a83', linewidth=1.5, label='Text Tokens')

ax1.set_title('Total Token Representation Changes Across Layers', fontsize=12, pad=10)
ax1.set_xlabel('Layer', fontsize=10)
ax1.set_ylabel(r'Avg Norm of Change ($\Delta$)', fontsize=10)
ax1.set_xticks(x_ticks)
ax1.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.show()
