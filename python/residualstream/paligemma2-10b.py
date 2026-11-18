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
model_id = "google/paligemma2-10b-pt-448"

if device == "cuda" and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

print(f"Loading processor and model on {device} with {compute_dtype}...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=compute_dtype,
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

def load_image(url, timeout=10, max_size=512):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None
llm_layers = None
for name, module in model.named_modules():
    if name.endswith('model.layers') and isinstance(module, torch.nn.ModuleList):
        llm_layers = module
        break

if llm_layers is None:
    try:
        llm_layers = model.language_model.model.layers
    except AttributeError:
        raise RuntimeError("Could not locate language model layers.")

num_layers = len(llm_layers)
layer_data = {i: {'pre': None, 'mid': None, 'post': None} for i in range(num_layers)}

def get_pre_hook(layer_idx):
    def hook(module, args):
        layer_data[layer_idx]['pre'] = args[0].detach().clone()
    return hook

def get_mid_hook(layer_idx):
    def hook(module, args):
        layer_data[layer_idx]['mid'] = args[0].detach().clone()
    return hook

def get_post_hook(layer_idx):
    def hook(module, args, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        layer_data[layer_idx]['post'] = out_tensor.detach().clone()
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
PROMPT = "describe en"

with torch.no_grad():
    for item in tqdm(dataset, total=100):
        image_url = item.get("url", "")
        if not image_url: continue
        
        image = load_image(image_url)
        if image is None: continue

        inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        is_nan_corrupted = False
        for layer_idx in range(num_layers):
            if torch.isnan(layer_data[layer_idx]['post']).any():
                is_nan_corrupted = True
                break
        
        if is_nan_corrupted:
            del inputs, outputs
            torch.cuda.empty_cache()
            continue

        for layer_idx in range(num_layers):
            pre = layer_data[layer_idx]['pre'].squeeze(0)   
            mid = layer_data[layer_idx]['mid'].squeeze(0)
            post = layer_data[layer_idx]['post'].squeeze(0)

            attn_update = torch.norm((mid.float() - pre.float()), dim=-1).cpu().numpy()
            mlp_update  = torch.norm((post.float() - mid.float()), dim=-1).cpu().numpy()
            total_update = torch.norm((post.float() - pre.float()), dim=-1).cpu().numpy()

            metrics["vision"]["attn"][layer_idx] += np.mean(attn_update[:NUM_VISION_TOKENS])
            metrics["vision"]["mlp"][layer_idx]  += np.mean(mlp_update[:NUM_VISION_TOKENS])
            metrics["vision"]["total"][layer_idx] += np.mean(total_update[:NUM_VISION_TOKENS])

            metrics["text"]["attn"][layer_idx] += np.mean(attn_update[NUM_VISION_TOKENS:])
            metrics["text"]["mlp"][layer_idx]  += np.mean(mlp_update[NUM_VISION_TOKENS:])
            metrics["text"]["total"][layer_idx] += np.mean(total_update[NUM_VISION_TOKENS:])

        valid_samples += 1

        del inputs, outputs
        torch.cuda.empty_cache()


for component in ["attn", "mlp", "total"]:
    metrics["vision"][component] /= max(1, valid_samples)
    metrics["text"][component] /= max(1, valid_samples)

relative_metrics = {"vision": {}, "text": {}}

for modality in ["vision", "text"]:
    total_sub = metrics[modality]["attn"] + metrics[modality]["mlp"]
    total_sub[total_sub == 0] = 1e-9
    
    relative_metrics[modality]["attn"] = (metrics[modality]["attn"] / total_sub) * 100
    relative_metrics[modality]["mlp"] = (metrics[modality]["mlp"] / total_sub) * 100

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'

layers = np.arange(num_layers)
x_ticks = np.arange(0, num_layers+1, 6)
fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(layers, metrics["vision"]["total"], marker='o', markersize=4, color='#1b7837', linewidth=1.5, label='Vision Tokens')
ax1.plot(layers, metrics["text"]["total"], marker='o', markersize=4, color='#762a83', linewidth=1.5, label='Text Tokens')

ax1.set_title('PaliGemma2-10B Total Token Representation Changes Across Layers', fontsize=12, pad=10)
ax1.set_xlabel('Layer', fontsize=10)
ax1.set_ylabel(r'Avg Norm of Change ($\Delta$)', fontsize=10)
ax1.set_xticks(x_ticks)
ax1.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.show()
