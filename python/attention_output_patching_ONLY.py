import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/paligemma-3b-pt-448"

print(f"Loading processor and model on {device}...")
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
print("Model hooks scrubbed and reset.")

dataset = load_dataset("tomg-group-umd/pixelprose", split="train[:100]")

def load_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: 
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except Exception:
        return None

clean_activations = {}

def get_save_hook(layer_id, mode_str):
    def hook(module, inp, out):
        hidden = out[0].detach() if isinstance(out, tuple) else out.detach()
        clean_activations[f"{mode_str}_{layer_id}"] = hidden
    return hook

def get_layer_lists(target_model):
    v_layers, t_layers = None, None
    for name, module in target_model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            if "vision" in name.lower() and "layer" in name.lower() and v_layers is None:
                v_layers = module
            elif ("language" in name.lower() or "text" in name.lower() or name.endswith("model.layers")) and "layer" in name.lower() and "vision" not in name.lower() and t_layers is None:
                t_layers = module
    return v_layers, t_layers

print("Mapping internal layer structures...")
vision_layers, text_layers = get_layer_lists(model)

if vision_layers is None or text_layers is None:
    raise RuntimeError("Failed to dynamically locate the vision or text layers inside the model.")

print("Preparing dataset and configurations...")
MAX_IMAGES = 100
valid_urls = []
for url in dataset["url"]:
    if len(valid_urls) >= MAX_IMAGES: break
    img = load_image(url)
    if img is not None: valid_urls.append(url)

print(f"Verified {len(valid_urls)} images for processing.")
assert len(valid_urls) > 0, "No valid images found to process."

prompt = "<image> How many objects are in the image?"
answer_token_id = processor.tokenizer("2", add_special_tokens=False).input_ids[0]

PATCH_CONFIGS = {
    "early_to_mid":  {"source": [2, 3, 4],    "target": [5, 6, 7]},
    "late_to_mid":   {"source": [10, 11, 12], "target": [5, 6, 7]},
    "early_to_late": {"source": [2, 3, 4],    "target": [10, 11, 12]},
    "mid_to_late":   {"source": [5, 6, 7],    "target": [10, 11, 12]},
    "late_to_early": {"source": [10, 11, 12], "target": [2, 3, 4]}
}

all_results_summary = []

for USE_VISION in [True, False]:
    layers = vision_layers if USE_VISION else text_layers
    mode_name = "VISION" if USE_VISION else "TEXT"

    for config_name, config in PATCH_CONFIGS.items():
        print(f"Running ATTENTION Patching: {mode_name} | {config_name}...")
        
        run_data = []

        for url in valid_urls:
            img = load_image(url)
            if img is None: continue

            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            
            labels = inputs["input_ids"].clone()
            labels[:] = -100
            labels[:, -1] = answer_token_id
            inputs["labels"] = labels 

            clean_activations.clear()
            
            h_save = [layers[src].self_attn.register_forward_hook(get_save_hook(src, mode_name)) for src in config["source"]]
            
            try:
                with torch.no_grad():
                    out_clean = model(**inputs)
            finally:
                for h in h_save: h.remove()
            
            clean_score = out_clean.logits[0, -1, answer_token_id].item()

            def get_patch_hook(s_idx, mode_str):
                def hook(module, inp, out):
                    key = f"{mode_str}_{s_idx}"
                    if key not in clean_activations:
                        return out 
                    
                    patched = clean_activations[key]
                    
                    if isinstance(out, tuple):
                        if out[0].shape[-1] != patched.shape[-1]:
                            return out
                        return (patched,) + out[1:]
                    else:
                        if out.shape[-1] != patched.shape[-1]:
                            return out
                        return patched
                return hook

            h_patch = [layers[tgt].self_attn.register_forward_hook(get_patch_hook(src, mode_name)) for src, tgt in zip(config["source"], config["target"])]
            
            try:
                with torch.no_grad():
                    out_patched = model(**inputs)
            finally:
                for h in h_patch: h.remove()
            
            patched_score = out_patched.logits[0, -1, answer_token_id].item()

            run_data.append({
                "clean_score": clean_score,
                "patched_score": patched_score,
                "delta": patched_score - clean_score,
                "clean_loss": out_clean.loss.item(),
                "patched_loss": out_patched.loss.item()
            })

        all_results_summary.append({
            "mode": mode_name,
            "config": config_name,
            "avg_clean_score": np.mean([r["clean_score"] for r in run_data]),
            "avg_patched_score": np.mean([r["patched_score"] for r in run_data]),
            "avg_delta": np.mean([r["delta"] for r in run_data]),
            "avg_clean_loss": np.mean([r["clean_loss"] for r in run_data]),
            "avg_patched_loss": np.mean([r["patched_loss"] for r in run_data])
        })

print("\n" + "="*110)
print(f"{'Tower Type':<12} | {'Config Name':<15} | {'Clean Score':<12} | {'Patch Score':<12} | {'Avg Δ Logit':<12} | {'Clean Loss':<12} | {'Patch Loss':<12}")
print("-" * 110)
for res in all_results_summary:
    print(f"{res['mode']:<12} | {res['config']:<15} | {res['avg_clean_score']:<12.4f} | {res['avg_patched_score']:<12.4f} | {res['avg_delta']:<12.4f} | {res['avg_clean_loss']:<12.4f} | {res['avg_patched_loss']:<12.4f}")
print("="*110)
