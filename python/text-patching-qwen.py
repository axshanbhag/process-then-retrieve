import torch
import numpy as np
import requests
import gc
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# setup and load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)

model.eval()

dataset = load_dataset("tomg-group-umd/pixelprose", split="train[:100]")

llm_layers = None

#different layer paths
layer_paths = [
    lambda m: m.model.layers,           # standard Qwen2-VL
    lambda m: m.language_model.layers,  # split architecture
    lambda m: m.layers,                 # flat architecture
    lambda m: m.language_model.model.layers # nested split
]

# example patching configuration
# source layers are late layers
SOURCE_LAYERS = [10, 11, 12]
# final three layers
TARGET_LAYERS = [-4, -3, -2]
answer = "2"
prompt_text = "How many objects are in the image?"

def load_image(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return None

# front hook
clean_text_activations = {}

def save_text_hook(layer_id):
    def hook(module, inp, out):
        # extract hidden states
        if isinstance(out, tuple):
            state = out[0]
        elif hasattr(out, "last_hidden_state"):
            state = out.last_hidden_state
        else:
            state = out
            
        # save to cpu
        clean_text_activations[layer_id] = state.detach().cpu()
        return out
    return hook

def patch_text_hook(src_layer):
    def hook(module, inp, out):
        if src_layer not in clean_text_activations:
            return out
            
        # identify reference device
        if isinstance(out, tuple):
            ref_tensor = out[0]
        elif hasattr(out, "last_hidden_state"):
            ref_tensor = out.last_hidden_state
        else:
            ref_tensor = out
            
        # move patched state to correct device
        patched_state = clean_text_activations[src_layer].to(ref_tensor.device, dtype=ref_tensor.dtype)

        # must have the same number of tokens
        if patched_state.shape != ref_tensor.shape:
             return out

        # reconstruct output
        if isinstance(out, tuple):
            return (patched_state,) + out[1:]
        elif hasattr(out, "to_tuple"):
            return (patched_state,) + out.to_tuple()[1:]
        else:
            return patched_state
    return hook

# execute
results = []
MAX_IMAGES = 100
processed = 0

print("Starting Patching Loop...")

for url in dataset["url"]:
    if processed >= MAX_IMAGES: break
    
    img = load_image(url)
    if img is None: continue
    
    processed += 1
    if processed % 10 == 0: print(f"Processing image {processed}...")
    messages = [{"role": "user", "content": [{"type": "image", "image": url}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(device)

    # label setup
    answer_token_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids
    answer_token_id = answer_token_ids[0]
    labels = inputs["input_ids"].clone()
    labels[:, :] = -100
    labels[:, -1] = answer_token_id

    # clean run
    clean_text_activations.clear()
    save_handles = [llm_layers[src].register_forward_hook(save_text_hook(src)) for src in SOURCE_LAYERS]
    
    try:
        with torch.no_grad():
            clean_outputs = model(**inputs, labels=labels)
        
        clean_logits = clean_outputs.logits[:, -1, :]
        clean_score = clean_logits[0, answer_token_id].item()
        clean_loss = clean_outputs.loss.item()
    except Exception as e:
        print(f"Clean run error: {e}")
        for h in save_handles: h.remove()
        continue
        
    for h in save_handles: h.remove()

    # patched run
    patch_handles = [llm_layers[tgt].register_forward_hook(patch_text_hook(src)) 
                     for src, tgt in zip(SOURCE_LAYERS, TARGET_LAYERS)]
    
    try:
        with torch.no_grad():
            patched_outputs = model(**inputs, labels=labels)
            
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
    except Exception as e:
         print(f"Patched run error: {e}")
    
    for h in patch_handles: h.remove()

# results
if len(results) > 0:
    avg_clean_score = np.mean([r["clean_score"] for r in results])
    avg_patched_score = np.mean([r["patched_score"] for r in results])
    avg_delta = np.mean([r["delta"] for r in results])
    avg_clean_loss = np.mean([r["clean_loss"] for r in results])
    avg_patched_loss = np.mean([r["patched_loss"] for r in results])

    print(f"\n--- Averaged Patching Results (Qwen2-VL Textual) ---")
    print(f"Images processed:     {len(results)}")
    print(f"Avg Clean score:      {avg_clean_score:.4f}")
    print(f"Avg Patched score:    {avg_patched_score:.4f}")
    print(f"Avg Δ logit:          {avg_delta:.4f}")
    print(f"Avg Clean loss:       {avg_clean_loss:.4f}")
    print(f"Avg Patched loss:     {avg_patched_loss:.4f}")
else:
    print("No images were successfully processed.")
