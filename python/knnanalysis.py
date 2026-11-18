import torch
import torch.nn.functional as F
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/paligemma-3b-pt-448"

print("Loading model and processor...")
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

print("Loading dataset...")
dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True).take(100)

def load_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def extract_top_k_neighbors(embeddings, k=5):
    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)

    num_tokens, embed_dim = embeddings.shape

    target_idx = random.randint(0, num_tokens - 1)
    target_token = embeddings[target_idx]

    target_norm = F.normalize(target_token.unsqueeze(0), p=2, dim=1)
    all_tokens_norm = F.normalize(embeddings, p=2, dim=1)

    similarities = torch.mm(target_norm, all_tokens_norm.T).squeeze(0)

    top_k_values, top_k_indices = torch.topk(similarities, k + 1)

    neighbor_indices = top_k_indices[1:]
    similarity_scores = top_k_values[1:]

    return target_idx, neighbor_indices, similarity_scores

def visualize_knn_patches(image, anchor_idx, neighbor_indices, title, patch_size=14, grid_size=32):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    def get_patch_coords(idx):
        row = idx // grid_size
        col = idx % grid_size
        return col * patch_size, row * patch_size

    for n_idx in neighbor_indices:
        nx, ny = get_patch_coords(n_idx)
        rect = patches.Rectangle(
            (nx, ny), patch_size, patch_size,
            linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

    ax_x, ax_y = get_patch_coords(anchor_idx)
    rect = patches.Rectangle(
        (ax_x, ax_y), patch_size, patch_size,
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    plt.title(title)
    plt.axis('off')
    plt.show()

print("\nSearching for a valid image...")

for sample in dataset:
    url = sample.get('url')
    if not url: continue

    image = load_image(url)
    if image is None: continue

    print("Successfully loaded an image. Running forward pass...")

    visual_image = image.resize((448, 448))

    inputs = processor(images=image, text="<image>", return_tensors="pt")
    pixel_values = inputs.pixel_values.to(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None and hasattr(model, "model"):
        vision_tower = getattr(model.model, "vision_tower", None)

    with torch.no_grad():
        vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
        layer_0 = vision_outputs.hidden_states[0]
        layer_penultimate = vision_outputs.hidden_states[-2]

    print("\n--- Layer 0 Results (Spatial Purity) ---")
    k = 5
    t_idx_0, n_idx_0, scores_0 = extract_top_k_neighbors(layer_0, k=k)

    print(f"Anchor Index: {t_idx_0}")
    print(f"Neighbor Indices: {n_idx_0.tolist()}")
    print(f"Similarities: {[round(s.item(), 4) for s in scores_0]}")

    visualize_knn_patches(
        visual_image,
        t_idx_0,
        n_idx_0.cpu().tolist(),
        title=f"Layer 0 | Anchor: {t_idx_0} (Red) | Neighbors (Blue)"
    )

    print("\n--- Layer -2 Results (Semantic Info-Density) ---")
    t_idx_deep, n_idx_deep, scores_deep = extract_top_k_neighbors(layer_penultimate, k=k)

    print(f"Anchor Index: {t_idx_deep}")
    print(f"Neighbor Indices: {n_idx_deep.tolist()}")
    print(f"Similarities: {[round(s.item(), 4) for s in scores_deep]}")

    visualize_knn_patches(
        visual_image,
        t_idx_deep,
        n_idx_deep.cpu().tolist(),
        title=f"Layer -2 | Anchor: {t_idx_deep} (Red) | Neighbors (Blue)"
    )
    break
