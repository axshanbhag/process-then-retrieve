import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model_id = "google/paligemma-3b-pt-448"

processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
).eval()

vision_tower = getattr(model, "vision_tower", None)
if vision_tower is None and hasattr(model, "model"):
    vision_tower = getattr(model.model, "vision_tower", None)

projector = getattr(model, "multi_modal_projector", None)
if projector is None and hasattr(model, "model"):
    projector = getattr(model.model, "multi_modal_projector", None)

GRID = 32
PATCH_PX = 14
K = 5

# SET CUSTOM IMAGE
# IMAGE_PATH = "holiday-buffet.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
image_448 = image.resize((448, 448))


# Show grid overlay to pick img patches

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_448)

# Draw full grid
for r in range(GRID + 1):
    ax.axhline(r * PATCH_PX, color="white", linewidth=0.3, alpha=0.5)
for c in range(GRID + 1):
    ax.axvline(c * PATCH_PX, color="white", linewidth=0.3, alpha=0.5)

# Label every 2nd row/col
for r in range(0, GRID, 2):
    for c in range(0, GRID, 2):
        idx = r * GRID + c
        ax.text(
            c * PATCH_PX + PATCH_PX // 2,
            r * PATCH_PX + PATCH_PX // 2,
            f"{r},{c}", fontsize=4, ha="center", va="center",
            color="yellow", fontweight="bold",
        )

ax.set_title("Grid overlay", fontsize=12)
ax.axis("off")
plt.tight_layout()
plt.show()

print("Find the right dog's ear in the grid above.")
print("Update ANCHOR_2 below if needed: row * 32 + col")


# Set anchors, update after checking the grid

ANCHOR_1 = 10 * GRID + 16   # row=10, col=11
LABEL_1 = "Person"

ANCHOR_2 = 22 * GRID + 16   # row=12, col=20
LABEL_2 = "Food"

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_448)

for anchor, label, color in [
    (ANCHOR_1, LABEL_1, "red"),
    (ANCHOR_2, LABEL_2, "cyan"),
]:
    r, c = divmod(anchor, GRID)
    rect = mpatches.Rectangle(
        (c * PATCH_PX, r * PATCH_PX), PATCH_PX, PATCH_PX,
        linewidth=3, edgecolor=color, facecolor=color, alpha=0.5,
    )
    ax.add_patch(rect)
    ax.text(
        c * PATCH_PX + PATCH_PX // 2,
        r * PATCH_PX - 4,
        f"{label}\n(row={r}, col={c})",
        fontsize=8, ha="center", color=color, fontweight="bold",
        bbox=dict(boxstyle="round", fc="black", alpha=0.6),
    )

ax.set_title("Check anchors", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.show()


# Extract post-projection tokens
inputs = processor(images=image, text="<image>", return_tensors="pt")
pixel_values = inputs.pixel_values.to(device, dtype=dtype)

with torch.no_grad():
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    siglip_L0 = vision_outputs.hidden_states[0].squeeze(0).float().cpu()
    post_proj = projector(vision_outputs.last_hidden_state).squeeze(0).float().cpu()

print(f"SigLIP L0:       {siglip_L0.shape}")
print(f"Post-projection: {post_proj.shape}")


# Defs
def patch_knn(embeddings, anchor_idx, k=K):
    anchor = F.normalize(embeddings[anchor_idx].unsqueeze(0), dim=-1)
    all_n = F.normalize(embeddings, dim=-1)
    sims = (anchor @ all_n.T).squeeze(0)
    vals, ids = sims.topk(k + 1)
    mask = ids != anchor_idx
    return ids[mask][:k].cpu().tolist(), vals[mask][:k].cpu().tolist()


def cosine_sim(embeddings, idx_a, idx_b):
    a = F.normalize(embeddings[idx_a].unsqueeze(0), dim=-1)
    b = F.normalize(embeddings[idx_b].unsqueeze(0), dim=-1)
    return (a @ b.T).item()


def draw_knn(ax, img, anchor, n_ids, n_sims, title, anchor_color="red", neighbor_color="blue"):
    ax.imshow(img)
    for n_idx, sim in zip(n_ids, n_sims):
        r, c = divmod(n_idx, GRID)
        rect = mpatches.Rectangle(
            (c * PATCH_PX, r * PATCH_PX), PATCH_PX, PATCH_PX,
            linewidth=2, edgecolor=neighbor_color, facecolor=neighbor_color,
            alpha=max(0.2, min(0.6, sim)),
        )
        ax.add_patch(rect)
    ar, ac = divmod(anchor, GRID)
    rect = mpatches.Rectangle(
        (ac * PATCH_PX, ar * PATCH_PX), PATCH_PX, PATCH_PX,
        linewidth=3, edgecolor=anchor_color, facecolor=anchor_color, alpha=0.6,
    )
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Post-Projection vs SigLIP L0 (single anchor)

n_base, s_base = patch_knn(siglip_L0, ANCHOR_1, k=K)
n_proj, s_proj = patch_knn(post_proj, ANCHOR_1, k=K)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
draw_knn(axes[0], image_448, ANCHOR_1, n_base, s_base,
         "SigLIP Layer 0 (baseline, 1152-d)")
draw_knn(axes[1], image_448, ANCHOR_1, n_proj, s_proj,
         "Post-Projection Layer 0 (2048-d)")
fig.suptitle(f"Section 4 | Anchor: {LABEL_1} (patch {ANCHOR_1})", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

print(f"Anchor: {ANCHOR_1} ({LABEL_1})")
print(f"SigLIP L0 neighbors:      {n_base}  sims: {[f'{s:.4f}' for s in s_base]}")
print(f"Post-projection neighbors: {n_proj}  sims: {[f'{s:.4f}' for s in s_proj]}")


# Both Anchors

n1_ids, n1_sims = patch_knn(post_proj, ANCHOR_1, k=K)
n2_ids, n2_sims = patch_knn(post_proj, ANCHOR_2, k=K)
cross_sim = cosine_sim(post_proj, ANCHOR_1, ANCHOR_2)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

draw_knn(axes[0], image_448, ANCHOR_1, n1_ids, n1_sims,
         f"{LABEL_1} (patch {ANCHOR_1})", anchor_color="red", neighbor_color="orange")

draw_knn(axes[1], image_448, ANCHOR_2, n2_ids, n2_sims,
         f"{LABEL_2} (patch {ANCHOR_2})", anchor_color="cyan", neighbor_color="blue")

# Both overlaid
axes[2].imshow(image_448)
for n_idx, sim in zip(n1_ids, n1_sims):
    r, c = divmod(n_idx, GRID)
    rect = mpatches.Rectangle(
        (c * PATCH_PX, r * PATCH_PX), PATCH_PX, PATCH_PX,
        linewidth=2, edgecolor="orange", facecolor="orange",
        alpha=max(0.2, min(0.6, sim)),
    )
    axes[2].add_patch(rect)
ar, ac = divmod(ANCHOR_1, GRID)
axes[2].add_patch(mpatches.Rectangle(
    (ac * PATCH_PX, ar * PATCH_PX), PATCH_PX, PATCH_PX,
    linewidth=3, edgecolor="red", facecolor="red", alpha=0.6,
))

for n_idx, sim in zip(n2_ids, n2_sims):
    r, c = divmod(n_idx, GRID)
    rect = mpatches.Rectangle(
        (c * PATCH_PX, r * PATCH_PX), PATCH_PX, PATCH_PX,
        linewidth=2, edgecolor="blue", facecolor="blue",
        alpha=max(0.2, min(0.6, sim)),
    )
    axes[2].add_patch(rect)
ar, ac = divmod(ANCHOR_2, GRID)
axes[2].add_patch(mpatches.Rectangle(
    (ac * PATCH_PX, ar * PATCH_PX), PATCH_PX, PATCH_PX,
    linewidth=3, edgecolor="cyan", facecolor="cyan", alpha=0.6,
))

legend = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="red",    markersize=10, label=LABEL_1),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="orange", markersize=10, label=f"{LABEL_1} neighbors"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",   markersize=10, label=LABEL_2),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="blue",   markersize=10, label=f"{LABEL_2} neighbors"),
]
axes[2].legend(handles=legend, loc="lower right", fontsize=8)
axes[2].set_title("Both anchors overlaid", fontsize=10)
axes[2].axis("off")

fig.suptitle(
    f"Post-Projection Layer 0 | Cross-anchor cosine sim: {cross_sim:.4f}",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.show()


# Cos Sim analysis
a1_to_n2 = [cosine_sim(post_proj, ANCHOR_1, n) for n in n2_ids]
a2_to_n1 = [cosine_sim(post_proj, ANCHOR_2, n) for n in n1_ids]

print(f"\n")
print("CosSim Analysis (Post-Projection Layer 0)")

print(f"\n{LABEL_1} (patch {ANCHOR_1}):")
print(f"  Neighbors: {n1_ids}")
print(f"  Sims:      {[f'{s:.4f}' for s in n1_sims]}")
print(f"  Mean:      {np.mean(n1_sims):.4f}")

print(f"\n{LABEL_2} (patch {ANCHOR_2}):")
print(f"  Neighbors: {n2_ids}")
print(f"  Sims:      {[f'{s:.4f}' for s in n2_sims]}")
print(f"  Mean:      {np.mean(n2_sims):.4f}")

print(f"\nBetween anchors ({LABEL_1} ↔ {LABEL_2}): {cross_sim:.4f}")

print(f"\n{LABEL_1} compared to {LABEL_2}'s neighbors: {[f'{s:.4f}' for s in a1_to_n2]}")
print(f"  Mean: {np.mean(a1_to_n2):.4f}")
print(f"{LABEL_2} compared to {LABEL_1}'s neighbors: {[f'{s:.4f}' for s in a2_to_n1]}")
print(f"  Mean: {np.mean(a2_to_n1):.4f}")

overlap = set(n1_ids) & set(n2_ids)

print(f"\n")
print(f"  {LABEL_1} self-neighborhood sim:  {np.mean(n1_sims):.4f}")
print(f"  {LABEL_2} self-neighborhood sim:  {np.mean(n2_sims):.4f}")
print(f"  Between anchors sim:                  {cross_sim:.4f}")
print(f"  {LABEL_1} compared to {LABEL_2}'s neighbors: {np.mean(a1_to_n2):.4f}")
print(f"  {LABEL_2} compared to {LABEL_1}'s neighbors: {np.mean(a2_to_n1):.4f}")
