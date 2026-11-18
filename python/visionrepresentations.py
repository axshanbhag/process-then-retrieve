def read_image(url, target_size=(448, 448)):
    try:
        r = requests.get(url, timeout=8)
        if "image" not in r.headers.get("Content-Type", ""):
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img.resize(target_size)
    except:
        return None

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()

accum = None
count = 0

for row in tqdm(dataset, desc="Processing Images"):
    url = row["url"]
    caption = row["original_caption"]

    img = read_image(url)
    if img is None:
        continue

    inputs = processor(images=img, text="<image> " + caption, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    hidden = [h[0] for h in out.hidden_states]
    num_layers = len(hidden) - 1

    if accum is None:
        accum = np.zeros(num_layers - 1)

    enc_ids = inputs["input_ids"][0]
    img_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    # Isolates ONLY the 1024 <image> tokens
    vision_mask = (enc_ids == img_token_id).nonzero(as_tuple=True)[0]

    if len(vision_mask) == 0:
        continue

    layer_sims = []

    for i in range(1, num_layers):
        h_current = hidden[i][vision_mask]
        h_next = hidden[i+1][vision_mask]
        layer_sims.append(cosine(h_current, h_next))

    accum += np.array(layer_sims)
    count += 1

if count > 0:
    means = accum / count

    plt.figure(figsize=(10, 5))

    x_values = range(1, num_layers)
    x_labels = [f"{i}->{i+1}" for i in x_values]

    plt.plot(x_values, means, marker="o", markersize=7, linewidth=2.5, color="#2c629e")
    plt.grid(True)
    plt.ylim(0.75, 1.0)

    plt.xticks(x_values, x_labels, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Layer Pair", fontsize=14, labelpad=10)
    plt.ylabel("Mean Cosine Similarity", fontsize=14, labelpad=10)
    plt.title("Pure Visual Token Representation Shift (PaliGemma-3B)", fontsize=14)

    plt.tight_layout()
    plt.show()
else:
    print("No images were successfully processed. Check network connection.")
