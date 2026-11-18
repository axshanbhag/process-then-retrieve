import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset

# device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# load Qwen2-VL-2B-Instruct
model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

model.eval()

# Dataset loading
dataset = load_dataset("tomg-group-umd/pixelprose", split="train[:100]")
