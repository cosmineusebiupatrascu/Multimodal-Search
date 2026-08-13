import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

# Creăm folderul dacă nu există
os.makedirs("clip_onnx", exist_ok=True)
model_id = "openai/clip-vit-base-patch32"

print("Se descarcă modelele PyTorch...")
processor = CLIPProcessor.from_pretrained(model_id)
text_model = CLIPTextModelWithProjection.from_pretrained(model_id)
vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id)

# Salvăm procesorul (pentru tokenizare)
processor.save_pretrained("clip_onnx")

print("Se exportă modelul de Text...")
dummy_text = processor(text=["test"], return_tensors="pt")
torch.onnx.export(
    text_model,
    (dummy_text["input_ids"], dummy_text["attention_mask"]),
    "clip_onnx/text_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["text_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"}
    }
)

print("Se exportă modelul de Imagine...")
dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
dummy_vision = processor(images=dummy_image, return_tensors="pt")
torch.onnx.export(
    vision_model,
    (dummy_vision["pixel_values"],),
    "clip_onnx/vision_model.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "image_embeds": {0: "batch_size"}
    }
)

print("Export complet! Verifică folderul clip_onnx/")