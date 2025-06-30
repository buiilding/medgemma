from transformers import pipeline
from PIL import Image
import requests
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

image_path = "/home/coder/vkist/medgemma/images/1.jpg"
# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
try:
    image = Image.open(image_path)
except Exception as e:
    print(f"Error loading image from {image_path}: {e}")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this knee ultrasound"},
            {"type": "image", "image": image},
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])

# maybe put another advanced model, medgemma is just for explaining the ultrasound
patient_input = "my knee hurt so bad, i dont know where it hurts"

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": output},
            {"type": "text", "text": "reply to the patient"},
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])