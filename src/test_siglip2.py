from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/LeBron_James_%2851959977144%29_%28cropped2%29.jpg/800px-LeBron_James_%2851959977144%29_%28cropped2%29.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["2 cats", "2 dogs"]
texts = [f"This is a photo of {label}." for label in candidate_labels]

inputs = processor(text=texts, images=image, max_num_patches=256, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# get embedding
image_embeds = outputs.vision_model_output.pooler_output
print(image_embeds) 

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")