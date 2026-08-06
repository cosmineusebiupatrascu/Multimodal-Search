from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Union, List

from src import config


class CLIPEncoder:
    def __init__(self, model_name: str = config.MODEL_NAME, device: str = config.DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

    def encode_text(self, text: Union[str, List[str]]) -> List[float]:
        if isinstance(text, str):
            text = [text]

        inputs = self.processor(text=text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True
                                ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features.pooler_output
            norm = torch.linalg.vector_norm(text_features, ord=2, dim=-1, keepdim=True)
            text_features = text_features / norm

        return text_features.cpu().numpy().tolist()[0] if len(text) == 1 else text_features.cpu().numpy().tolist()

    def encode_image(self, image_input: Union[str, Image.Image]) -> List[float]:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features.pooler_output

            norm = torch.linalg.vector_norm(image_features, ord=2, dim=-1, keepdim=True)
            image_features = image_features / norm

        return image_features.cpu().numpy().tolist()[0]