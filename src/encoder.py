import onnxruntime as ort
from transformers import CLIPProcessor
from PIL import Image


class CLIPEncoder:
    def __init__(self, model_path="clip_onnx"):
        self.processor = CLIPProcessor.from_pretrained(model_path)

        self.text_session = ort.InferenceSession(f"{model_path}/text_model.onnx")
        self.vision_session = ort.InferenceSession(f"{model_path}/vision_model.onnx")

    def encode_text(self, text: str) -> list[float]:
        inputs = self.processor(text=text, return_tensors="np", padding=True)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        outputs = self.text_session.run(None, onnx_inputs)

        return outputs[0][0].tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="np")

        onnx_inputs = {
            "pixel_values": inputs["pixel_values"]
        }

        outputs = self.vision_session.run(None, onnx_inputs)
        return outputs[0][0].tolist()