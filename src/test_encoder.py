from src.encoder import CLIPEncoder
from PIL import Image

if __name__ == "__main__":
    encoder = CLIPEncoder()

    vector_text = encoder.encode_text("Red image")
    print(f"Vector Text generat. Dimensiune: {len(vector_text)}")
    print(f"Primele 5 valori: {vector_text[:5]}")

    test_image = Image.new("RGB", (224, 224), color="red")

    vector_image = encoder.encode_image(test_image)

    print("Input imagine: Obiect PIL (224x224 RGB)")
    print(f"Dimensiune vector: {len(vector_image)}")
    print(f"Primele 5 valori: {vector_image[:5]}\n")

    assert len(vector_text) == 512, "Dimensiunea vectorului de text este incorectă!"
    assert len(vector_image) == 512, "Dimensiunea vectorului de imagine este incorectă!"