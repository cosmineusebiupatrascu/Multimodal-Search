from src.encoder import CLIPEncoder

if __name__ == "__main__":
    encoder = CLIPEncoder()

    vector_text = encoder.encode_text("A red sports car")
    print(f"Vector Text generat cu succes! Dimensiune: {len(vector_text)}")
    print(f"Primele 5 valori: {vector_text[:5]}")