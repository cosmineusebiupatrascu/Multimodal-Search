import os
import urllib.request
import tarfile
import shutil
from pathlib import Path

# Arhiva la rezoluție originală (~1.5 GB)
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
ARCHIVE_NAME = "imagenette2.tgz"
EXTRACT_DIR = "imagenette_temp"
OUTPUT_DIR = "data/raw_images"

CLASS_MAPPING = {
    "n01440764": "fish",
    "n02102040": "dog",
    "n02979186": "cassette_player",
    "n03028079": "church",
    "n03394916": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
    "n03930630": "french_horn",
    "n04330267": "chain_saw"
}


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading archive: {percent}%", end="")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ARCHIVE_NAME):
        urllib.request.urlretrieve(URL, ARCHIVE_NAME, reporthook=download_progress)
        print("\nFinished downloading")
    else:
        print("Archive found locally")

    print("Unpacking")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)

    train_dir = Path(EXTRACT_DIR) / "imagenette2" / "train"

    image_count = 0
    IMAGES_PER_CLASS = 20

    for wn_id, class_name in CLASS_MAPPING.items():
        class_dir = train_dir / wn_id
        if not class_dir.exists():
            continue

        images = list(class_dir.glob("*.JPEG"))
        if IMAGES_PER_CLASS:
            images = images[:IMAGES_PER_CLASS]

        for i, img_path in enumerate(images):
            new_name = f"{class_name}_{i}.jpg"
            dest_path = Path(OUTPUT_DIR) / new_name
            shutil.copy2(img_path, dest_path)
            image_count += 1

    print("Deleting temp files")
    shutil.rmtree(EXTRACT_DIR)
    os.remove(ARCHIVE_NAME)

    print(f"\n{image_count} images added to {OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()