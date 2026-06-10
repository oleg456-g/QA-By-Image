import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        self.image_paths = list(set(sample["image_path"] for sample in samples))
        print(f"Уникальных изображений: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]


def main():
    json_path = "vqa_val2014_preprocessed.json"
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(json_path)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    print("Загрузка CLIP...")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.eval()
    image_to_patches = {}

    print("Извлечение эмбеддингов...")

    with torch.no_grad():
        for batch_paths in tqdm(dataloader):
            images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Ошибка {path}: {e}")

            if len(images) == 0:
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            vision_outputs = model.vision_model(**inputs)

            patch_features = model.visual_projection(vision_outputs.last_hidden_state)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            patch_features = patch_features.cpu()

            for path, feat in zip(valid_paths, patch_features):
                image_to_patches[path] = feat

    output_path = os.path.join(output_dir, "val_vqa_patches_dict.pt")

    torch.save(image_to_patches, output_path)

    print(f"\nСохранено: {output_path}")
    print(f"Количество изображений: {len(image_to_patches)}")


if __name__ == "__main__":
    main()
