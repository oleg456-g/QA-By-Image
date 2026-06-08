import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class PreprocessedVQADataset(Dataset):
    def __init__(self, preprocessed_json_path):
        with open(preprocessed_json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
        self.unique_images = list(set(sample['image_path'] for sample in self.samples))
        print(f"Загружено записей: {len(self.samples)}. Уникальных картинок: {len(self.unique_images)}")

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        return self.unique_images[idx]

def main():
    # Настройка путей
    json_path = "vqa_val2014_preprocessed.json"
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    # Инициализируем датасет уникальных картинок
    dataset = PreprocessedVQADataset(json_path)
    dataloader = DataLoader(
            dataset, 
            batch_size=8,
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
    
    # Загружаем CLIP
    print("Загрузка CLIP-ViT-B/32...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Хэш-мапа для хранения векторов: {image_path: tensor}
    image_to_embedding = {}
    
    print("Начало извлечения эмбеддингов картинок...")
    with torch.no_grad():
        for batch_paths in tqdm(dataloader):
            valid_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                try:
                    # Проверяем наличие файла на диске
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        valid_images.append(img)
                        valid_paths.append(img_path)
                    else:
                        print(f"\nФайл не найден: {img_path}")
                except Exception as e:
                    print(f"\nОшибка чтения {img_path}: {e}")
                    continue
                    
            if not valid_images:
                continue
                
            # Прогоняем батч через CLIP
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)

            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            # Нормализация
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu() # переносим на CPU для экономии VRAM
            
            # Сохраняем в память
            for path, feat in zip(valid_paths, image_features):
                image_to_embedding[path] = feat

    # Теперь сопоставляем готовые векторы с вопросами
    print("\nСвязывание эмбеддингов с вопросами и формирование финальных матриц...")
    final_embeddings = []
    final_metadata = []
    
    for sample in dataset.samples:
        path = sample['image_path']
        if path in image_to_embedding:
            final_embeddings.append(image_to_embedding[path])
            final_metadata.append({
                'question': sample['question'],
                'answer': sample['answer'],
                'answer_type': sample['answer_type']
            })
            
    # Конкатенируем в один большой тензор [N, 512]
    final_embeddings_tensor = torch.stack(final_embeddings)
    
    # Сохраняем файлы
    emb_out = os.path.join(output_dir, "val_vqa_embeddings.pt")
    meta_out = os.path.join(output_dir, "val_vqa_metadata.json")
    
    torch.save(final_embeddings_tensor, emb_out)
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)
        
    print(f"\nУспешно сохранено!")
    print(f"Матрица эмбеддингов: {emb_out} (Размер: {final_embeddings_tensor.shape})")
    print(f"Метаданные вопросов: {meta_out}")

if __name__ == "__main__":
    main()
