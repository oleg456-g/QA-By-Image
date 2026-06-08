import os
import json
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx = 2

    def build_vocab(self, texts, max_size=10000):
        words = []
        for text in texts:
            words.extend(clean_text(text).split())
        most_common = Counter(words).most_common(max_size)
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def tokenize(self, text, max_len=15):
        tokens = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in clean_text(text).split()]
        if len(tokens) < max_len:
            tokens = tokens + [self.word2idx['<PAD>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.idx

class VQAInMemoryDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, max_answers=1000):
        print("Загрузка эмбеддингов картинок из .pt файла...")
        self.embeddings = torch.load(embeddings_path)

        print("Загрузка метаданных вопросов из .json...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        assert len(self.embeddings) == len(self.metadata), "Размеры эмбеддингов и метаданных не совпадают!"

        print("Построение словаря вопросов...")
        questions = [item['question'] for item in self.metadata]
        self.vocab_q = Vocabulary()
        self.vocab_q.build_vocab(questions)

        print("Построение словаря ответов (выделяем классы классификации)...")
        answers = [item['answer'] for item in self.metadata]
        most_common_answers = Counter(answers).most_common(max_answers)

        self.ans2idx = {}
        self.idx2ans = {}
        for idx, (ans, _) in enumerate(most_common_answers):
            self.ans2idx[ans] = idx
            self.idx2ans[idx] = ans

        print(f"-> Размер словаря вопросов: {len(self.vocab_q)}")
        print(f"-> Количество классов-ответов (выходной слой): {len(self.ans2idx)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_emb = self.embeddings[idx]
        meta = self.metadata[idx]

        q_tokens = self.vocab_q.tokenize(meta['question'])
        ans_text = meta['answer']
        ans_class = self.ans2idx.get(ans_text, len(self.ans2idx) - 1)

        return img_emb, q_tokens, torch.tensor(ans_class, dtype=torch.long)
    

class VQAMultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=512, nhead=8, num_layers=6, clip_dim=512):
        super().__init__()

        # Переходник для картинок (теперь сохраняем полную ширину d_model)
        self.image_projection = nn.Linear(clip_dim, d_model)

        # Текстовый эмбеддинг (тоже расширяем до d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Обучаемый разделительный токен [SEP]
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, image_features, question_tokens):
        batch_size = image_features.size(0)

        img_emb = self.image_projection(image_features).unsqueeze(1) # [batch, 1, d_model]
        txt_emb = self.token_embedding(question_tokens) # [batch, seq_len, d_model]
        sep_emb = self.sep_token.expand(batch_size, -1, -1)

        multimodal_sequence = torch.cat([img_emb, sep_emb, txt_emb], dim=1)

        transformer_output = self.transformer(multimodal_sequence)

        pooled_output = transformer_output[:, 0, :]

        logits = self.classifier(pooled_output)
        return logits
    

if __name__ == '_main__':
    # Пути к обучаемым данным
    embeddings_path = "embeddings/val_vqa_embeddings.pt"
    metadata_path = "embeddings/val_vqa_metadata.json"

    full_dataset = VQAInMemoryDataset(embeddings_path, metadata_path, max_answers=5000)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    model = VQAMultimodalTransformer(
        vocab_size=len(full_dataset.vocab_q),
        num_classes=len(full_dataset.ans2idx),
        d_model=768,
        nhead=12,
        num_layers=6
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print("\nЗапуск обучения...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
        for img_emb, q_tokens, labels in pbar:
            img_emb = img_emb.to(device)
            q_tokens = q_tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(img_emb, q_tokens)
            loss = criterion(logits, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        scheduler.step()

        train_acc = correct_train / total_train
        avg_loss = total_loss / len(train_loader)

        # Валидация
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for img_emb, q_tokens, labels in val_loader:
                img_emb = img_emb.to(device)
                q_tokens = q_tokens.to(device)
                labels = labels.to(device)

                logits = model(img_emb, q_tokens)
                preds = torch.argmax(logits, dim=1)

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        print(f"==> Эпоха {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_heavy_vqa_model.pth")
            print("Модель сохранена")

    print(f"\nОбучение завершено! Лучшая точность на валидации: {best_val_acc*100:.2f}%")
