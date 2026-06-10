import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQAInMemoryDataset(Dataset):
    def __init__(self, compiled_metadata_path, embeddings_path):
        print("Загрузка скомпилированных метаданных...")
        payload = torch.load(compiled_metadata_path, map_location='cpu', mmap=True)

        self.samples = payload['samples']
        self.num_classes = payload['num_classes']
        
        print("Загрузка эмбеддингов патчей картинок...")
        self.embeddings = torch.load(embeddings_path, map_location='cpu', mmap=True)

        print(f"-> Количество классов-ответов (из .pt): {self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_emb = self.embeddings[sample['image_path']]
        if img_emb.shape[0] != 50:
            raise ValueError(f"Ошибка: Ожидалось 50 токенов CLIP, получено {img_emb.shape[0]} для {sample['image_path']}")
            
        q_tokens = sample['q_tokens']
        attention_mask = sample['attention_mask']
        soft_target = sample['soft_target'].float()

        target_sum = soft_target.sum()
        if target_sum > 0:
            soft_target = soft_target / target_sum

        return img_emb.clone(), q_tokens.clone(), attention_mask.clone(), soft_target.clone()
    

class VQACrossAttentionTransformer(nn.Module):
    def __init__(self, num_classes, d_model=768, nhead=12, num_layers=4, clip_dim=512):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.embeddings.parameters():
            p.requires_grad = True
        for layer in self.text_encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.fusion_token = nn.Parameter(torch.randn(1, 1, d_model))

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, image_features, question_tokens, attention_mask):
        batch_size = image_features.size(0)

        bert_output = self.text_encoder(input_ids=question_tokens, attention_mask=attention_mask)
        txt_emb = bert_output.last_hidden_state # [batch, 25, 768]

        img_emb = self.image_projection(image_features) # [batch, 50, 768]

        fusion_tokens = self.fusion_token.expand(batch_size, -1, -1) # [batch, 1, 768]

        full_sequence = torch.cat([fusion_tokens, txt_emb, img_emb], dim=1)

        fusion_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        text_mask = (attention_mask == 0)
        image_mask = torch.zeros((batch_size, 50), dtype=torch.bool, device=device)

        full_attention_mask = torch.cat([fusion_mask, text_mask, image_mask], dim=1)

        transformer_out = self.fusion_transformer(full_sequence, src_key_padding_mask=full_attention_mask)

        final_repr = transformer_out[:, 0]

        return self.classifier(final_repr)
    

def compute_vqa_accuracy(logits, soft_labels):
    preds = torch.argmax(logits, dim=1)
    scores = soft_labels[torch.arange(soft_labels.size(0)), preds]
    return scores.sum().item()

def compute_vqa_top5_accuracy(logits, soft_labels):
    top5_indices = torch.topk(logits, k=5, dim=1).indices # [batch, 5]
    batch_size = soft_labels.size(0)

    matched = torch.zeros(batch_size, device=logits.device)
    for i in range(5):
        idx = top5_indices[:, i]
        scores = soft_labels[torch.arange(batch_size), idx]
        matched = torch.max(matched, scores)
    return matched.sum().item()


def soft_cross_entropy(logits, targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


if __name__ == '__main__':
    patches_path = "/embeddings/val_vqa_patches_dict.pt"
    metadata_path = "/embeddings/val_vqa_compiled_metadata.pt"

    full_dataset = VQAInMemoryDataset(metadata_path, patches_path)
    model = VQACrossAttentionTransformer(
        num_classes=full_dataset.num_classes,
        d_model=768,
        nhead=12,
        num_layers=4,
    ).to(device)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=612, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=612, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.fusion_token], "lr": 5e-4},
            {"params": model.image_projection.parameters(), "lr": 2e-4},
            {"params": model.classifier.parameters(), "lr": 2e-4},
            {"params": model.fusion_transformer.parameters(), "lr": 2e-4},
            {"params": model.text_encoder.embeddings.parameters(), "lr": 2e-5},
            {"params": model.text_encoder.encoder.layer[-2:].parameters(), "lr": 3e-5},
        ],
        weight_decay=1e-2
    )
    epochs = 15
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    best_vqa_acc = 0

    print("Запускаем обучение...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
        for img_patches, q_tokens, attention_mask, soft_labels in pbar:
            img_patches = img_patches.to(device)
            q_tokens = q_tokens.to(device)
            attention_mask = attention_mask.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(img_patches, q_tokens, attention_mask)
                loss = soft_cross_entropy(logits, soft_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_loss += loss.item()
            correct_train += compute_vqa_accuracy(logits, soft_labels)
            total_train += soft_labels.size(0)

            current_lr = optimizer.param_groups[3]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr': f"{current_lr:.2e}"
            })

        train_vqa_acc = correct_train / total_train
        avg_loss = total_loss / len(train_loader)

        # Валидация
        model.eval()
        correct_val = 0
        correct_top5 = 0
        total_val = 0

        with torch.no_grad():
            for img_patches, q_tokens, attention_mask, soft_labels in val_loader:
                img_patches = img_patches.to(device)
                q_tokens = q_tokens.to(device)
                attention_mask = attention_mask.to(device)
                soft_labels = soft_labels.to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(img_patches, q_tokens, attention_mask)

                correct_val += compute_vqa_accuracy(logits, soft_labels)
                correct_top5 += compute_vqa_top5_accuracy(logits, soft_labels)
                total_val += soft_labels.size(0)

        val_vqa_acc = correct_val / total_val
        val_top5_acc = correct_top5 / total_val

        print(f"==> Эпоха {epoch+1} | Train Loss: {avg_loss:.4f} | "
              f"Train VQA Acc: {train_vqa_acc*100:.2f}% | "
              f"Val VQA Acc: {val_vqa_acc*100:.2f}% | "
              f"Val Top5 VQA Acc: {val_top5_acc*100:.2f}%")

        if val_vqa_acc > best_vqa_acc:
            best_vqa_acc = val_vqa_acc
            torch.save(model.state_dict(), "best_patch_vqa_model.pth")
            print("Новая лучшая модель сохранена!")
            