import json
import os
import torch
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

max_answers = 5000
max_question_length = 25
input_json = "vqa_val2014_preprocessed.json"
output_pt = "embeddings/val_vqa_compiled_metadata.pt"
os.makedirs(os.path.dirname(output_pt), exist_ok=True)

def main():
    print("Чтение JSON...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Загружено записей: {len(data)}")
    print("Строим словарь ответов...")

    answers = [item["answer"] for item in data]
    most_common_answers = Counter(answers).most_common(max_answers)
    ans2idx = {ans: idx for idx, (ans, _) in enumerate(most_common_answers)}

    print(f"Количество классов ответов: {len(ans2idx)}")
    print("Токенизация вопросов и сборка soft targets...")

    compiled_samples = []

    for item in tqdm(data):
        encoded = tokenizer(
            item["question"],
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt"
        )

        q_tokens = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        soft_target = torch.zeros(max_answers, dtype=torch.float16)
        all_answers = item.get("all_answers", [item["answer"]])

        answer_counts = Counter(all_answers)
        for ans, count in answer_counts.items():
            if ans in ans2idx:
                soft_target[ans2idx[ans]] = min(1.0, count / 3.0)

        compiled_samples.append({
            "image_path": item["image_path"],
            "q_tokens": q_tokens,
            "attention_mask": attention_mask,
            "soft_target": soft_target
        })

    with open("ans2idx.json", 'w', encoding='utf-8') as f:
        json.dump(ans2idx, f, ensure_ascii=False, indent=2)

    print("Сохранение...")

    payload = {
        "samples": compiled_samples,
        "num_classes": max_answers,
        "bert_vocab_size": tokenizer.vocab_size
    }
    torch.save(payload, output_pt)

    print(f"\nФайл сохранен: {output_pt}")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    main()
