import cv2
import torch
import asyncio
from json import load
from googletrans import Translator
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from model import VQACrossAttentionTransformer
from convert_meta import max_answers, max_question_length


device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_to_model = 'best_patch_vqa_model.pth'
path_to_ans2idx = 'ans2idx.json'

model = VQACrossAttentionTransformer(num_classes=max_answers, d_model=768, nhead=12, num_layers=4)
model.load_state_dict(torch.load(path_to_model, map_location=device))
model.to(device).eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
BERT_model = AutoTokenizer.from_pretrained("bert-base-uncased")

with open(path_to_ans2idx, 'r', encoding='utf-8') as f:
    ans2idx = load(f)
idx2ans = {v: k for k, v in ans2idx.items()}


async def translate_text(text, source, end):
    async with Translator() as translator:
        res = await translator.translate(text, src=source, dest=end)
        return res.text
     

def proceed_text_for_question(question):
    question = asyncio.run(translate_text(question, 'ru', 'en'))
    with torch.no_grad():
        encoded = BERT_model(question, padding="max_length", truncation=True, max_length=max_question_length, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def proceed_img_for_question(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        patch_features = clip_model.visual_projection(vision_outputs.last_hidden_state)
        return (patch_features / patch_features.norm(dim=-1, keepdim=True)).to(device)


cap = cv2.VideoCapture(0)
current_answer = "Жду вопрос..."
question_text = "Сколько людей в кадре?"
print("Нажми 'q' для выхода, 'r' чтобы переспросить")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, current_answer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VQA', frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        print("Обработка...")
        patch_features = proceed_img_for_question(frame)
        q_tokens, attention_mask = proceed_text_for_question(question_text)
        print(patch_features.shape)
        print(q_tokens.shape)
        print(attention_mask.shape)
        with torch.no_grad():
            logits = model(patch_features, q_tokens, attention_mask)
            current_answer = idx2ans[logits.argmax(dim=-1).item()]
            print("Ответ:", asyncio.run(translate_text(current_answer, 'en', 'ru')))
    elif key == 27 or key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
