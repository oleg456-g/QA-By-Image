import cv2
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from json import load
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
whisper_model = WhisperModel(
    "small",
    device=device,
    compute_type="int8"
)

with open(path_to_ans2idx, 'r', encoding='utf-8') as f:
    ans2idx = load(f)
idx2ans = {v: k for k, v in ans2idx.items()}


def recognize_speech(duration=5):
    fs = 16000
    print("Говорите...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16',
        device=2
    )

    sd.wait()
    write("question.wav", fs, audio)
    segments, _ = whisper_model.transcribe(
        "question.wav",
        task="translate"
    )

    text = " ".join(segment.text for segment in segments)
    print("Распознано:", text)
    return text
     

def proceed_text_for_question(question):
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
print("Нажми 'q' для выхода, 'r' чтобы задать вопрос и получить ответ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, current_answer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VQA', frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        question_text = recognize_speech()
        print("Обработка...")
        patch_features = proceed_img_for_question(frame)
        q_tokens, attention_mask = proceed_text_for_question(question_text)
        with torch.no_grad():
            logits = model(patch_features, q_tokens, attention_mask)
            current_answer = idx2ans[logits.argmax(dim=-1).item()]
            print("Ответ:", current_answer)

    elif key == 27 or key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
