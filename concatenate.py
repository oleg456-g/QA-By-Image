import os
import json
from tqdm import tqdm

def main():
    questions_json_path = "v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_json_path = "v2_mscoco_val2014_annotations.json"
    images_dir = "val2014"
    output_json_path = "vqa_val2014_preprocessed.json"
    
    print("1. Чтение сырых файлов...")
    with open(questions_json_path, 'r', encoding='utf-8') as f:
        questions_list = json.load(f)['questions']
    with open(annotations_json_path, 'r', encoding='utf-8') as f:
        annotations_list = json.load(f)['annotations']
        
    print("2. Создание словаря...")
    annotations_dict = {
        ann['question_id']: {
            'multiple_choice_answer': ann['multiple_choice_answer'],
            'answer_type': ann['answer_type'],
            'all_answers': [a['answer'] for a in ann['answers']]
        }
        for ann in annotations_list
    }
    
    print("3. Склеивание данных...")
    preprocessed_data = []
    
    for q in tqdm(questions_list):
        q_id = q['question_id']
        if q_id in annotations_dict:
            image_id = int(q['image_id'])
            image_filename = f"COCO_val2014_{image_id:012d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            preprocessed_data.append({
                'question_id': q_id,
                'image_path': image_path,
                'question': q['question'],
                'answer': annotations_dict[q_id]['multiple_choice_answer'],
                'answer_type': annotations_dict[q_id]['answer_type'],
                'all_answers': annotations_dict[q_id]['all_answers']
            })
            
    print(f"4. Сохранение датасета в {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
        
    print("Готово!")

if __name__ == "__main__":
    main()
