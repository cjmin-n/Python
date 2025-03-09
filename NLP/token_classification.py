import json
# from transformers import pipeline, BertForTokenClassification

# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
# text = "집에 강력히 가고 싶다.."

# # classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
# classifier = pipeline("ner", model="KPF/KPF-bert-ner")
# # huggingface 개체명 인식 모델 불러오기
# model = BertForTokenClassification.from_pretrained("KPF/KPF-bert-ner")

# result = classifier(text)

# print(result)



from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner = pipeline("ner", model=model, tokenizer=tokenizer)
example = "서울역으로 안내해줘."
ner_results = ner(example)
print(ner_results)

# Convert the result to JSON-serializable format
json_result = []
for item in ner_results:
    json_result.append({
        'entity': item['entity'],
        'score': float(item['score']),  # Convert numpy.float32 to Python float
        'word': item['word'],
        'start': item['start'],
        'end': item['end']
    })

print(json.dumps(json_result, indent=4))

