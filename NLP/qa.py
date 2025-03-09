# 허깅페이스 인증코드 받아야함
# from transformers import pipeline

# question = "How many programming languages does BLOOM support?"
# context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

# question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
# result = question_answerer(question=question, context=context)
# print(result)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
model = AutoModelForQuestionAnswering.from_pretrained("yjgwak/klue-bert-base-finetuned-squard-kor-v1")
tokenizer = AutoTokenizer.from_pretrained("yjgwak/klue-bert-base-finetuned-squard-kor-v1")
question = "오늘 날씨가 어때?"
context = "오늘은 맑은 날씨에 오후 비가 조금 오는 날씨야."
input = tokenizer(question, context, return_tensors="pt")
outputs = model(**input)
start_scores = outputs.start_logits
end_scores = outputs.end_logits
# print output text
print(tokenizer.decode(input["input_ids"][0][torch.argmax(start_scores):torch.argmax(end_scores)+1]))

