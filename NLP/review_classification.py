from transformers import pipeline

# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "I don't think this plan will work."

# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
result = classifier(text)
# print(result)

if result[0]['label'] == 'LABEL_1':
    print("This is a positive review.")
else:
    print("This is a negative review.")

