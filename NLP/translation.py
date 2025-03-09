from transformers import pipeline
from huggingface_hub import login
text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
translator = pipeline("translation_xx_to_yy", model="google-t5/t5-small")
result = translator(text)
print(result)