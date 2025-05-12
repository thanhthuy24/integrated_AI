from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Tải mô hình về máy
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Lưu vào thư mục cục bộ
model.save_pretrained("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")

print("✅ Mô hình đã tải về máy!")