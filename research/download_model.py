from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_ckpt = "google/pegasus-cnn_dailymail"
save_directory = "artifacts/local_model"

print("Downloading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, cache_dir="./model_cache")
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, cache_dir="./model_cache")

# Save the model and tokenizer
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print(f"Model and tokenizer saved to {save_directory}")
