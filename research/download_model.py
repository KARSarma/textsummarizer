from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Specify the model checkpoint
model_ckpt = "google/pegasus-cnn_dailymail"

# Load and save the model and tokenizer locally
print("Downloading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Specify the local directory to save the model
save_directory = "artifacts/local_model"

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved to {save_directory}")
