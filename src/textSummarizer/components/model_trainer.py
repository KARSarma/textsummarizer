from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk
import os


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        # Set device to CPU for training
        device = "cpu"

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Enable gradient checkpointing for better memory management
        model.gradient_checkpointing_enable()

        # Initialize the data collator for sequence-to-sequence models
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Load dataset and subset for training and evaluation
        dataset = load_from_disk(self.config.data_path)
        train_dataset = dataset["test"].select(range(self.config.train_subset))
        eval_dataset = dataset["validation"].select(range(self.config.eval_subset))

        # Set training arguments
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=False  # Use FP16 only with GPUs
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train the model
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))


class ModelTrainerConfig:
    def __init__(self, root_dir, data_path, model_ckpt, num_train_epochs, warmup_steps,
                 per_device_train_batch_size, per_device_eval_batch_size, weight_decay, logging_steps,
                 evaluation_strategy, eval_steps, save_steps, gradient_accumulation_steps, train_subset, eval_subset):
        self.root_dir = root_dir
        self.data_path = data_path
        self.model_ckpt = model_ckpt
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_subset = train_subset
        self.eval_subset = eval_subset

