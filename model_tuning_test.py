import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Specify model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset from CSV
    dataset_name = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Update with the correct CSV file path
    dataset = load_dataset("csv", data_files=dataset_name)
    seed = 42
    subset = dataset.select(range(1000))

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = subset.map(preprocess_function, batched=True)

    # Prepare the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="none",  # Change to "wandb" or other if tracking is needed
        optim="adamw_torch",
        dataloader_pin_memory=False,
    )

    # Data collator
    def data_collator(features):
        labels = torch.tensor([f["input_ids"] for f in features])
        input_ids = labels.clone()
        attention_mask = torch.tensor([f["attention_mask"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
