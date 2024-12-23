import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Specify model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load dataset from CSV
    dataset_name = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Update with the correct CSV file path
    dataset = load_dataset("csv", data_files=dataset_name, split="train")
    subset = dataset.select(range(1000))

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = subset.map(preprocess_function, batched=True)

    # Convert to DataLoader
    train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Prepare everything with the accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    model.train()
    for epoch in range(3):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**{k: v.to(accelerator.device) for k, v in batch.items()})
            loss = outputs.loss

            if step % 10 == 0:
                print(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()}")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed")

    # Save the trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist", save_function=accelerator.save)
    tokenizer.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")

if __name__ == "__main__":
    main()
