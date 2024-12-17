import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

def create_bnb_config(load_in_4bit, bnb_4_bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
    Configure model quantization using bitsandbytes to speed up training and inference
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4_bit_use_double_quant=bnb_4_bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    return bnb_config

def load_model(model_name, bnb_config):
    """
    Load the model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def preprocess_for_next_token_prediction(sample, tokenizer, max_length):
    """
    Tokenizes each text in the dataset for next-token prediction.
    """
    encoding = tokenizer(sample["text"], truncation=True, max_length=max_length)
    sample["input_ids"] = encoding["input_ids"]
    sample["attention_mask"] = encoding["attention_mask"]
    sample["labels"] = encoding["input_ids"].copy()
    return sample

def preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed):
    """
    Prepares the dataset for next-token prediction by applying tokenization.
    """
    dataset = dataset.map(lambda sample: preprocess_for_next_token_prediction(sample, tokenizer, max_length), batched=True)
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset

def fine_tune(model, tokenizer, dataset, per_device_train_batch_size, gradient_accumulation_steps, warmup_steps, max_steps, learning_rate, output_dir):
    # Initialise Accelerator for multi-GPU training
    accelerator = Accelerator()

    # DataLoader setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, collate_fn=data_collator)

    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Prepare model, dataloader, and optimizer using Accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Training loop for 1 epoch
    model.train()
    total_steps = 0
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        total_steps += 1
        # Logging progress
        if total_steps % 10 == 0:
            print(f"Step {total_steps}: Loss = {loss.item():.4f}")

    # Save model
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    model.save_pretrained(output_dir, save_function=accelerator.save)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"

    # Bitsandbytes parameters
    load_in_4bit = True
    bnb_4bit_use_double_quant = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = torch.bfloat16

    # Load model from Hugging Face Hub with model name and bitsandbytes configuration
    bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
    model, tokenizer = load_model(model_name, bnb_config)

    # Load and preprocess dataset
    dataset_name = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"
    dataset = load_dataset('csv', data_files=dataset_name, split='train')
    seed = 42
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 4096
    preprocessed_dataset = preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed)

    # Training parameters
    output_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_steps = 4000

    fine_tune(
        model,
        tokenizer,
        preprocessed_dataset,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        warmup_steps=2,
        max_steps=max_steps,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
