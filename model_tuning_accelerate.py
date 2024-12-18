import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from torch.nn.utils.rnn import pad_sequence
from transformers import get_scheduler

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

    # Clean up any LoRA-specific attributes if present
    if hasattr(model, "peft_config"):
        del model.peft_config  # Remove PEFT configuration if it exists

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def preprocess_for_next_token_prediction(sample, tokenizer, max_length):
    """
    Tokenizes each text in the dataset for next-token prediction.
    :param sample: Dictionary containing 'text' to tokenize
    :param tokenizer: Tokenizer for the model
    :param max_length: Maximum sequence length for the model
    """
    # Tokenize the text column and create labels for next-token prediction
    encoding = tokenizer(sample["text"], truncation=True, max_length=max_length, padding="max_length")
    sample["input_ids"] = encoding["input_ids"]
    sample["attention_mask"] = encoding["attention_mask"]
    sample["labels"] = encoding["input_ids"].copy()  # Copy input_ids as labels for next-token prediction
    return sample

def preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed):
    """
    Prepares the dataset for next-token prediction by applying tokenization.
    :param dataset: The original dataset with 'text' column
    :param tokenizer: Tokenizer to use for tokenizing the text
    :param max_length: Max token length for inputs
    :param seed: Seed for shuffling dataset
    """
    # Remove any null or empty texts
    dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text'].strip()) > 0)
    
    # Apply preprocessing with proper batching
    tokenized_dataset = dataset.map(
        lambda x: preprocess_for_next_token_prediction(x, tokenizer, max_length),
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Tokenizing dataset",
        batched=False  # Process one sample at a time to avoid inconsistencies
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    # Shuffle the dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)

    return tokenized_dataset

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches properly for causal LM.
    """
    input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
    attention_mask = [torch.tensor(sample['attention_mask']) for sample in batch]
    labels = [torch.tensor(sample['labels']) for sample in batch]

    # Pad sequences to the longest in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Padding for labels is -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def fine_tune(model, tokenizer, dataset, per_device_train_batch_size, gradient_accumulation_steps, warmup_steps, max_steps, learning_rate, output_dir):
    # Initialise Accelerator for multi-GPU training
    accelerator = Accelerator()

    # DataLoader setup
    dataloader = DataLoader(
        dataset, 
        batch_size=per_device_train_batch_size, 
        collate_fn=custom_collate_fn, 
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    # Optimizer setup
    # Learning rate scheduler
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=max_steps
        )
    # Prepare model, dataloader, and optimizer using Accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

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
    subset = dataset.select(range(1000))
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 4096
    preprocessed_dataset = preprocess_dataset_for_next_token_prediction(subset, tokenizer, max_length, seed)

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
