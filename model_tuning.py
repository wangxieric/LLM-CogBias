import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

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
    n_gpus = torch.cuda.device_count()
    max_memory = f'{70}GB'
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 max_memory={i: max_memory for i in range(n_gpus)})

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
    sample["labels"] = encoding["input_ids"].copy()  # Next-token prediction labels
    return sample

def preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed):
    """
    Prepares the dataset for next-token prediction by applying tokenization.
    """
    dataset = dataset.map(lambda sample: preprocess_for_next_token_prediction(sample, tokenizer, max_length), batched=True)
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset

def fine_tune(model, tokenizer, dataset, per_device_train_batch_size, 
              gradient_accumulation_steps, warmup_steps, max_steps, 
              learning_rate, fp16, logging_steps, output_dir, optim):
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Print model parameters to check total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All Parameters: {total_params:,d} || Trainable Parameters: {trainable_params:,d}")

    # Training parameters
    trainer = Trainer(
        model=model, 
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=logging_steps,
            output_dir=output_dir,
            optim=optim
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Disable cache
    model.config.use_cache = False
    print("Training model without adapter")

    # Start training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free up memory
    del model
    del trainer
    torch.cuda.empty_cache()

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
    dataset_name = "xx.csv"
    dataset = load_dataset('csv', data_files=dataset_name, split='train')
    seed = 42
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 4096
    preprocessed_dataset = preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed)

    # TrainingArguments parameters
    output_dir = "./fine_tune_llama3_xx"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    optim = "paged_adamw_32bit"
    max_steps = 4000
    warmup_steps = 2
    fp16 = True
    logging_steps = 1

    fine_tune(
        model,
        tokenizer,
        preprocessed_dataset,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        warmup_steps,
        max_steps,
        learning_rate,
        fp16,
        logging_steps,
        output_dir,
        optim
    )
