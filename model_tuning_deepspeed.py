import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch


def main():
    # Model and tokenizer
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Replace with actual LLaMA 3 checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to('cuda')

    # Load dataset
    DATA_FILE = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Path to your text dataset
    dataset = load_dataset('csv', data_files=DATA_FILE, split='train')

    # Tokenize dataset
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    sub_dataset = dataset.select(range(1000))
    num_processes = os.cpu_count()
    tokenized_dataset = sub_dataset.map(tokenize_function, batched=True, num_proc=num_processes, remove_columns=["text", "pile_set_name"])
    print(tokenized_dataset.column_names)

    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # DeepSpeed configuration file
    DS_CONFIG_PATH = "ds_config.json"
    ds_config = {
        "train_batch_size": 'auto',
        "gradient_accumulation_steps": 16,
        "steps_per_print": 100,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            # "allgather_partitions": True,
            # "allgather_bucket_size": 2e8,
            # "reduce_scatter": True,
            # "reduce_bucket_size": 2e8,
            # "overlap_comm": True,
            # "contiguous_gradients": True
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False
    }

    with open(DS_CONFIG_PATH, "w") as f:
        import json
        json.dump(ds_config, f)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        deepspeed=DS_CONFIG_PATH,
        report_to=["tensorboard"],
        logging_dir="./logs",
        remove_unused_columns=False
    )
    print("train_dataset: ", train_dataset.column_names)  # Inspect columns
    print("eval_dataset: ", eval_dataset.column_names)

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print(trainer._signature_columns)

    # Training
    trainer.train()

    # Save the model
    model.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")
    tokenizer.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")

if __name__ == "__main__":
    main()