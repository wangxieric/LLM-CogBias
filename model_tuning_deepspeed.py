import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

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
        "deepspeed_multinode_launcher": "standard",
        "offload_optimizer_device": "none",
        "offload_param_device": "none",
        "zero3_init_flag": False,
        "zero3_save_16bit_model": True,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "nvme",
            }
        },
        "fp16":{
            "enabled": True
        } 
    }

    with open(DS_CONFIG_PATH, "w") as f:
        import json
        json.dump(ds_config, f)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        save_steps=1000,
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