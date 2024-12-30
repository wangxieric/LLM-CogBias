import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

def main():
    # Model and tokenizer
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Replace with actual LLaMA 3 checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to('cuda')
    model.gradient_checkpointing_enable()

    # Load dataset
    DATA_FILE = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Path to your text dataset
    dataset = load_dataset('csv', data_files=DATA_FILE, split='train')

    # Tokenize dataset
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=2048)
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    sub_dataset = dataset.select(range(1000))
    num_processes = os.cpu_count()
    tokenized_dataset = sub_dataset.map(tokenize_function, batched=True, num_proc=num_processes)

    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        warmup_ratio=0.1,
        learning_rate=8e-5,
        num_train_epochs=2,
        fp16=True,
        report_to="none",
        deepspeed="ds_config.json"
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