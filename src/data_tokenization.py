from datasets import load_dataset
import os
from transformers import AutoTokenizer

characters = ['literal_classist', 'scientific_scholar', 'scientific_mathematician',
                  'legal_analyst', 'biomedical_expert', 'health_advisor',
                  'business_advisor', 'technical_communicator', 'cultural_scholar', 
                  'patent_strategist', 'inventive_technologist']

base_model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

max_length = 512

for i, character in enumerate(characters):
    if i == 0:
        continue
    DATA_FILE = f"/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/{character}_{max_length}.csv"
    dataset = load_dataset('csv', data_files=DATA_FILE, split='train')

    num_processes = os.cpu_count()
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=2)


    # save the tokenized dataset        
    TOKENISED_DATASET_PATH = f"/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/tokenized_{character}_{max_length}"
    tokenized_dataset.save_to_disk(TOKENISED_DATASET_PATH)