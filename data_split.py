from datasets import load_dataset
import pandas as pd
from tqdm import tqdm 
import csv

def flatten_metadata(example):
    # Ensure 'meta' and 'pile_set_name' keys exist
    if 'meta' in example and 'pile_set_name' in example['meta']:
        example['pile_set_name'] = example['meta']['pile_set_name']
    else:
        # If the keys are missing, set pile_set_name as None
        example['pile_set_name'] = None
    return example

# Load dataset
ds = load_dataset("monology/pile-uncopyrighted", split="train")

print("Star processing the dataset...")
# Flatten metadata
flatten_ds = ds.map(flatten_metadata)

file_handles = {}

# Process each example and write directly to its corresponding CSV file
for example in tqdm(flatten_ds, desc="Processing examples", unit="example"):
    pile_set_name = example['pile_set_name']
    text = example['text']

    if pile_set_name is not None:
        if pile_set_name not in file_handles:
            # Open a new CSV file for this pile_set_name
            file_handles[pile_set_name] = open(f'/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/{pile_set_name}.csv', 'a')
            csv_writer = csv.writer(file_handles[pile_set_name])
            
            # Write header row for the first time
            csv_writer.writerow(['text','pile_set_name'])
        else:
            csv_writer = csv.writer(file_handles[pile_set_name])
            
        # Write the row to the corresponding file
        csv_writer.writerow([text, pile_set_name])

# Close all open file handles
for handle in file_handles.values():
    handle.close()

print("Dataset split and saved to CSV files.")