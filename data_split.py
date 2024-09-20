from datasets import load_dataset
import pandas as pd

def flatten_metadata(example):
    # Extract 'pile_set_name' from 'metadata' and add it as a separate column
    example['pile_set_name'] = example['meta']['pile_set_name']
    return example

# Load dataset
ds = load_dataset("monology/pile-uncopyrighted", split="train")

# Flatten metadata
flatten_ds = ds.map(flatten_metadata)

# Convert the dataset to a pandas DataFrame
df = flatten_ds.set_format(type='pandas', columns=['text', 'pile_set_name'])  # Use the appropriate split, e.g., "train", "test"

# Step 2: Split dataset based on the pile_set_name in the metadata
groups = df.groupby(df['pile_set_name'])

# Step 3: Save each split to a CSV file
for pile_set_name, group in groups:
    group.to_csv(f'/mnt/parscratch/users/ac1xwa/pythia/pre-train_data/{pile_set_name}.csv', index=False)

print("Dataset split and saved to CSV files.")