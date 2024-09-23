from datasets import load_dataset
import pandas as pd

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

# Flatten metadata
flatten_ds = ds.map(flatten_metadata)

# Convert the dataset to a pandas DataFrame
df = flatten_ds.to_pandas()  # Correct conversion to a DataFrame

# Step 2: Split dataset based on the pile_set_name in the metadata
groups = df.groupby('pile_set_name')

# Step 3: Save each split to a CSV file
for pile_set_name, group in groups:
    group.to_csv(f'/mnt/parscratch/users/ac1xwa/pythia/pre-train_data/{pile_set_name}.csv', index=False)

print("Dataset split and saved to CSV files.")