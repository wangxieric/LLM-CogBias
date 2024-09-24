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

# Extract the columns into lists
text_data = flatten_ds['text']
pile_set_names = flatten_ds['pile_set_name']

# Create a pandas DataFrame manually
df = pd.DataFrame({
    'text': text_data,
    'pile_set_name': pile_set_names
})

# Step 2: Split dataset based on the pile_set_name in the metadata
groups = df.groupby('pile_set_name')

# Step 3: Save each split to a CSV file
for pile_set_name, group in groups:
    if pd.notna(pile_set_name): 
        group.to_csv(f'/mnt/parscratch/users/ac1xwa/pythia/pre-train_data/{pile_set_name}.csv', index=False)

print("Dataset split and saved to CSV files.")