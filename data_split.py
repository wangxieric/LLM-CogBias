from datasets import load_dataset
import pandas as pd

# Load dataset
ds = load_dataset("monology/pile-uncopyrighted", split="train")

# Convert the dataset to a pandas DataFrame
df = ds.to_pandas()  # Use the appropriate split, e.g., "train", "test"

# Step 2: Split dataset based on the pile_set_name in the metadata
groups = df.groupby(df['metadata'].apply(lambda x: x['pile_set_name']))

# Step 3: Save each split to a CSV file
for pile_set_name, group in groups:
    group.to_csv(f'{pile_set_name}.csv', index=False)

print("Dataset split and saved to CSV files.")