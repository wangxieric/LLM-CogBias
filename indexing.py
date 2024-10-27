import pandas as pd
import pyterrier as pt
from datetime import datetime
categories = ['ArXiv', 'Enron Emails', 'FreeLaw', 'Gutenberg (PG-19)', 'NIH ExPorter', 'Pile-CC', 
              'PubMed Central', 'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', 'EuroParl', 
              'Github', 'HackerNews', 'PhilPapers', 'PubMed Abstracts', 'StackExchange', 'USPTO Backgrounds']

# Load dataset & indexing using pyterrier
# Initialize PyTerrier
if not pt.started():
    pt.init()

# # Load dataset
# dataset_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
# for category in categories:
#     # read data in csv format
#     ds = pd.read_csv(f"{dataset_dir}/{category}.csv")
#     print(f"Original shape: {ds.shape}")
    
#     # Filter out rows where 'text' is empty or only whitespace
#     ds = ds[ds['text'].notnull() & ds['text'].str.strip().astype(bool)]
#     print(f"Filtered shape (no empty documents): {ds.shape}")

#     columns = ds.columns # text, pile_set_name
#     print(f"Columns: {columns}")

#     # add docno and combined with category name
#     ds['docno'] = ds.index  # add docno 
#     ds['docno'] = ds['docno'].apply(lambda x: f"{category}-{x}")
#     ds['docno'] = ds['docno'].astype(str)
#     print("Finish adding docno ", datetime.now())
#     # Drop pile_set_name column
#     ds = ds.drop(columns=['pile_set_name'])
#     iter_indexer = pt.IterDictIndexer(f"{dataset_dir}/index/{category}")
#     indexref = iter_indexer.index(ds.to_dict(orient='records'))
#     print(f"Index saved at: {indexref}")
#     print(f"Finish indexing {category} ", datetime.now())


# OOM error handling via chunking

dataset_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
chunk_size = 10000  # Adjust based on memory availability

for category in categories:
    index_path = f"{dataset_dir}/index/{category}"
    iter_indexer = pt.IterDictIndexer(index_path)
    print(f"Indexing {category}...")

    for chunk_idx, ds_chunk in enumerate(pd.read_csv(f"{dataset_dir}/{category}.csv", chunksize=chunk_size)):
        print(f"Processing chunk {chunk_idx} with shape: {ds_chunk.shape}")
        
        # Filter out rows where 'text' is empty or only whitespace
        ds_chunk = ds_chunk[ds_chunk['text'].notnull() & ds_chunk['text'].str.strip().astype(bool)]
        
        # Add 'docno' column
        ds_chunk['docno'] = ds_chunk.index + (chunk_idx * chunk_size)
        ds_chunk['docno'] = ds_chunk['docno'].apply(lambda x: f"{category}-{x}")
        ds_chunk['docno'] = ds_chunk['docno'].astype(str)
        
        # Drop unnecessary columns
        ds_chunk = ds_chunk.drop(columns=['pile_set_name'])

        # Index each filtered chunk
        indexref = iter_indexer.index(ds_chunk.to_dict(orient='records'))
        print(f"Chunk {chunk_idx} indexed at {datetime.now()}")

    print(f"Finished indexing {category} at {index_path} at {datetime.now()}")