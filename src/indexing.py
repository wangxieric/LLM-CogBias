import pandas as pd
import pyterrier as pt
from datetime import datetime
import gc

# categories = ['ArXiv', 'Enron Emails', 'FreeLaw', 'Gutenberg (PG-19)', 'NIH ExPorter', 'Pile-CC', 
#               'PubMed Central', 'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', 'EuroParl', 
#               'Github', 'HackerNews', 'PhilPapers', 'PubMed Abstracts', 'StackExchange', 'USPTO Backgrounds']

categories = ['Pile-CC', 
              'PubMed Central', 'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', 'EuroParl', 
              'Github', 'HackerNews', 'PhilPapers', 'PubMed Abstracts', 'StackExchange', 'USPTO Backgrounds']

# Load dataset & indexing using pyterrier
# Initialize PyTerrier
if not pt.started():
    pt.init()


def document_generator(df, category, batch_size=10000):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            yield {
                'docno': f"{category}-{row.name}",
                'text': row['text'][:4096]  # Limit text to 4096 characters
            }
        gc.collect()  # Force garbage collection after each batch

# # Load dataset
dataset_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
for category in categories:
    # read data in csv format
    ds = pd.read_csv(f"{dataset_dir}/{category}.csv")
    print(f"Original shape: {ds.shape}")
    
    # Filter out rows where 'text' is empty or only whitespace
    ds = ds[ds['text'].notnull() & ds['text'].str.strip().astype(bool)]
    print(f"Filtered shape (no empty documents): {ds.shape}")

    columns = ds.columns # text, pile_set_name
    print(f"Columns: {columns}")

    # add docno and combined with category name
    ds['docno'] = ds.index  # add docno 
    ds['docno'] = ds['docno'].apply(lambda x: f"{category}-{x}")
    ds['docno'] = ds['docno'].astype(str)
    print("Finish adding docno ", datetime.now())
    # Drop pile_set_name column
    ds = ds.drop(columns=['pile_set_name'])
    iter_indexer = pt.IterDictIndexer(f"{dataset_dir}/index/{category}")
    indexref = iter_indexer.index(document_generator(ds, category))
    print(f"Index saved at: {indexref}")
    print(f"Finish indexing {category} ", datetime.now())

    # Clear memory
    del ds, iter_indexer
    gc.collect()

# # OOM error handling via chunking

# dataset_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
# index_base_path = f"{dataset_dir}/index"
# chunk_size = 10000  # Adjust based on memory availability

# for category in categories:
#     # Index new documents in chunks
#     for chunk_idx, ds_chunk in enumerate(pd.read_csv(f"{dataset_dir}/{category}.csv", chunksize=chunk_size)):
#         # Path to the existing index
#         existing_index_path = f"{index_base_path}/{category}"
        
#         # Load the existing index if it exists
#         if os.path.exists(existing_index_path):
#             print(f"Loading existing index at {existing_index_path}")
#             existing_index = pt.IndexFactory.of(existing_index_path)
#         else:
#             print(f"Creating a new index for {category}")
#             existing_index = None

#         # Temporary path for new documents
#         temp_index_path = f"{index_base_path}/temp_{category}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
#         os.makedirs(temp_index_path, exist_ok=True)
#         iter_indexer = pt.IterDictIndexer(temp_index_path)
    
#         print(f"Processing chunk {chunk_idx} for {category}")

#         # Filter out empty or null text rows
#         ds_chunk = ds_chunk[ds_chunk['text'].notnull() & ds_chunk['text'].str.strip().astype(bool)]
        
#         # Add 'docno' column
#         ds_chunk['docno'] = ds_chunk.index + (chunk_idx * chunk_size)
#         ds_chunk['docno'] = ds_chunk['docno'].apply(lambda x: f"{category}-{x}")
        
#         # Drop unnecessary columns
#         ds_chunk = ds_chunk.drop(columns=['pile_set_name'])
        
#         # Index this chunk to the temporary index
#         iter_indexer.index(ds_chunk.to_dict(orient='records'))

#         # Merge temporary index with existing index if it exists
#         if existing_index:
#             print(f"Merging {temp_index_path} into existing index at {existing_index_path}")
#             pt.IndexMerger([existing_index, temp_index_path]).merge(existing_index_path)
#         else:
#             print(f"Copying {temp_index_path} to {existing_index_path} as the initial index")
#             shutil.move(temp_index_path, existing_index_path)
        
#         # Clean up temporary index directory
#         if os.path.exists(temp_index_path):
#             shutil.rmtree(temp_index_path)

#     print(f"Finished updating index for {category} at {existing_index_path} at {datetime.now()}")
