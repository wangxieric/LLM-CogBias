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

# Load dataset
dataset_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
for category in categories:
    # read data in csv format
    ds = pd.read_csv(f"{dataset_dir}/{category}.csv")
    print(ds.shape)
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
    indexref = iter_indexer.index(ds.to_dict(orient='records'))
    print(f"Index saved at: {indexref}")
    print(f"Finish indexing {category} ", datetime.now())

