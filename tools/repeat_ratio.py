import os
import os.path as osp
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


dataset = 'GoogleLocal'
path=f'../data/{dataset}/{dataset}.csv'

df = pd.read_csv(path)

edge_first_seen = defaultdict(lambda: None) 

repeated_edges = 0
duplicate_first_time_count = 0
total_edges = len(df)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing edges"):
    src = row['src']
    dst = row['dst']
    time = row['time']
    
    if edge_first_seen[(src, dst)] is None:
        edge_first_seen[(src, dst)] = time

    if time > edge_first_seen[(src, dst)]:
        repeated_edges += 1
        continue
    else:
        duplicate_first_time_count += 1

duplicate_ratio = repeated_edges / total_edges if total_edges > 0 else 0

print(f"number of total_edges: {total_edges}")
print(f"number of repeated_edges: {repeated_edges}")
print(f"repeat_ratio: {duplicate_ratio:.4f}")
