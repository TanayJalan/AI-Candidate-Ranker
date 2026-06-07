import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time
import gc
from tqdm import tqdm

def main():
    start_time = time.time()
    # Paths
    data_dir = '/Users/tanayjalan/candidate_matcher/data'
    candidates_path = os.path.join(data_dir, 'candidates_processed.parquet')
    index_path = os.path.join(data_dir, 'candidate_index.faiss')
    ids_path = os.path.join(data_dir, 'candidate_ids.pkl')
    
    # Set torch threads to 1 to reduce memory usage
    try:
        import torch
        torch.set_num_threads(1)
        print("Set torch threads to 1")
    except ImportError:
        print("Torch not available, skipping thread setting")
    
    # Load processed candidates - only necessary columns
    print("Loading candidate data...")
    df = pd.read_parquet(candidates_path, columns=['text', 'candidate_id'])
    print(f"Loaded {len(df)} candidates")
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model_load_start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
    model_load_end = time.time()
    print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
    
    # Get embedding dimension by testing with a sample
    sample_emb = model.encode(["sample text"], convert_to_numpy=True)
    dim = sample_emb.shape[1]
    print(f"Embedding dimension: {dim}")
    
    # Build FAISS index (IndexFlatIP for inner product = cosine similarity after normalization)
    print(f"Building FAISS index with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    
    # Encode texts in batches and add to index immediately
    print(f"Encoding {len(df)} candidate texts and building index...")
    encode_start = time.time()
    batch_size = 32  # Very small batch size to conserve memory
    candidate_ids = []
    
    for start in tqdm(range(0, len(df), batch_size), desc="Encoding batches"):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        batch_texts = batch_df['text'].tolist()
        batch_ids = batch_df['candidate_id'].tolist()
        
        # Encode batch
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity (inner product)
        faiss.normalize_L2(emb)
        
        # Add to index
        index.add(emb)
        
        # Store IDs
        candidate_ids.extend(batch_ids)
        
        # Clear batch from memory
        del batch_texts
        del batch_ids
        del emb
        del batch_df
        # Periodic garbage collection
        if start % (batch_size * 50) == 0:
            gc.collect()
    
    encode_end = time.time()
    print(f"Encoding and indexing took {encode_end - encode_start:.2f} seconds")
    print(f"Index size: {index.ntotal}")
    print(f"Number of IDs stored: {len(candidate_ids)}")
    
    # Save index and IDs
    print("Saving index and metadata...")
    save_start = time.time()
    faiss.write_index(index, index_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(candidate_ids, f)
    save_end = time.time()
    print(f"Saving took {save_end - save_start:.2f} seconds")
    
    print(f"Saved index to {index_path}")
    print(f"Saved candidate IDs to {ids_path}")
    
    total_end = time.time()
    print(f"Total time: {total_end - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
