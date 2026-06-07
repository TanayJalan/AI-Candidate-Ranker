import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import time
from tqdm import tqdm

def main():
    start_time = time.time()
    # Paths
    data_dir = '/Users/tanayjalan/candidate_matcher/data'
    candidates_path = os.path.join(data_dir, 'candidates_processed.parquet')
    index_path = os.path.join(data_dir, 'candidate_index.faiss')
    ids_path = os.path.join(data_dir, 'candidate_ids.pkl')
    
    # Load processed candidates
    print("Loading candidate data...")
    df = pd.read_parquet(candidates_path)
    texts = df['text'].tolist()
    candidate_ids = df['candidate_id'].tolist()
    
    print(f"Loaded {len(texts)} candidates")
    
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
    print(f"Encoding {len(texts)} candidate texts and building index...")
    encode_start = time.time()
    batch_size = 512  # Adjust based on memory availability
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_end = min(i + batch_size, len(texts))
        batch = texts[i:batch_end]
        
        # Encode batch
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity (inner product)
        faiss.normalize_L2(emb)
        
        # Add to index
        index.add(emb)
        
        # Clear batch from memory
        del batch
        del emb
    
    encode_end = time.time()
    print(f"Encoding and indexing took {encode_end - encode_start:.2f} seconds")
    print(f"Index size: {index.ntotal}")
    
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
