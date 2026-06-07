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
    embeddings_path = os.path.join(data_dir, 'candidate_embeddings.npy')
    
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
    
    # Encode texts in batches
    print(f"Encoding {len(texts)} candidate texts...")
    encode_start = time.time()
    batch_size = 1024  # Increased batch size for speed
    all_embeddings = []
    
    # Set number of threads for PyTorch to use all available CPUs
    # torch.set_num_threads(os.cpu_count())  # Uncomment if you want to set explicitly
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(emb)
    
    embeddings = np.vstack(all_embeddings)
    encode_end = time.time()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Encoding took {encode_end - encode_start:.2f} seconds")
    
    # Normalize embeddings for cosine similarity (inner product)
    print("Normalizing embeddings...")
    norm_start = time.time()
    faiss.normalize_L2(embeddings)
    norm_end = time.time()
    print(f"Normalization took {norm_end - norm_start:.2f} seconds")
    
    # Build FAISS index (IndexFlatIP for inner product = cosine similarity after normalization)
    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")
    index_start = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_end = time.time()
    print(f"Index built in {index_end - index_start:.2f} seconds")
    print(f"Index size: {index.ntotal}")
    
    # Save index, ids, and embeddings
    print("Saving index and metadata...")
    save_start = time.time()
    faiss.write_index(index, index_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(candidate_ids, f)
    np.save(embeddings_path, embeddings)
    save_end = time.time()
    print(f"Saving took {save_end - save_start:.2f} seconds")
    
    print(f"Saved index to {index_path}")
    print(f"Saved candidate IDs to {ids_path}")
    print(f"Saved embeddings to {embeddings_path}")
    
    total_end = time.time()
    print(f"Total time: {total_end - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
