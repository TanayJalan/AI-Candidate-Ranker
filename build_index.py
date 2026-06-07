import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from tqdm import tqdm

def main():
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
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
    
    # Encode texts in batches
    print(f"Encoding {len(texts)} candidate texts...")
    batch_size = 256
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(emb)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity (inner product)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index (IndexFlatIP for inner product = cosine similarity after normalization)
    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index size: {index.ntotal}")
    
    # Save index, ids, and embeddings
    print("Saving index and metadata...")
    faiss.write_index(index, index_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(candidate_ids, f)
    np.save(embeddings_path, embeddings)
    
    print(f"Saved index to {index_path}")
    print(f"Saved candidate IDs to {ids_path}")
    print(f"Saved embeddings to {embeddings_path}")
    print("Done!")

if __name__ == '__main__':
    main()
