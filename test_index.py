import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

data_dir = '/Users/tanayjalan/candidate_matcher/data'
index_path = os.path.join(data_dir, 'candidate_index.faiss')
ids_path = os.path.join(data_dir, 'candidate_ids.pkl')

print("Loading index...")
index = faiss.read_index(index_path)
print(f"Index loaded, ntotal: {index.ntotal}")

print("Loading IDs...")
with open(ids_path, 'rb') as f:
    candidate_ids = pickle.load(f)
print(f"Loaded {len(candidate_ids)} IDs")

# Test with a random query
model = SentenceTransformer('all-MiniLM-L6-v2')
query_text = "This is a test query about machine learning and AI"
query_emb = model.encode([query_text], convert_to_numpy=True)
faiss.normalize_L2(query_emb)

# Search for top 5
k = 5
distances, indices = index.search(query_emb, k)
print(f"\nTop {k} results for query: '{query_text}'")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if idx < len(candidate_ids):
        print(f"  {i+1}. ID: {candidate_ids[idx]}, Distance: {dist:.4f}")
    else:
        print(f"  {i+1}. Index {idx} out of bounds")

print("\nIndex test completed successfully.")
