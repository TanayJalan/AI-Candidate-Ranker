print("Starting simple test...")
try:
    import faiss
    print("FAISS imported")
except Exception as e:
    print(f"FAISS import error: {e}")

try:
    import numpy as np
    print("NumPy imported")
except Exception as e:
    print(f"NumPy import error: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported")
except Exception as e:
    print(f"SentenceTransformer import error: {e}")

print("Simple test completed")
