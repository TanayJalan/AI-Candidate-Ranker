

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config.settings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_CACHE,
    PROCESSED_DIR,
)


class SemanticScorer:


    def __init__(self, model_name: Optional[str] = None):

        if model_name is None:
            model_name = EMBEDDING_MODEL_NAME

        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_embedding_dimension()
        assert dimension is not None, "Embedding dimension could not be determined"
        self.dimension: int = dimension
        print(f"  Embedding dimension: {self.dimension}")

    def score(
        self,
        jd_text: str,
        candidate_texts: List[str],
        candidate_ids: List[str],
        use_cache: bool = True,
    ) -> Dict[str, float]:
        assert len(candidate_texts) == len(candidate_ids), \
        
        candidate_embeddings = self._encode_candidates(candidate_texts, use_cache)

        jd_embedding = self.model.encode([jd_text], convert_to_numpy=True)
        faiss.normalize_L2(jd_embedding)

        index = faiss.IndexFlatIP(self.dimension) 
        index.add(candidate_embeddings)

        k = len(candidate_ids)
        similarities, indices = index.search(jd_embedding, k)

        raw_scores = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(candidate_ids):
                raw_scores[candidate_ids[idx]] = float(sim)

        if raw_scores:
            min_sim = min(raw_scores.values())
            max_sim = max(raw_scores.values())
            spread = max_sim - min_sim

            if spread > 1e-6:
                scores = {
                    cid: (sim - min_sim) / spread
                    for cid, sim in raw_scores.items()
                }
            else:
                scores = {cid: 0.5 for cid in raw_scores}
        else:
            scores = {}

        return scores

    def _encode_candidates(
        self, texts: List[str], use_cache: bool
    ) -> np.ndarray:
        cache_path = EMBEDDINGS_CACHE

        if use_cache and cache_path.exists():
            print(f"  Loading cached embeddings from {cache_path.name}")
            embeddings = np.load(str(cache_path))
            if embeddings.shape[0] == len(texts):
                return embeddings
            print(f"  Cache size mismatch ({embeddings.shape[0]} vs {len(texts)}), re-encoding")

        print(f"  Encoding {len(texts)} candidate texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        print(f"  Encoding complete. Shape: {embeddings.shape}")

       
        faiss.normalize_L2(embeddings)

      
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        print(f"  Cached embeddings to {cache_path.name}")

        return embeddings
