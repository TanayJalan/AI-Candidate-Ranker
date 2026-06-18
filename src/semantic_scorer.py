"""
Encodes job description and candidate texts using sentence-transformers,
then computes cosine similarity via FAISS for fast retrieval.
Optionally re-ranks top candidates with a cross-encoder for higher accuracy.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config.settings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_CACHE,
    PROCESSED_DIR,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_TOP_K,
    CROSS_ENCODER_WEIGHT,
)


class SemanticScorer:
    """
    Compute semantic similarity between a job description and candidate texts
    using sentence-transformers + FAISS, with optional cross-encoder re-ranking.
    """

    def __init__(self, model_name: Optional[str] = None, use_cross_encoder: bool = True):
        """
        Initialize the semantic scorer.

        Args:
            model_name: Name of the sentence-transformers model to use.
            use_cross_encoder: Whether to load and use the cross-encoder for re-ranking.
        """
        if model_name is None:
            model_name = EMBEDDING_MODEL_NAME

        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_embedding_dimension()
        assert dimension is not None, "Embedding dimension could not be determined"
        self.dimension: int = dimension
        print(f"  Embedding dimension: {self.dimension}")

        # Lazy-load cross-encoder
        self._cross_encoder: Optional[CrossEncoder] = None
        self._use_cross_encoder = use_cross_encoder

    def _get_cross_encoder(self) -> CrossEncoder:
        """Load cross-encoder on first use."""
        if self._cross_encoder is None:
            print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL_NAME}")
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return self._cross_encoder

    def score(
        self,
        jd_text: str,
        candidate_texts: List[str],
        candidate_ids: List[str],
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Compute semantic similarity scores between JD and all candidates.

        Uses a two-stage approach:
        1. Bi-encoder (FAISS) for fast retrieval of all candidates
        2. Cross-encoder re-ranking of top-K for higher accuracy

        Args:
            jd_text: The job description text (semantic version).
            candidate_texts: List of enriched candidate texts.
            candidate_ids: List of candidate IDs (same order as texts).
            use_cache: Whether to cache/load embeddings from disk.

        Returns:
            Dict mapping candidate_id → semantic_score (0 to 1).
        """
        assert len(candidate_texts) == len(candidate_ids), \
        
        candidate_embeddings = self._encode_candidates(candidate_texts, use_cache)

        jd_embedding = self.model.encode([jd_text], convert_to_numpy=True)
        faiss.normalize_L2(jd_embedding)

        index = faiss.IndexFlatIP(self.dimension) 
        index.add(candidate_embeddings)

        k = len(candidate_ids)
        similarities, indices = index.search(jd_embedding, k)

        # Build score dict — collect raw bi-encoder similarities
        raw_scores = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(candidate_ids):
                raw_scores[candidate_ids[idx]] = float(sim)

        # Cross-encoder re-ranking for top candidates
        if self._use_cross_encoder and len(candidate_ids) > 1:
            raw_scores = self._rerank_with_cross_encoder(
                jd_text, candidate_texts, candidate_ids, raw_scores
            )

        # Min-max normalize across the candidate pool for better spread
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

    def _rerank_with_cross_encoder(
        self,
        jd_text: str,
        candidate_texts: List[str],
        candidate_ids: List[str],
        bi_encoder_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Re-rank the top candidates using a cross-encoder for higher accuracy.

        The cross-encoder attends to both JD and candidate text jointly,
        producing more accurate similarity scores than the bi-encoder.

        Blended score = (1 - weight) * bi_encoder + weight * cross_encoder
        """
        # Sort by bi-encoder score to get top candidates for re-ranking
        sorted_candidates = sorted(
            bi_encoder_scores.items(), key=lambda x: -x[1]
        )
        top_k = min(CROSS_ENCODER_TOP_K, len(sorted_candidates))
        top_candidates = sorted_candidates[:top_k]

        # Build ID-to-index mapping
        id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

        # Create pairs for cross-encoder
        pairs = []
        rerank_ids = []
        for cid, _ in top_candidates:
            idx = id_to_idx.get(cid)
            if idx is not None:
                pairs.append((jd_text, candidate_texts[idx]))
                rerank_ids.append(cid)

        if not pairs:
            return bi_encoder_scores

        # Score with cross-encoder
        cross_encoder = self._get_cross_encoder()
        print(f"  Cross-encoder re-ranking top {len(pairs)} candidates...")
        cross_scores_raw = cross_encoder.predict(pairs)

        # Normalize cross-encoder scores to [0, 1]
        ce_min = float(min(cross_scores_raw))
        ce_max = float(max(cross_scores_raw))
        ce_spread = ce_max - ce_min

        # Blend scores
        blended = dict(bi_encoder_scores)  # Start with all bi-encoder scores
        weight = CROSS_ENCODER_WEIGHT

        for cid, ce_score in zip(rerank_ids, cross_scores_raw):
            # Normalize cross-encoder score
            if ce_spread > 1e-6:
                ce_norm = (float(ce_score) - ce_min) / ce_spread
            else:
                ce_norm = 0.5

            bi_score = bi_encoder_scores.get(cid, 0.0)
            blended[cid] = (1 - weight) * bi_score + weight * ce_norm

        return blended

    def _encode_candidates(
        self, texts: List[str], use_cache: bool
    ) -> np.ndarray:
        cache_path = EMBEDDINGS_CACHE
        hash_path = cache_path.with_suffix(".hash.json")

        # Compute content hash for cache invalidation
        content_hash = self._compute_content_hash(texts)

        if use_cache and cache_path.exists():
            # Check hash-based cache validity
            if hash_path.exists():
                try:
                    with open(hash_path, "r") as f:
                        cached_meta = json.load(f)
                    if (cached_meta.get("content_hash") == content_hash
                            and cached_meta.get("count") == len(texts)):
                        print(f"  Loading cached embeddings from {cache_path.name} (hash verified)")
                        embeddings = np.load(str(cache_path))
                        if embeddings.shape[0] == len(texts):
                            return embeddings
                except (json.JSONDecodeError, KeyError):
                    pass

            # Fallback: size-only check (legacy cache without hash)
            print(f"  Loading cached embeddings from {cache_path.name}")
            embeddings = np.load(str(cache_path))
            if embeddings.shape[0] == len(texts):
                # Save hash for future validation
                self._save_cache_hash(hash_path, content_hash, len(texts))
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
        self._save_cache_hash(hash_path, content_hash, len(texts))
        print(f"  Cached embeddings to {cache_path.name}")

        return embeddings

    @staticmethod
    def _compute_content_hash(texts: List[str]) -> str:
        """Compute SHA-256 hash of concatenated candidate texts for cache invalidation."""
        hasher = hashlib.sha256()
        for t in texts:
            hasher.update(t.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def _save_cache_hash(hash_path: Path, content_hash: str, count: int):
        """Save cache metadata for future hash validation."""
        with open(hash_path, "w") as f:
            json.dump({"content_hash": content_hash, "count": count}, f)
