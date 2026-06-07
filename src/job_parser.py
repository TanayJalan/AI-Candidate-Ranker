"""
Job Description Parser

Parses the job description DOCX file and extracts structured requirements.
Uses mammoth for DOCX→text conversion and pattern matching for extraction.
"""

import mammoth
from pathlib import Path
from typing import Dict, Any

from config.settings import JOB_DESCRIPTION_DOCX, JD_REQUIREMENTS


def parse_job_description(docx_path: Path = None) -> Dict[str, Any]:
    """
    Parse job description from DOCX file and return structured requirements.

    Since the JD is pre-analyzed and its requirements are captured in
    config/settings.py, this function primarily loads the raw text for
    semantic comparison and returns the pre-extracted requirements.

    Args:
        docx_path: Path to the DOCX file. Defaults to config path.

    Returns:
        Dict containing:
            - raw_text: Full plain text of the JD
            - requirements: Structured requirements from config
            - semantic_text: Condensed text optimized for embedding
    """
    if docx_path is None:
        docx_path = JOB_DESCRIPTION_DOCX

    # Extract raw text from DOCX
    with open(docx_path, "rb") as f:
        result = mammoth.extract_raw_text(f)
        raw_text = result.value

    # Build a condensed semantic text that captures the essence of the role
    # This is what we'll embed for semantic comparison with candidates
    semantic_text = _build_semantic_text(raw_text)

    return {
        "raw_text": raw_text,
        "requirements": JD_REQUIREMENTS,
        "semantic_text": semantic_text,
    }


def _build_semantic_text(raw_text: str) -> str:
    """
    Build a condensed text representation of the JD optimized for embedding.

    Instead of embedding the entire verbose JD (which includes hiring philosophy,
    anti-patterns, etc.), we construct a focused text that captures what the
    role actually needs — so the embedding similarity is meaningful.
    """
    # Construct a focused representation of what the ideal candidate looks like
    parts = [
        "Senior AI Engineer at an AI-native talent intelligence platform.",
        "Building ranking, retrieval, and matching systems for recruiters.",
        "Requires 5-9 years experience, ideally 6-8 years in applied ML/AI at product companies.",

        # Core technical requirements
        "Must have production experience with embeddings-based retrieval systems, "
        "sentence-transformers, vector databases like FAISS, Pinecone, or Weaviate.",
        "Strong Python, deep learning, NLP, information retrieval.",
        "Experience designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing.",
        "Hybrid retrieval, LLM-based re-ranking, recommendation systems.",

        # What the role does
        "Ship ranking systems that improve recruiter engagement metrics.",
        "Set up evaluation infrastructure with offline benchmarks and online A/B testing.",
        "Own the intelligence layer: ranking, retrieval, and matching.",
        "Work closely with product managers on what to build.",

        # Cultural signals
        "Startup mentality, scrappy product-engineering attitude.",
        "Ship fast, iterate, learn from real users.",
        "Located in Pune or Noida, India. Hybrid work mode.",

        # Nice-to-haves
        "LLM fine-tuning experience with LoRA, QLoRA, PEFT.",
        "Learning-to-rank models, distributed systems, MLOps.",
        "Open-source contributions in AI/ML space.",
    ]

    return " ".join(parts)


def get_jd_summary() -> str:
    """Return a one-line summary of the job for logging/display."""
    req = JD_REQUIREMENTS
    return (
        f"{req['title']} | {req['experience_range'][0]}-{req['experience_range'][1]} yrs | "
        f"{', '.join(req['location_preferences'][:3]).title()} | {req['work_mode'].title()}"
    )
