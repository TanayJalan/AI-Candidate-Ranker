import mammoth
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import JOB_DESCRIPTION_DOCX, JD_REQUIREMENTS


def parse_job_description(docx_path: Optional[Path] = None) -> Dict[str, Any]:

    if docx_path is None:
        docx_path = JOB_DESCRIPTION_DOCX

    with open(docx_path, "rb") as f:
        result = mammoth.extract_raw_text(f)
        raw_text = result.value

    semantic_text = _build_semantic_text(raw_text)

    return {
        "raw_text": raw_text,
        "requirements": JD_REQUIREMENTS,
        "semantic_text": semantic_text,
    }


def _build_semantic_text(raw_text: str) -> str:

    parts = [
        "Senior AI Engineer at an AI-native talent intelligence platform.",
        "Building ranking, retrieval, and matching systems for recruiters.",
        "Requires 5-9 years experience, ideally 6-8 years in applied ML/AI at product companies.",

        "Must have production experience with embeddings-based retrieval systems, "
        "sentence-transformers, vector databases like FAISS, Pinecone, or Weaviate.",
        "Strong Python, deep learning, NLP, information retrieval.",
        "Experience designing evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing.",
        "Hybrid retrieval, LLM-based re-ranking, recommendation systems.",

        "Ship ranking systems that improve recruiter engagement metrics.",
        "Set up evaluation infrastructure with offline benchmarks and online A/B testing.",
        "Own the intelligence layer: ranking, retrieval, and matching.",
        "Work closely with product managers on what to build.",

        "Startup mentality, scrappy product-engineering attitude.",
        "Ship fast, iterate, learn from real users.",
        "Located in Pune or Noida, India. Hybrid work mode.",

        "LLM fine-tuning experience with LoRA, QLoRA, PEFT.",
        "Learning-to-rank models, distributed systems, MLOps.",
        "Open-source contributions in AI/ML space.",
    ]

    return " ".join(parts)


def get_jd_summary() -> str:

    req = JD_REQUIREMENTS
    return (
        f"{req['title']} | {req['experience_range'][0]}-{req['experience_range'][1]} yrs | "
        f"{', '.join(req['location_preferences'][:3]).title()} | {req['work_mode'].title()}"
    )
