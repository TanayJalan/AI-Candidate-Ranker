"""
Central configuration for the AI Candidate Ranking System.
All paths, weights, thresholds, and model settings live here.
"""

import os
from pathlib import Path

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

CANDIDATES_JSON = RAW_DIR / "sample_candidates.json"
JOB_DESCRIPTION_DOCX = RAW_DIR / "job_description.docx"
CANDIDATE_SCHEMA = RAW_DIR / "candidate_schema.json"

SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
EMBEDDINGS_CACHE = PROCESSED_DIR / "candidate_embeddings.npy"

# ─── Model settings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # 768-dim, best quality general-purpose
EMBEDDING_BATCH_SIZE = 64

# ─── Scoring weights (must sum to 1.0) ───────────────────────────────────────
WEIGHT_SEMANTIC = 0.25
WEIGHT_STRUCTURED = 0.40
WEIGHT_BEHAVIORAL = 0.25
WEIGHT_BONUS = 0.10

assert abs(WEIGHT_SEMANTIC + WEIGHT_STRUCTURED + WEIGHT_BEHAVIORAL + WEIGHT_BONUS - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"

# ─── Structured scorer sub-weights ───────────────────────────────────────────
STRUCTURED_WEIGHTS = {
    "title_match": 0.35,       # How well current title matches JD role — primary trap catcher
    "skills_match": 0.30,      # Skill overlap weighted by proficiency
    "experience_fit": 0.15,    # Years-of-experience fit to JD range
    "industry_relevance": 0.12, # Current/past industry overlap
    "education_tier": 0.08,    # Institution tier bonus
}

# ─── Behavioral scorer sub-weights ───────────────────────────────────────────
BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.15,
    "profile_completeness": 0.15,
    "recency": 0.15,            # How recently active
    "github_activity": 0.10,
    "offer_acceptance_rate": 0.10,
    "response_time": 0.05,      # Lower is better
    "notice_period": 0.05,
    "search_appearance": 0.05,
}

# ─── JD-specific parameters (extracted from job_description.docx) ────────────
# These are the key requirements we extract from the JD
JD_REQUIREMENTS = {
    "title": "Senior AI Engineer",
    "experience_range": (5, 9),       # 5-9 years, ideal 6-8
    "ideal_experience": 7.0,          # Center of ideal range
    "location_preferences": [
        "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
        "delhi ncr", "india"
    ],
    "work_mode": "hybrid",
    "max_notice_period_days": 30,     # Prefer sub-30 day notice

    # Core required skills (must-haves from JD)
    "required_skills": [
        "python", "machine learning", "ml", "deep learning",
        "embeddings", "sentence-transformers", "vector search",
        "faiss", "retrieval", "ranking", "search", "nlp",
        "natural language processing", "information retrieval",
        "recommendation systems", "a/b testing", "evaluation",
        "pytorch", "tensorflow", "transformers", "llm",
        "large language models", "fine-tuning", "rag",
        "retrieval augmented generation"
    ],

    # Nice-to-have skills
    "preferred_skills": [
        "lora", "qlora", "peft", "xgboost", "learning to rank",
        "distributed systems", "kubernetes", "docker",
        "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "spark", "airflow",
        "data engineering", "mlops"
    ],

    # Industries that signal product-company experience (positive)
    "preferred_industries": [
        "technology", "software", "ai", "artificial intelligence",
        "machine learning", "data science", "saas", "internet",
        "fintech", "e-commerce", "product", "food delivery",
        "transportation", "marketplace"
    ],

    # Consulting-heavy backgrounds (negative signal per JD)
    "consulting_companies": [
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis",
        "l&t infotech", "mindtree"
    ],

    # Titles that are a STRONG fit
    "strong_fit_titles": [
        "ai engineer", "ml engineer", "machine learning engineer",
        "data scientist", "senior ai engineer", "senior ml engineer",
        "senior data scientist", "research engineer",
        "applied scientist", "nlp engineer", "search engineer",
        "ranking engineer", "recommendation engineer",
        "recommendation systems engineer", "applied ml engineer",
        "software engineer", "backend engineer", "senior software engineer",
        "data engineer", "deep learning engineer",
    ],

    # Titles that are a WEAK fit (keyword-stuffer trap per JD)
    "weak_fit_titles": [
        "hr manager", "marketing manager", "content writer",
        "graphic designer", "accountant", "civil engineer",
        "mechanical engineer", "sales executive", "customer support",
        "operations manager", "project manager"
    ],
}

# ─── Bonus modifiers ─────────────────────────────────────────────────────────
BONUS_OPEN_TO_WORK = 0.30
BONUS_VERIFIED = 0.10           # verified email + phone
BONUS_RELOCATE = 0.15           # willing to relocate
BONUS_LINKEDIN = 0.05           # linkedin connected
BONUS_LOW_NOTICE = 0.20         # notice period <= 30 days
BONUS_SAVED_BY_RECRUITERS = 0.20  # saved by recruiters in 30d

# ─── Output settings ─────────────────────────────────────────────────────────
TOP_K = 100  # Number of candidates to rank in submission
REASONING_MAX_LENGTH = 300  # Max chars per reasoning string (spec allows generous length)
