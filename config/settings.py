import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

CANDIDATES_JSON = RAW_DIR / "sample_candidates.json"
JOB_DESCRIPTION_DOCX = RAW_DIR / "job_description.docx"
CANDIDATE_SCHEMA = RAW_DIR / "candidate_schema.json"

SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
EMBEDDINGS_CACHE = PROCESSED_DIR / "candidate_embeddings.npy"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 64

# Cross-encoder re-ranking (two-stage scoring)
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CROSS_ENCODER_TOP_K = 50       # Re-rank top N candidates from bi-encoder
CROSS_ENCODER_WEIGHT = 0.6     # Blend: (1-w)*bi_encoder + w*cross_encoder

WEIGHT_SEMANTIC = 0.25
WEIGHT_STRUCTURED = 0.40
WEIGHT_BEHAVIORAL = 0.25
WEIGHT_BONUS = 0.10

assert abs(WEIGHT_SEMANTIC + WEIGHT_STRUCTURED + WEIGHT_BEHAVIORAL + WEIGHT_BONUS - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"

STRUCTURED_WEIGHTS = {
    "title_match": 0.35,
    "skills_match": 0.30,
    "experience_fit": 0.15,
    "industry_relevance": 0.12,
    "education_tier": 0.08,
}

BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.15,
    "profile_completeness": 0.15,
    "recency": 0.15,
    "github_activity": 0.10,
    "offer_acceptance_rate": 0.10,
    "response_time": 0.05,
    "notice_period": 0.05,
    "search_appearance": 0.05,
}

JD_REQUIREMENTS = {
    "title": "Senior AI Engineer",
    "experience_range": (5, 9),
    "ideal_experience": 7.0,
    "location_preferences": [
        "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
        "delhi ncr", "india"
    ],
    "work_mode": "hybrid",
    "max_notice_period_days": 30,

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

    "preferred_skills": [
        "lora", "qlora", "peft", "xgboost", "learning to rank",
        "distributed systems", "kubernetes", "docker",
        "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "spark", "airflow",
        "data engineering", "mlops"
    ],

    "preferred_industries": [
        "technology", "software", "ai", "artificial intelligence",
        "machine learning", "data science", "saas", "internet",
        "fintech", "e-commerce", "product", "food delivery",
        "transportation", "marketplace"
    ],

    "consulting_companies": [
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis",
        "l&t infotech", "mindtree"
    ],

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

    "weak_fit_titles": [
        "hr manager", "marketing manager", "content writer",
        "graphic designer", "accountant", "civil engineer",
        "mechanical engineer", "sales executive", "customer support",
        "operations manager", "project manager"
    ],
}

BONUS_OPEN_TO_WORK = 0.30
BONUS_VERIFIED = 0.10
BONUS_RELOCATE = 0.15
BONUS_LINKEDIN = 0.05
BONUS_LOW_NOTICE = 0.20
BONUS_SAVED_BY_RECRUITERS = 0.20

TOP_K = 100
REASONING_MAX_LENGTH = 300
