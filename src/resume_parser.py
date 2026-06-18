"""
Resume Parser

Handles extraction of text from uploaded single PDF or DOCX resumes.
Builds a mock Candidate JSON object using free heuristic regex extraction
to plug into the existing scoring pipeline without needing a paid LLM.
"""

import io
import re
from typing import Dict, Any, List
import pypdf
import mammoth


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using mammoth."""
    result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
    return result.value


def heuristic_parse_resume(raw_text: str, jd_reqs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a mock Candidate JSON object using a free heuristic approach.
    Extracts years of experience and checks for JD keywords.
    """
    # 1. Base default candidate
    candidate: Dict[str, Any] = {
        "candidate_id": "UPLOADED_RESUME",
        "profile": {
            "current_title": "Applicant",
            "current_company": "Unknown",
            "current_industry": "Unknown",
            "years_of_experience": 0.0,
            "location": "Unknown",
            "country": "Unknown",
            "headline": "Uploaded Resume Candidate",
            "summary": raw_text,  # Storing the raw text in the summary so semantic_scorer picks it up
            "github_url": "",
        },
        "skills": [],
        "career_history": [],
        "education": [],
        "certifications": [],
        "redrob_signals": {},
    }

    # 2. Extract GitHub URL
    github_match = re.search(r'(?:https?://)?(?:www\.)?github\.com/[^\s/]+', raw_text, re.IGNORECASE)
    if github_match:
        candidate["profile"]["github_url"] = github_match.group(0)

    # 3. Heuristic: Estimate years of experience
    # e.g., "5+ years of experience", "10 years experience"
    yoe_matches = re.findall(r'(\d+)(?:\+)?\s*(?:-\s*\d+)?\s*years?(?:\s*of)?\s*experience', raw_text, re.IGNORECASE)
    if yoe_matches:
        try:
            # Take the max value found to be safe, or just the first
            max_yoe = max(float(m) for m in yoe_matches)
            candidate["profile"]["years_of_experience"] = max_yoe
        except ValueError:
            pass

    # 4. Heuristic: Extract Skills based on the JD
    # We scan the raw text for required and preferred skills from the JD.
    # If found, we add them to the candidate's skills list so structured_scorer gives them credit.
    required_skills = jd_reqs.get("required_skills", [])
    preferred_skills = jd_reqs.get("preferred_skills", [])
    all_jd_skills = required_skills + preferred_skills

    found_skills = []
    text_lower = raw_text.lower()
    
    for skill in all_jd_skills:
        # Simple word boundary regex to find skill mentions
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append({
                "name": skill,
                "proficiency": "advanced",  # Guess advanced if it's explicitly mentioned
                "duration_months": int(candidate["profile"]["years_of_experience"] * 12),
                "endorsements": 0
            })

    candidate["skills"] = found_skills

    return candidate
