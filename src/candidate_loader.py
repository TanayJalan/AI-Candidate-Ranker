"""
Candidate Loader

Loads candidate profiles from JSON/JSONL files, normalizes all fields,
and handles missing data gracefully.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import CANDIDATES_JSON


def load_candidates(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load candidate profiles from a JSON or JSONL file.

    Handles both formats:
    - JSON: Array of candidate objects
    - JSONL: One candidate per line

    Args:
        path: Path to candidates file. Defaults to config path.

    Returns:
        List of normalized candidate dicts.
    """
    if path is None:
        path = CANDIDATES_JSON

    path = Path(path)

    if path.suffix == ".jsonl":
        candidates = _load_jsonl(path)
    else:
        candidates = _load_json(path)

    # Normalize all candidates
    normalized = [_normalize_candidate(c) for c in candidates]

    print(f"Loaded {len(normalized)} candidates from {path.name}")
    return normalized


def _load_json(path: Path) -> List[Dict]:
    """Load from a JSON file containing an array of candidates."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected JSON array, got {type(data).__name__}")


def _load_jsonl(path: Path) -> List[Dict]:
    """Load from a JSONL file (one candidate per line)."""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
    return candidates


def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a candidate dict to ensure all expected fields exist
    with sensible defaults.
    """
    # Ensure top-level keys exist
    candidate.setdefault("candidate_id", "UNKNOWN")
    candidate.setdefault("profile", {})
    candidate.setdefault("career_history", [])
    candidate.setdefault("education", [])
    candidate.setdefault("skills", [])
    candidate.setdefault("certifications", [])
    candidate.setdefault("languages", [])
    candidate.setdefault("redrob_signals", {})

    # Normalize profile fields
    profile = candidate["profile"]
    profile.setdefault("anonymized_name", "")
    profile.setdefault("headline", "")
    profile.setdefault("summary", "")
    profile.setdefault("location", "")
    profile.setdefault("country", "")
    profile.setdefault("years_of_experience", 0.0)
    profile.setdefault("current_title", "")
    profile.setdefault("current_company", "")
    profile.setdefault("current_company_size", "")
    profile.setdefault("current_industry", "")

    # Normalize skills
    for skill in candidate["skills"]:
        skill.setdefault("name", "")
        skill.setdefault("proficiency", "beginner")
        skill.setdefault("endorsements", 0)
        skill.setdefault("duration_months", 0)

    # Normalize career history
    for job in candidate["career_history"]:
        job.setdefault("company", "")
        job.setdefault("title", "")
        job.setdefault("description", "")
        job.setdefault("industry", "")
        job.setdefault("company_size", "")
        job.setdefault("duration_months", 0)
        job.setdefault("is_current", False)

    # Normalize education
    for edu in candidate["education"]:
        edu.setdefault("institution", "")
        edu.setdefault("degree", "")
        edu.setdefault("field_of_study", "")
        edu.setdefault("tier", "unknown")

    # Normalize redrob signals with sensible defaults
    signals = candidate["redrob_signals"]
    signals.setdefault("profile_completeness_score", 0)
    signals.setdefault("open_to_work_flag", False)
    signals.setdefault("recruiter_response_rate", 0.0)
    signals.setdefault("avg_response_time_hours", 999)
    signals.setdefault("github_activity_score", -1)
    signals.setdefault("interview_completion_rate", 0.0)
    signals.setdefault("offer_acceptance_rate", -1)
    signals.setdefault("notice_period_days", 90)
    signals.setdefault("verified_email", False)
    signals.setdefault("verified_phone", False)
    signals.setdefault("linkedin_connected", False)
    signals.setdefault("willing_to_relocate", False)
    signals.setdefault("search_appearance_30d", 0)
    signals.setdefault("saved_by_recruiters_30d", 0)
    signals.setdefault("last_active_date", "2020-01-01")
    signals.setdefault("signup_date", "2020-01-01")
    signals.setdefault("preferred_work_mode", "onsite")
    signals.setdefault("connection_count", 0)
    signals.setdefault("endorsements_received", 0)
    signals.setdefault("applications_submitted_30d", 0)
    signals.setdefault("profile_views_received_30d", 0)
    signals.setdefault("skill_assessment_scores", {})
    signals.setdefault("expected_salary_range_inr_lpa", {"min": 0, "max": 0})

    return candidate


def get_candidate_ids(candidates: List[Dict]) -> List[str]:
    """Extract candidate IDs from a list of candidates."""
    return [c["candidate_id"] for c in candidates]
