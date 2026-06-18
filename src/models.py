"""
Data Models

Typed dataclasses for the candidate ranking pipeline.
These are opt-in — existing dict-based code continues to work,
but new code should prefer these for type safety and autocomplete.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Profile:
    """Candidate profile information."""
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""


@dataclass
class Skill:
    """A single skill entry."""
    name: str = ""
    proficiency: str = "beginner"
    endorsements: int = 0
    duration_months: int = 0


@dataclass
class CareerEntry:
    """A single career history entry."""
    company: str = ""
    title: str = ""
    description: str = ""
    industry: str = ""
    company_size: str = ""
    duration_months: int = 0
    is_current: bool = False


@dataclass
class Education:
    """A single education entry."""
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    tier: str = "unknown"


@dataclass
class SalaryRange:
    """Expected salary range."""
    min: float = 0.0
    max: float = 0.0


@dataclass
class RedrobSignals:
    """Platform behavioral signals."""
    profile_completeness_score: int = 0
    open_to_work_flag: bool = False
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 999.0
    github_activity_score: int = -1
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    notice_period_days: int = 90
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    willing_to_relocate: bool = False
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    last_active_date: str = "2020-01-01"
    signup_date: str = "2020-01-01"
    preferred_work_mode: str = "onsite"
    connection_count: int = 0
    endorsements_received: int = 0
    applications_submitted_30d: int = 0
    profile_views_received_30d: int = 0
    skill_assessment_scores: Dict[str, float] = field(default_factory=dict)
    expected_salary_range_inr_lpa: Dict[str, float] = field(
        default_factory=lambda: {"min": 0.0, "max": 0.0}
    )
    profile_completeness_pct: float = 0.0


@dataclass
class Candidate:
    """A fully normalized candidate record."""
    candidate_id: str = "UNKNOWN"
    profile: Profile = field(default_factory=Profile)
    skills: List[Skill] = field(default_factory=list)
    career_history: List[CareerEntry] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, Any]] = field(default_factory=list)
    redrob_signals: RedrobSignals = field(default_factory=RedrobSignals)


@dataclass
class ScoredCandidate:
    """A candidate with all scoring dimensions computed."""
    candidate_id: str
    candidate: Dict[str, Any]
    semantic_score: float = 0.0
    structured_score: float = 0.0
    behavioral_score: float = 0.0
    bonus_score: float = 0.0
    final_score: float = 0.0


@dataclass
class RankedEntry:
    """A fully ranked candidate with rank position and reasoning."""
    candidate_id: str
    rank: int
    candidate: Dict[str, Any]
    semantic_score: float = 0.0
    structured_score: float = 0.0
    behavioral_score: float = 0.0
    bonus_score: float = 0.0
    final_score: float = 0.0
    reasoning: str = ""


@dataclass
class JDRequirements:
    """Parsed job description requirements."""
    title: str = ""
    experience_range: tuple = (0, 99)
    ideal_experience: float = 5.0
    location_preferences: List[str] = field(default_factory=list)
    work_mode: str = "hybrid"
    max_notice_period_days: int = 30
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    preferred_industries: List[str] = field(default_factory=list)
    consulting_companies: List[str] = field(default_factory=list)
    strong_fit_titles: List[str] = field(default_factory=list)
    weak_fit_titles: List[str] = field(default_factory=list)
