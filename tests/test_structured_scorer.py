"""Tests for structured_scorer module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.structured_scorer import (
    score_candidate,
    _score_title,
    _score_skills,
    _score_experience,
    _score_industry,
    _score_education,
    _title_match,
    _skill_match,
    _detect_keyword_stuffing,
)
from config.settings import JD_REQUIREMENTS


# ── Helpers ──

def _make_candidate(
    title="Software Engineer",
    company="TechCo",
    industry="Technology",
    years=5.0,
    skills=None,
    career=None,
    education=None,
):
    return {
        "candidate_id": "TEST001",
        "profile": {
            "current_title": title,
            "current_company": company,
            "current_industry": industry,
            "years_of_experience": years,
            "location": "Pune",
            "country": "India",
            "headline": "",
            "summary": "",
            "anonymized_name": "",
            "current_company_size": "",
        },
        "skills": skills or [],
        "career_history": career or [],
        "education": education or [],
        "certifications": [],
        "redrob_signals": {"skill_assessment_scores": {}},
    }


# ── Title matching ──

class TestTitleMatch:
    def test_exact_match(self):
        assert _title_match("ai engineer", "ai engineer") is True

    def test_case_insensitive(self):
        assert _title_match("AI Engineer", "ai engineer") is True

    def test_subset_match(self):
        assert _title_match("senior ai engineer", "ai engineer") is True

    def test_no_match(self):
        assert _title_match("hr manager", "ai engineer") is False

    def test_empty_strings(self):
        assert _title_match("", "ai engineer") is False
        assert _title_match("ai engineer", "") is False


# ── Skill matching ──

class TestSkillMatch:
    def test_exact_match(self):
        assert _skill_match("python", "python") is True

    def test_short_skill_exact_only(self):
        assert _skill_match("ml", "ml") is True
        assert _skill_match("ml", "nlp") is False

    def test_substring_match(self):
        # "machine learning" IS a substring of "machine learning engineer" (len >= 5)
        assert _skill_match("machine learning", "machine learning engineer") is True
        # Short strings should NOT substring match
        assert _skill_match("ml", "html") is False
        assert _skill_match("pytorch", "pytorch") is True

    def test_fuzzy_match(self):
        assert _skill_match("natural language processing", "natural language process") is True


# ── Experience scoring ──

class TestExperienceScoring:
    def test_ideal_experience(self):
        req = {"experience_range": (5, 9), "ideal_experience": 7.0}
        c = _make_candidate(years=7.0)
        score = _score_experience(c, req)
        assert score > 0.95

    def test_within_range_edges(self):
        req = {"experience_range": (5, 9), "ideal_experience": 7.0}
        assert 0.7 < _score_experience(_make_candidate(years=5.0), req) < 0.95
        assert 0.7 < _score_experience(_make_candidate(years=9.0), req) < 0.95

    def test_under_experienced(self):
        req = {"experience_range": (5, 9), "ideal_experience": 7.0}
        score = _score_experience(_make_candidate(years=1.0), req)
        assert score < 0.4

    def test_over_experienced(self):
        req = {"experience_range": (5, 9), "ideal_experience": 7.0}
        score = _score_experience(_make_candidate(years=20.0), req)
        assert score < 0.5

    def test_zero_experience(self):
        req = {"experience_range": (5, 9), "ideal_experience": 7.0}
        score = _score_experience(_make_candidate(years=0), req)
        assert score == 0.1


# ── Industry scoring ──

class TestIndustryScoring:
    def test_tech_industry_high_score(self):
        c = _make_candidate(industry="Technology")
        score = _score_industry(c, JD_REQUIREMENTS)
        assert score >= 0.8

    def test_consulting_only_low_score(self):
        c = _make_candidate(
            company="Infosys",
            industry="Consulting",
            career=[
                {"company": "Infosys", "industry": "Consulting", "title": "Dev", "duration_months": 36},
                {"company": "TCS", "industry": "Consulting", "title": "Dev", "duration_months": 24},
            ],
        )
        score = _score_industry(c, JD_REQUIREMENTS)
        assert score <= 0.2


# ── Education scoring ──

class TestEducationScoring:
    def test_tier1_cs_max_score(self):
        c = _make_candidate(education=[
            {"institution": "IIT Delhi", "degree": "BTech", "field_of_study": "Computer Science", "tier": "tier_1"}
        ])
        score = _score_education(c, JD_REQUIREMENTS)
        assert score == 1.0

    def test_no_education(self):
        c = _make_candidate(education=[])
        score = _score_education(c, JD_REQUIREMENTS)
        assert score == 0.3

    def test_irrelevant_field(self):
        c = _make_candidate(education=[
            {"institution": "Unknown", "degree": "BA", "field_of_study": "History", "tier": "tier_4"}
        ])
        score = _score_education(c, JD_REQUIREMENTS)
        assert score <= 0.3


# ── Keyword stuffing ──

class TestKeywordStuffing:
    def test_no_stuffing(self):
        skills = [
            {"name": "python", "proficiency": "expert", "duration_months": 48, "endorsements": 10},
            {"name": "cooking", "proficiency": "expert", "duration_months": 60, "endorsements": 5},
        ]
        penalty = _detect_keyword_stuffing(_make_candidate(skills=skills), JD_REQUIREMENTS)
        assert penalty == 1.0  # No penalty

    def test_heavy_stuffing(self):
        # All skills are JD keywords but beginner with 0 months
        stuffed_skills = [
            {"name": skill, "proficiency": "beginner", "duration_months": 0, "endorsements": 0}
            for skill in list(JD_REQUIREMENTS["required_skills"])[:10]
        ]
        penalty = _detect_keyword_stuffing(_make_candidate(skills=stuffed_skills), JD_REQUIREMENTS)
        assert penalty < 0.5  # Should be heavily penalized

    def test_empty_skills(self):
        penalty = _detect_keyword_stuffing(_make_candidate(skills=[]), JD_REQUIREMENTS)
        assert penalty == 1.0


# ── Full scoring ──

class TestFullScoring:
    def test_score_candidate_returns_all_keys(self):
        c = _make_candidate(
            title="AI Engineer",
            years=7.0,
            skills=[{"name": "python", "proficiency": "expert", "duration_months": 48, "endorsements": 10}],
        )
        scores = score_candidate(c)
        assert "total" in scores
        assert "title_match" in scores
        assert "skills_match" in scores
        assert "experience_fit" in scores
        assert "industry_relevance" in scores
        assert "education_tier" in scores
        assert "keyword_stuffing_penalty" in scores
        assert 0 <= scores["total"] <= 1.0

    def test_weak_title_penalty(self):
        good = score_candidate(_make_candidate(title="AI Engineer", years=7.0))
        bad = score_candidate(_make_candidate(title="HR Manager", years=7.0))
        assert good["total"] > bad["total"]
