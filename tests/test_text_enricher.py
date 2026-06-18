"""Tests for text_enricher and bias_mitigator modules."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text_enricher import enrich_candidate_text, enrich_all_candidates
from src.bias_mitigator import sanitize_text


def _make_candidate(**overrides):
    base = {
        "candidate_id": "TEST001",
        "profile": {
            "anonymized_name": "Candidate A",
            "headline": "Senior ML Engineer",
            "summary": "Experienced in deep learning and NLP.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "TechCo",
            "current_company_size": "1000+",
            "current_industry": "Technology",
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 15, "duration_months": 48},
            {"name": "TensorFlow", "proficiency": "advanced", "endorsements": 3, "duration_months": 8},
        ],
        "career_history": [
            {"title": "ML Engineer", "company": "TechCo", "industry": "Technology",
             "duration_months": 36, "is_current": True, "description": "Built ML pipelines",
             "company_size": ""},
        ],
        "education": [
            {"institution": "IIT Delhi", "degree": "BTech", "field_of_study": "Computer Science", "tier": "tier_1"},
        ],
        "certifications": [{"name": "AWS ML Specialty"}],
        "redrob_signals": {
            "skill_assessment_scores": {"Python": 85, "TensorFlow": 70},
        },
    }
    base.update(overrides)
    return base


# ── Text enrichment ──

class TestEnrichCandidateText:
    def test_includes_title(self):
        text = enrich_candidate_text(_make_candidate())
        assert "ML Engineer" in text

    def test_includes_skills(self):
        text = enrich_candidate_text(_make_candidate())
        assert "Python" in text

    def test_includes_education(self):
        text = enrich_candidate_text(_make_candidate())
        assert "IIT Delhi" in text

    def test_includes_certifications(self):
        text = enrich_candidate_text(_make_candidate())
        assert "AWS ML Specialty" in text

    def test_includes_career_history(self):
        text = enrich_candidate_text(_make_candidate())
        assert "TechCo" in text

    def test_includes_assessments(self):
        text = enrich_candidate_text(_make_candidate())
        assert "85/100" in text

    def test_empty_profile(self):
        c = {
            "candidate_id": "EMPTY",
            "profile": {
                "anonymized_name": "", "headline": "", "summary": "",
                "location": "", "country": "", "years_of_experience": 0,
                "current_title": "", "current_company": "",
                "current_company_size": "", "current_industry": "",
            },
            "skills": [], "career_history": [], "education": [],
            "certifications": [], "redrob_signals": {"skill_assessment_scores": {}},
        }
        text = enrich_candidate_text(c)
        assert isinstance(text, str)


class TestEnrichAllCandidates:
    def test_returns_id_text_pairs(self):
        candidates = [_make_candidate()]
        results = enrich_all_candidates(candidates)
        assert len(results) == 1
        assert results[0][0] == "TEST001"
        assert isinstance(results[0][1], str)


# ── Bias mitigation ──

class TestSanitizeText:
    def test_strips_gendered_pronouns(self):
        text = "He is an experienced engineer. She has strong skills."
        sanitized = sanitize_text(text)
        assert "He " not in sanitized
        assert "She " not in sanitized

    def test_strips_graduation_years(self):
        text = "Graduated 2015 from IIT Delhi. Class of 2012."
        sanitized = sanitize_text(text)
        assert "2015" not in sanitized
        assert "2012" not in sanitized

    def test_strips_age_indicators(self):
        text = "28 years old candidate. Born in 1996."
        sanitized = sanitize_text(text)
        assert "years old" not in sanitized
        assert "Born in 1996" not in sanitized

    def test_preserves_skill_content(self):
        text = "Skills: Python (expert), Machine Learning, Deep Learning"
        sanitized = sanitize_text(text)
        assert "Python" in sanitized
        assert "Machine Learning" in sanitized

    def test_collapses_spaces(self):
        text = "He   is   experienced"
        sanitized = sanitize_text(text)
        assert "   " not in sanitized

    def test_empty_string(self):
        assert sanitize_text("") == ""
