"""Tests for reasoning_generator module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reasoning_generator import (
    generate_reasoning,
    generate_all_reasoning,
    _get_rank_tier,
    _get_matched_skills,
    _get_concerns,
)
from config.settings import REASONING_MAX_LENGTH


def _make_entry(rank=1, title="AI Engineer", company="TechCo", years=7.0,
                skills=None, signals=None):
    if skills is None:
        skills = [
            {"name": "python", "proficiency": "expert", "duration_months": 48, "endorsements": 10},
            {"name": "machine learning", "proficiency": "advanced", "duration_months": 36, "endorsements": 8},
        ]
    if signals is None:
        signals = {
            "recruiter_response_rate": 0.8,
            "open_to_work_flag": True,
            "github_activity_score": 60,
            "notice_period_days": 15,
        }
    return {
        "rank": rank,
        "candidate_id": "TEST001",
        "final_score": 0.85,
        "semantic_score": 0.9,
        "structured_score": 0.8,
        "behavioral_score": 0.7,
        "bonus_score": 0.5,
        "candidate": {
            "candidate_id": "TEST001",
            "profile": {
                "current_title": title,
                "current_company": company,
                "current_industry": "Technology",
                "years_of_experience": years,
                "location": "Pune",
                "country": "India",
                "headline": "Experienced AI Engineer",
                "summary": "",
            },
            "skills": skills,
            "career_history": [
                {"title": "AI Engineer", "company": "TechCo", "industry": "Technology",
                 "duration_months": 36, "is_current": True},
                {"title": "Data Scientist", "company": "OtherCo", "industry": "Technology",
                 "duration_months": 24, "is_current": False},
            ],
            "redrob_signals": signals,
        },
    }


# ── Rank tier ──

class TestRankTier:
    def test_top_tier(self):
        assert _get_rank_tier(1, 100) == "top"
        assert _get_rank_tier(15, 100) == "top"

    def test_mid_tier(self):
        assert _get_rank_tier(30, 100) == "mid"

    def test_low_tier(self):
        assert _get_rank_tier(80, 100) == "low"


# ── Matched skills ──

class TestMatchedSkills:
    def test_finds_required_skills(self):
        entry = _make_entry()
        req, pref = _get_matched_skills(entry["candidate"])
        assert any("python" in s.lower() for s in req)

    def test_no_skills(self):
        entry = _make_entry(skills=[])
        req, pref = _get_matched_skills(entry["candidate"])
        assert len(req) == 0
        assert len(pref) == 0


# ── Concerns ──

class TestConcerns:
    def test_weak_title_concern(self):
        entry = _make_entry(title="HR Manager")
        concerns = _get_concerns(entry["candidate"], 1, 50)
        assert any("not aligned" in c for c in concerns)

    def test_underexperienced_concern(self):
        entry = _make_entry(years=2.0)
        concerns = _get_concerns(entry["candidate"], 1, 50)
        assert any("underexperienced" in c for c in concerns)

    def test_overexperienced_concern(self):
        entry = _make_entry(years=15.0)
        concerns = _get_concerns(entry["candidate"], 1, 50)
        assert any("overexperienced" in c for c in concerns)

    def test_no_concerns_for_ideal(self):
        entry = _make_entry(title="AI Engineer", years=7.0)
        concerns = _get_concerns(entry["candidate"], 1, 50)
        # Ideal candidate should have no major concerns
        assert not any("underexperienced" in c for c in concerns)


# ── Full reasoning generation ──

class TestGenerateReasoning:
    def test_generates_string(self):
        entry = _make_entry()
        reasoning = generate_reasoning(entry, total_ranked=50)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_top_tier_starter(self):
        entry = _make_entry(rank=1)
        reasoning = generate_reasoning(entry, total_ranked=100)
        top_starters = ["Strong fit:", "Excellent match:", "Top candidate:", "Highly relevant:"]
        assert any(reasoning.startswith(s) for s in top_starters)

    def test_low_tier_starter(self):
        entry = _make_entry(rank=90)
        reasoning = generate_reasoning(entry, total_ranked=100)
        low_starters = ["Weak fit:", "Limited alignment:", "Peripheral match:", "Below threshold:"]
        assert any(reasoning.startswith(s) for s in low_starters)

    def test_respects_max_length(self):
        entry = _make_entry()
        reasoning = generate_reasoning(entry, total_ranked=50)
        assert len(reasoning) <= REASONING_MAX_LENGTH

    def test_honeypot_mention(self):
        entry = _make_entry()
        hp_results = {
            "TEST001": {
                "is_honeypot": True,
                "confidence": 0.8,
                "flags": ["career duration mismatch"],
            }
        }
        reasoning = generate_reasoning(entry, total_ranked=50, honeypot_results=hp_results)
        assert "inconsistencies" in reasoning


class TestGenerateAllReasoning:
    def test_populates_all_entries(self):
        entries = [_make_entry(rank=i) for i in range(1, 4)]
        for i, e in enumerate(entries):
            e["candidate"]["candidate_id"] = f"C{i}"
            e["candidate_id"] = f"C{i}"

        result = generate_all_reasoning(entries)
        assert all("reasoning" in e for e in result)
        assert all(len(e["reasoning"]) > 0 for e in result)
