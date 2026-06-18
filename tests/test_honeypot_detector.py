"""Tests for honeypot_detector module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.honeypot_detector import (
    detect_honeypot,
    detect_all_honeypots,
    _check_career_duration_mismatch,
    _check_impossible_skill_proficiency,
    _check_excessive_expert_skills,
    _check_endorsement_anomalies,
    _check_career_timeline_gaps,
    _check_assessment_vs_proficiency,
    _check_impossible_experience_for_career,
)


def _make_candidate(**overrides):
    base = {
        "candidate_id": "TEST001",
        "profile": {"years_of_experience": 5.0},
        "skills": [],
        "career_history": [],
        "redrob_signals": {"skill_assessment_scores": {}},
    }
    base.update(overrides)
    return base


# ── Individual checks ──

class TestCareerDurationMismatch:
    def test_no_career(self):
        c = _make_candidate(career_history=[])
        flagged, _ = _check_career_duration_mismatch(c)
        assert flagged is False

    def test_valid_timeline(self):
        c = _make_candidate(
            career_history=[{"duration_months": 60}],
        )
        c["profile"]["years_of_experience"] = 5.0
        flagged, _ = _check_career_duration_mismatch(c)
        assert flagged is False

    def test_impossible_timeline(self):
        c = _make_candidate(
            career_history=[{"duration_months": 24}],
        )
        c["profile"]["years_of_experience"] = 10.0
        flagged, msg = _check_career_duration_mismatch(c)
        assert flagged is True
        assert "mismatch" in msg


class TestImpossibleSkillProficiency:
    def test_no_flags_with_valid_skills(self):
        c = _make_candidate(skills=[
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
            {"name": "ML", "proficiency": "advanced", "duration_months": 24},
        ])
        flagged, _ = _check_impossible_skill_proficiency(c)
        assert flagged is False

    def test_flags_many_expert_with_no_duration(self):
        c = _make_candidate(skills=[
            {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 1}
            for i in range(5)
        ])
        flagged, msg = _check_impossible_skill_proficiency(c)
        assert flagged is True


class TestExcessiveExpertSkills:
    def test_reasonable_experts(self):
        c = _make_candidate(skills=[
            {"name": f"Skill{i}", "proficiency": "expert"} for i in range(3)
        ])
        flagged, _ = _check_excessive_expert_skills(c)
        assert flagged is False

    def test_too_many_experts(self):
        c = _make_candidate(skills=[
            {"name": f"Skill{i}", "proficiency": "expert"} for i in range(10)
        ])
        flagged, msg = _check_excessive_expert_skills(c)
        assert flagged is True
        assert "10" in msg


class TestEndorsementAnomalies:
    def test_normal_endorsements(self):
        c = _make_candidate(skills=[
            {"name": "Python", "endorsements": 10, "duration_months": 36},
        ])
        flagged, _ = _check_endorsement_anomalies(c)
        assert flagged is False

    def test_suspicious_endorsements(self):
        c = _make_candidate(skills=[
            {"name": "Python", "endorsements": 50, "duration_months": 1},
            {"name": "ML", "endorsements": 40, "duration_months": 2},
        ])
        flagged, _ = _check_endorsement_anomalies(c)
        assert flagged is True


class TestAssessmentVsProficiency:
    def test_consistent_assessment(self):
        c = _make_candidate(
            skills=[{"name": "Python", "proficiency": "expert"}],
        )
        c["redrob_signals"]["skill_assessment_scores"] = {"Python": 90}
        flagged, _ = _check_assessment_vs_proficiency(c)
        assert flagged is False

    def test_contradictory_assessment(self):
        c = _make_candidate(
            skills=[
                {"name": "Python", "proficiency": "expert"},
                {"name": "ML", "proficiency": "expert"},
            ],
        )
        c["redrob_signals"]["skill_assessment_scores"] = {"Python": 10, "ML": 15}
        flagged, msg = _check_assessment_vs_proficiency(c)
        assert flagged is True
        assert "contradictions" in msg


class TestImpossibleExperienceForCareer:
    def test_valid(self):
        c = _make_candidate(career_history=[{"title": "Dev"}, {"title": "Sr Dev"}])
        c["profile"]["years_of_experience"] = 6.0
        flagged, _ = _check_impossible_experience_for_career(c)
        assert flagged is False

    def test_impossible(self):
        c = _make_candidate(career_history=[
            {"title": f"Role{i}"} for i in range(5)
        ])
        c["profile"]["years_of_experience"] = 1.0
        flagged, _ = _check_impossible_experience_for_career(c)
        assert flagged is True


# ── Composite detection ──

class TestDetectHoneypot:
    def test_clean_candidate(self):
        c = _make_candidate(
            skills=[{"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 5}],
            career_history=[{"duration_months": 60}],
        )
        result = detect_honeypot(c)
        assert result["is_honeypot"] is False
        assert result["confidence"] == 0.0

    def test_multiple_flags(self):
        c = _make_candidate(
            skills=[
                {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 1, "endorsements": 50}
                for i in range(10)
            ],
            career_history=[{"duration_months": 12}],
        )
        c["profile"]["years_of_experience"] = 10.0
        result = detect_honeypot(c)
        assert result["is_honeypot"] is True
        assert result["confidence"] >= 0.6
        assert len(result["flags"]) >= 2


class TestDetectAllHoneypots:
    def test_batch_detection(self):
        candidates = [
            _make_candidate(),
            _make_candidate(),
        ]
        candidates[0]["candidate_id"] = "A"
        candidates[1]["candidate_id"] = "B"
        results = detect_all_honeypots(candidates)
        assert "A" in results
        assert "B" in results
