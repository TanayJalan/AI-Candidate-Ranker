"""Tests for candidate_loader module."""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidate_loader import _normalize_candidate, load_candidates


# ── Normalization ──

class TestNormalizeCandidate:
    def test_defaults_applied(self):
        raw = {"candidate_id": "C001"}
        c = _normalize_candidate(raw)
        assert c["candidate_id"] == "C001"
        assert c["profile"]["current_title"] == ""
        assert c["profile"]["years_of_experience"] == 0.0
        assert c["skills"] == []
        assert c["career_history"] == []
        assert c["education"] == []

    def test_signal_defaults(self):
        raw = {"candidate_id": "C001"}
        c = _normalize_candidate(raw)
        signals = c["redrob_signals"]
        assert signals["notice_period_days"] == 90
        assert signals["open_to_work_flag"] is False
        assert signals["recruiter_response_rate"] == 0.0

    def test_preserves_existing_values(self):
        raw = {
            "candidate_id": "C001",
            "profile": {"current_title": "Engineer", "years_of_experience": 5.0},
            "skills": [{"name": "Python"}],
        }
        c = _normalize_candidate(raw)
        assert c["profile"]["current_title"] == "Engineer"
        assert c["profile"]["years_of_experience"] == 5.0
        assert c["skills"][0]["name"] == "Python"

    def test_skill_defaults(self):
        raw = {
            "candidate_id": "C001",
            "skills": [{"name": "Python"}],
        }
        c = _normalize_candidate(raw)
        assert c["skills"][0]["proficiency"] == "beginner"
        assert c["skills"][0]["endorsements"] == 0
        assert c["skills"][0]["duration_months"] == 0

    def test_career_defaults(self):
        raw = {
            "candidate_id": "C001",
            "career_history": [{"title": "Dev"}],
        }
        c = _normalize_candidate(raw)
        assert c["career_history"][0]["company"] == ""
        assert c["career_history"][0]["is_current"] is False
        assert c["career_history"][0]["duration_months"] == 0

    def test_education_defaults(self):
        raw = {
            "candidate_id": "C001",
            "education": [{"institution": "MIT"}],
        }
        c = _normalize_candidate(raw)
        assert c["education"][0]["tier"] == "unknown"
        assert c["education"][0]["degree"] == ""

    def test_empty_candidate(self):
        raw = {}
        c = _normalize_candidate(raw)
        assert c["candidate_id"] == "UNKNOWN"


# ── File loading ──

class TestLoadCandidates:
    def test_load_json(self, tmp_path):
        data = [
            {"candidate_id": "A", "profile": {"current_title": "Dev"}},
            {"candidate_id": "B", "profile": {"current_title": "PM"}},
        ]
        f = tmp_path / "candidates.json"
        f.write_text(json.dumps(data))
        candidates = load_candidates(f)
        assert len(candidates) == 2
        assert candidates[0]["candidate_id"] == "A"

    def test_load_jsonl(self, tmp_path):
        lines = [
            json.dumps({"candidate_id": "A"}),
            json.dumps({"candidate_id": "B"}),
        ]
        f = tmp_path / "candidates.jsonl"
        f.write_text("\n".join(lines))
        candidates = load_candidates(f)
        assert len(candidates) == 2

    def test_invalid_json_format(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="Expected JSON array"):
            load_candidates(f)
