"""Tests for behavioral_scorer module."""
import pytest
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavioral_scorer import (
    score_candidate,
    score_all_candidates,
    _score_response_rate,
    _score_interview_rate,
    _score_completeness,
    _score_recency,
    _score_github,
    _score_offer_acceptance,
    _score_response_time,
    _score_notice_period,
    _score_search_appearance,
)


def _make_signals(**overrides):
    signals = {
        "recruiter_response_rate": 0.5,
        "interview_completion_rate": 0.7,
        "profile_completeness_score": 80,
        "last_active_date": date.today().isoformat(),
        "github_activity_score": 50,
        "offer_acceptance_rate": 0.8,
        "avg_response_time_hours": 12,
        "notice_period_days": 30,
        "search_appearance_30d": 10,
    }
    signals.update(overrides)
    return signals


# ── Individual signal scorers ──

class TestResponseRate:
    def test_high_rate(self):
        assert _score_response_rate({"recruiter_response_rate": 0.9}) == 0.9

    def test_zero_rate(self):
        assert _score_response_rate({"recruiter_response_rate": 0.0}) == 0.0

    def test_clamped_above_one(self):
        assert _score_response_rate({"recruiter_response_rate": 1.5}) == 1.0


class TestRecency:
    def test_active_today(self):
        score = _score_recency({"last_active_date": date.today().isoformat()})
        assert score == 1.0

    def test_active_last_week(self):
        d = (date.today() - timedelta(days=5)).isoformat()
        score = _score_recency({"last_active_date": d})
        assert score == 1.0

    def test_active_last_month(self):
        d = (date.today() - timedelta(days=20)).isoformat()
        score = _score_recency({"last_active_date": d})
        assert score == 0.9

    def test_stale_profile(self):
        score = _score_recency({"last_active_date": "2020-01-01"})
        assert score == 0.1

    def test_invalid_date(self):
        score = _score_recency({"last_active_date": "not-a-date"})
        assert score == 0.1


class TestGithub:
    def test_no_github(self):
        assert _score_github({"github_activity_score": -1}) == 0.3

    def test_active_github(self):
        assert _score_github({"github_activity_score": 75}) == 0.75


class TestResponseTime:
    def test_fast_response(self):
        assert _score_response_time({"avg_response_time_hours": 1}) == 1.0

    def test_slow_response(self):
        assert _score_response_time({"avg_response_time_hours": 100}) == 0.1


class TestNoticePeriod:
    def test_short_notice(self):
        assert _score_notice_period({"notice_period_days": 15}) == 1.0

    def test_long_notice(self):
        assert _score_notice_period({"notice_period_days": 120}) == 0.2


class TestSearchAppearance:
    def test_zero_appearances(self):
        assert _score_search_appearance({"search_appearance_30d": 0}) == 0.1

    def test_many_appearances(self):
        assert _score_search_appearance({"search_appearance_30d": 50}) == 1.0


# ── Full behavioral scoring ──

class TestFullBehavioralScoring:
    def test_returns_all_keys(self):
        c = {
            "candidate_id": "TEST001",
            "redrob_signals": _make_signals(),
        }
        scores = score_candidate(c)
        assert "total" in scores
        assert 0 <= scores["total"] <= 1.0

    def test_score_all_candidates(self):
        candidates = [
            {"candidate_id": "A", "redrob_signals": _make_signals(recruiter_response_rate=0.9)},
            {"candidate_id": "B", "redrob_signals": _make_signals(recruiter_response_rate=0.1)},
        ]
        results = score_all_candidates(candidates)
        assert "A" in results
        assert "B" in results
        assert results["A"] > results["B"]
