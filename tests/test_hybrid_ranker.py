"""Tests for hybrid_ranker module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_ranker import compute_bonus, rank_candidates


def _make_candidate(cid="TEST001", **signal_overrides):
    signals = {
        "open_to_work_flag": False,
        "verified_email": False,
        "verified_phone": False,
        "willing_to_relocate": False,
        "linkedin_connected": False,
        "notice_period_days": 90,
        "saved_by_recruiters_30d": 0,
    }
    signals.update(signal_overrides)
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": "Engineer",
            "current_company": "TechCo",
            "years_of_experience": 5.0,
        },
        "redrob_signals": signals,
    }


# ── Bonus computation ──

class TestComputeBonus:
    def test_no_bonus(self):
        c = _make_candidate()
        bonus = compute_bonus(c)
        assert bonus == 0.0

    def test_open_to_work(self):
        c = _make_candidate(open_to_work_flag=True)
        bonus = compute_bonus(c)
        assert bonus > 0.0

    def test_all_bonuses(self):
        c = _make_candidate(
            open_to_work_flag=True,
            verified_email=True,
            verified_phone=True,
            willing_to_relocate=True,
            linkedin_connected=True,
            notice_period_days=15,
            saved_by_recruiters_30d=20,
        )
        bonus = compute_bonus(c)
        assert bonus == 1.0  # Capped at 1.0

    def test_partial_verification(self):
        only_email = compute_bonus(_make_candidate(verified_email=True))
        both = compute_bonus(_make_candidate(verified_email=True, verified_phone=True))
        assert both > only_email


# ── Ranking ──

class TestRankCandidates:
    def test_ranking_order(self):
        candidates = [_make_candidate(cid=f"C{i}") for i in range(3)]
        semantic = {"C0": 0.9, "C1": 0.3, "C2": 0.6}
        structured = {"C0": 0.8, "C1": 0.4, "C2": 0.5}
        behavioral = {"C0": 0.7, "C1": 0.5, "C2": 0.6}

        ranked = rank_candidates(candidates, semantic, structured, behavioral)
        assert ranked[0]["candidate_id"] == "C0"
        assert ranked[0]["rank"] == 1
        assert len(ranked) == 3

    def test_top_k(self):
        candidates = [_make_candidate(cid=f"C{i}") for i in range(5)]
        semantic = {f"C{i}": 0.5 for i in range(5)}
        structured = {f"C{i}": 0.5 for i in range(5)}
        behavioral = {f"C{i}": 0.5 for i in range(5)}

        ranked = rank_candidates(candidates, semantic, structured, behavioral, top_k=2)
        assert len(ranked) == 2

    def test_custom_weights(self):
        candidates = [
            _make_candidate(cid="SEM"),
            _make_candidate(cid="STR"),
        ]
        # SEM has high semantic, STR has high structured
        semantic = {"SEM": 1.0, "STR": 0.0}
        structured = {"SEM": 0.0, "STR": 1.0}
        behavioral = {"SEM": 0.0, "STR": 0.0}

        # With semantic-heavy weights, SEM should win
        sem_weights = {"semantic": 0.9, "structured": 0.05, "behavioral": 0.05, "bonus": 0.0}
        ranked = rank_candidates(candidates, semantic, structured, behavioral, weights=sem_weights)
        assert ranked[0]["candidate_id"] == "SEM"

        # With structured-heavy weights, STR should win
        str_weights = {"semantic": 0.05, "structured": 0.9, "behavioral": 0.05, "bonus": 0.0}
        ranked = rank_candidates(candidates, semantic, structured, behavioral, weights=str_weights)
        assert ranked[0]["candidate_id"] == "STR"

    def test_tie_breaking_by_id(self):
        candidates = [_make_candidate(cid=cid) for cid in ["B", "A", "C"]]
        semantic = {c: 0.5 for c in ["A", "B", "C"]}
        structured = {c: 0.5 for c in ["A", "B", "C"]}
        behavioral = {c: 0.5 for c in ["A", "B", "C"]}

        ranked = rank_candidates(candidates, semantic, structured, behavioral)
        # Alphabetical tiebreak
        assert ranked[0]["candidate_id"] == "A"
        assert ranked[1]["candidate_id"] == "B"
