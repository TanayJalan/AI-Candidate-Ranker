---
title: Candidate Matcher
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
# 🎯 AI Candidate Ranking System

A hybrid AI-powered candidate ranking engine built for the **Redrob Hackathon**. It goes beyond keyword matching to evaluate job fit the way a great recruiter would — through semantic understanding, structured signal analysis, behavioral profiling, and honeypot detection.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      AI CANDIDATE RANKER                            │
├─────────────┬──────────────┬──────────────┬────────────┬────────────┤
│  Semantic   │  Structured  │  Behavioral  │   Bonus    │  Honeypot  │
│   (25%)     │    (40%)     │    (25%)     │   (10%)    │  Detector  │
├─────────────┼──────────────┼──────────────┼────────────┼────────────┤
│ Sentence    │ Title Match  │ Response     │ Open to    │ Career     │
│ Transformers│ Skills Fit   │ Rates        │ Work       │ Duration   │
│ + FAISS     │ Experience   │ GitHub       │ Verified   │ Skill      │
│ Cosine Sim  │ Industry     │ Completeness │ Notice     │ Proficiency│
│             │ Education    │ Recency      │ Location   │ Endorsement│
│             │ Trap Penalty │              │            │ Anomalies  │
└─────────────┴──────────────┴──────────────┴────────────┴────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Hybrid Ranker    │
                    │  (Weighted Sum +  │
                    │  Tie-breaking)    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Reasoning Gen    │
                    │  (Fact-based,     │
                    │  Rank-aware tone) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  submission.csv   │
                    └───────────────────┘
```

## Dataset

The evaluation code was built and tested against the official dataset provided by the hackathon organizers. The raw candidate data schema and samples can be found here: [Organizer Drive Link](https://drive.google.com/file/d/1MfD47XvVdRKBGRAyzGOxDCEf2ve96Jjo/view)

## Quick Start (For Evaluators)

We've included a convenience script that automatically installs dependencies and runs the entire pipeline with verbose output.

```bash
git clone https://github.com/TanayJalan/Catcher.git
cd Catcher
chmod +x run.sh
./run.sh
```

### Manual Execution

If you prefer to run things manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Produce Submission CSV
python rank.py --candidates data/raw/sample_candidates.json --out data/output/submission.csv --verbose
```

### Launch Interactive Dashboard

We built a Streamlit dashboard to satisfy the mandatory sandbox requirement. It visualizes score breakdowns, flags honeypots, and provides a deep dive into candidate reasoning.

```bash
streamlit run app.py
```

### Run Unit Tests

We've included automated tests to prove the robustness of our ranking logic (e.g., Gaussian experience penalties and honeypot detection).

```bash
pytest tests/
```

## Project Structure

```text
candidate_matcher/
├── run.sh                   # Convenience execution wrapper
├── rank.py                  # Main entry point (7-step pipeline)
├── app.py                   # Streamlit interactive dashboard
├── config/
│   └── settings.py          # All weights, thresholds, JD requirements
├── src/
│   ├── job_parser.py        # Parse job description DOCX
│   ├── candidate_loader.py  # Load & normalize candidate JSON
│   ├── text_enricher.py     # Build rich text for embeddings
│   ├── semantic_scorer.py   # Sentence-transformers + FAISS
│   ├── structured_scorer.py # Rule-based scoring (title, skills, etc.)
│   ├── behavioral_scorer.py # RedRob platform signal scoring
│   ├── hybrid_ranker.py     # Weighted combination + ranking
│   ├── reasoning_generator.py # Fact-based reasoning strings
│   └── honeypot_detector.py # Impossible profile detection
├── tests/
│   └── test_pipeline.py     # Unit tests for scoring logic
├── data/
│   └── raw/                 # Input files (candidates, JD, schema)
├── pitch.md                 # 2-minute defense of the architecture
├── requirements.txt         
├── submission_metadata.yaml # Hackathon metadata
└── README.md
```

## How It Works

### 1. Semantic Scoring (25%)
Uses `all-mpnet-base-v2` sentence-transformer to encode both the JD and candidate profiles into dense vectors. FAISS computes cosine similarity with min-max normalization for proper differentiation.

### 2. Structured Scoring (40%)
The heaviest component — and the one that catches keyword-stuffer traps:
- **Title Match (35%)**: Word-set comparison (not fuzzy!) to prevent "Civil Engineer" from matching "AI Engineer"
- **Skills Match (30%)**: Weighted by proficiency × duration × endorsements. Short skill names (e.g., "ML") use exact matching to avoid "HTML" false positives
- **Experience Fit (15%)**: Gaussian penalty centered on ideal range (5-9 years)
- **Industry Relevance (12%)**: Penalizes consulting-only backgrounds per JD instruction
- **Education Tier (8%)**: Bonus for tier-1 institutions in relevant fields

A **keyword-stuffer multiplier** applies a 0.55x penalty when title is a clear mismatch (e.g., "Marketing Manager" with AI skills listed).

### 3. Behavioral Scoring (25%)
Integrates RedRob platform signals: recruiter response rates, interview completion, GitHub activity, profile completeness, recency, and response time.

### 4. Bonus Modifiers (10%)
Rewards actively available candidates: open-to-work flag, verified contacts, willingness to relocate, low notice period, and recent recruiter saves.

### 5. Honeypot Detection
Seven checks identify impossible profiles:
- Career duration vs. claimed experience mismatch
- Expert proficiency with zero usage duration
- Excessive expert-level skills
- Endorsement/duration anomalies
- Impossible career timelines
- Assessment score contradictions
- Too many career entries for stated experience

Flagged honeypots receive a 90% reduction in structured score.

### 6. Reasoning Generation
Each candidate gets a unique, fact-based reasoning string with:
- Rank-appropriate tone (positive for top, honest about gaps for bottom)
- Specific references to actual profile data (no hallucination)
- JD requirement connections
- Honest concern acknowledgment

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Structured > Semantic weight | Semantic embeddings are too compressed (all candidates look similar). Structured scoring provides real differentiation. |
| Word-set title matching | SequenceMatcher falsely matches "civil engineer" ↔ "ai engineer" at 0.80 due to shared "engineer" suffix. |
| No LLM in ranking | Spec explicitly bans API calls. Local models too slow for 100K candidates in 5 min. |
| `all-mpnet-base-v2` model | Best quality general-purpose sentence-transformer that runs efficiently on CPU. |
| Multiplicative trap penalty | Additive penalties can be overcome by high scores in other components. Multiplicative ensures traps stay at the bottom. |

## Compute Compliance

| Constraint | Our System |
|------------|-----------|
| Runtime ≤ 5 min | ✅ ~47 seconds for 100 candidates |
| Memory ≤ 16 GB | ✅ Peak ~2 GB |
| CPU only | ✅ No GPU required |
| No network | ✅ No API calls during ranking |
| Disk ≤ 5 GB | ✅ ~500 MB (model + data) |

## License

Built for the Redrob India Runs Data & AI Challenge.
