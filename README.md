---
title: Candidate Matcher
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
app_port: 7860
---

# AI Candidate Ranker

## Problem
Basic recruitment systems rely entirely on primitive keyword matching. This leads to two massive problems:
1. **Keyword Stuffers Win**: A "Marketing Manager" who lists 10 AI keywords on their resume outranks an actual "Machine Learning Engineer" who only listed 3. 
2. **Semantic Mismatch**: Great candidates are ignored simply because they used the term "NLP" instead of "Natural Language Processing".
Furthermore, these systems cannot evaluate behavioral signals (like interview completion rates) and are easily fooled by "honeypot" (fake or impossible) profiles.

## Solution
A 4-layer hybrid AI ranking system built for the **Redrob Hackathon**. It evaluates job fit the way a great recruiter would:
- **Two-Stage Semantic Scoring**: Uses advanced NLP to conceptually understand the resume rather than just matching keywords. It first filters candidates via a fast FAISS Bi-Encoder (`all-MiniLM-L6-v2`), then re-ranks the top results with a highly accurate Cross-Encoder (`ms-marco-MiniLM-L-12-v2`).
- **Structured Scoring**: Applies strict rule-based heuristics to heavily penalize "keyword stuffer" traps and evaluate hard skills.
- **Behavioral Scoring**: Uses platform signals to evaluate candidate engagement and reliability, including **Live GitHub Scraping** to pull real contribution metrics.
- **Honeypot Detection**: Automatically flags mathematically impossible profiles.
- **Bias Mitigation**: Automatically strips demographic proxies (names, gendered language, graduation years) prior to text embedding to ensure 100% merit-based evaluation.
- **Universal Resume Parsing**: Supports batch JSON parsing as well as individual **PDF and DOCX** raw resume uploads via a free regex-heuristic engine.

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

## Tech Stack
- **Language**: Python
- **AI/ML**: HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2` & `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Web App / Dashboard**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Document Parsing**: `pypdf`, `mammoth`

## Screenshots
![Dashboard Overview](assets/dashboard_1.png)
![Ranked Candidates](assets/dashboard_2.png)
![Candidate Deep Dive](assets/dashboard_3.png)

## Demo Link
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TanayJalan/candidate-matcher)

## Installation

### 1. Quick Start
We've included a convenience script that automatically installs dependencies and runs the entire pipeline:
```bash
git clone https://github.com/TanayJalan/AI-Candidate-Ranker.git
cd AI-Candidate-Ranker
chmod +x run.sh
./run.sh
```

### 2. Manual Execution
If you prefer to run things manually:
```bash
# Install dependencies
pip install -r requirements.txt

# Produce Submission CSV
python rank.py --candidates data/raw/sample_candidates.json --out data/output/submission.csv --verbose
```

### 3. Launch Interactive Dashboard
We built a Streamlit dashboard to visualize score breakdowns, flag honeypots, and provide a deep dive into candidate reasoning.
```bash
streamlit run app.py
```

## Results
The system strictly adheres to hackathon compute limits while processing large candidate volumes:
- **Runtime**: ~47 seconds for 100 candidates (Well under the 5 min limit)
- **Memory**: Peak ~2 GB (Under the 16 GB limit)
- **Compute**: Runs 100% locally on CPU. No GPU or external APIs required.
- **Disk**: ~500 MB for data and the downloaded Transformer model.

## Future Improvements
- **Learning to Rank**: Replace hardcoded heuristic weights with an XGBoost or LightGBM model trained on historical hiring data to dynamically learn the best weights.
- **Skill Ontology Knowledge Graph**: Implement a graph where the system understands that "Pandas" and "NumPy" are children of "Python Data Science" for smarter skill mapping.
- **Local LLMs**: Utilize a quantized local LLM (like `Llama-3-8B`) to dynamically generate highly conversational, unique reasoning paragraphs for each candidate without relying on templates.

## License

MIT License