---
title: Candidate Matcher
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
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
- **Semantic Scoring**: Uses advanced NLP to conceptually understand the resume rather than just matching keywords.
- **Structured Scoring**: Applies strict rule-based heuristics to heavily penalize "keyword stuffer" traps and evaluate hard skills.
- **Behavioral Scoring**: Uses platform signals to evaluate candidate engagement and reliability.
- **Honeypot Detection**: Automatically flags mathematically impossible profiles.

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
- **AI/ML**: HuggingFace `sentence-transformers` (`all-mpnet-base-v2`)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Web App / Dashboard**: Streamlit
- **Data Handling**: Pandas, NumPy

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
- **Two-Stage Retrieval**: Integrate a Cross-Encoder for the final Top 100 candidates to significantly boost the accuracy of the semantic ranking.
- **Learning to Rank**: Replace hardcoded heuristic weights with an XGBoost or LightGBM model trained on historical hiring data to dynamically learn the best weights.
- **Skill Ontology Knowledge Graph**: Implement a graph where the system understands that "Pandas" and "NumPy" are children of "Python Data Science" for smarter skill mapping.
- **Local LLMs**: Utilize a quantized local LLM (like `Llama-3-8B`) to dynamically generate highly conversational, unique reasoning paragraphs for each candidate without relying on templates.
- **Bias Mitigation Layer**: Automatically strip demographic proxies (names, gendered language, graduation years) prior to text embedding to ensure 100% merit-based evaluation.

## License

MIT License

Copyright (c) 2026 Tanay Jalan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

