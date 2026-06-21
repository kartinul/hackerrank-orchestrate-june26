# HackerRank Orchestrate — Multi-Modal Evidence Review

A system that verifies damage claims using submitted images, claim conversations, user history, and minimum evidence requirements.

Built for the **HackerRank Orchestrate** 24-hour hackathon.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec and I/O schema.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Solution approach](#solution-approach)
3. [Quick start](#quick-start)
4. [Running evaluation](#running-evaluation)
5. [Submission checklist](#submission-checklist)

---

## Repository layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools and transcript logging
├── problem_statement.md              # Full task description and I/O schema
├── README.md                         # You are here
├── code/                             # Full runnable solution
│   ├── README.md                     # Setup, run instructions, and approach overview
│   ├── main.py                       # Entry point — runs inference on dataset/claims.csv
│   ├── orchestrator.py               # Per-claim pipeline
│   ├── ai_helpers.py                 # Gemini VLM client, key manager, response parser
│   ├── prompts.py                    # Prompt engineering (system instruction + user prompt)
│   ├── constants.py                  # Allowed categorical values
│   ├── common_helpers.py             # CSV read/write helpers
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # API key template
│   └── evaluation/
│       ├── main.py                   # Evaluation entry point
│       └── evaluation_report.md      # Operational analysis
└── dataset/
    ├── sample_claims.csv             # Inputs + expected outputs (development)
    ├── claims.csv                    # Inputs only — run your system on these rows
    ├── user_history.csv              # Historical claim counts and risk context
    ├── evidence_requirements.csv     # Minimum image evidence requirements
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

---

## Solution approach

The system uses a **single-pass Vision-Language Model pipeline** built on Google Gemini.

For each claim:

1. **Context assembly** — user claim transcript, user history (rejection rate, risk flags), and per-object evidence requirements are assembled into a structured prompt.
2. **VLM call** — images are uploaded via the Gemini Files API and passed in a single multimodal request alongside the prompt.
3. **Structured output** — the model responds in JSON, validated by a Pydantic schema. This eliminates markdown/prose parsing errors.
4. **Normalisation** — image IDs are normalised to consistent `img_N` labels; composite `issue_type` values are resolved to a single canonical token.
5. **Parallel execution** — a `ThreadPoolExecutor` scales workers to `5 × number_of_api_keys`, with a sliding-window rate limiter (5 RPM per key).

See [`code/README.md`](./code/README.md) for a full breakdown of design decisions and file structure.

---

## Quick start

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cd code
cp .env.example .env
# Edit .env and add at least GEMINI_KEY_1=<your_key>
```

### 3. Run inference

```bash
cd code
python main.py
```

Output is written to `dataset/output.csv`.

---

## Running evaluation

```bash
cd code
python evaluation/main.py
```

Evaluates the system against `dataset/sample_claims.csv` (labeled) and writes a report to `code/evaluation/evaluation_report.md`.

---

## Submission checklist

- [ ] `dataset/output.csv` has one row per row in `dataset/claims.csv`
- [ ] `dataset/output.csv` has the exact required columns in the required order
- [ ] `code/evaluation/` folder is included in `code.zip`
- [ ] No secrets are committed — API keys live in `code/.env` only
- [ ] Chat transcript (`%USERPROFILE%\hackerrank_orchestrate\log.txt`) is ready to upload
