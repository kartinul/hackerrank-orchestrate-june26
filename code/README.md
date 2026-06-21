# Multi-Modal Evidence Review — Solution

This folder contains the complete solution for the HackerRank Orchestrate hackathon challenge.

---

## Approach Overview

The system uses a **single-pass Vision-Language Model (VLM) pipeline** powered by Google Gemini.

For each damage claim the system:

1. **Loads context** — user claim transcript, user history (risk flags, rejection history), and per-object evidence requirements.
2. **Builds a structured prompt** — a carefully engineered system instruction that enforces strict categorical outputs, chain-of-thought reasoning, and explicit rules for `object_part`, `issue_type`, `claim_status`, and image ID referencing.
3. **Uploads images and calls the VLM** — images are uploaded to the Gemini Files API and passed alongside the prompt in a single multimodal request.
4. **Parses and normalises output** — the JSON response is validated via a Pydantic schema (`ClaimVerificationOutput`), image IDs are normalised to consistent `img_N` labels, and composite issue types are resolved to a single canonical value.
5. **Writes `output.csv`** — results are reconstructed in original input order and written to `dataset/output.csv`.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Single VLM call per claim | Minimises latency and token overhead; Gemini's context window handles all evidence at once |
| `response_schema` (JSON mode) | Forces structured output, eliminating markdown/prose parsing errors |
| Multi-key `KeyManager` | Rotates across `GEMINI_KEY_*` env vars, enforcing 5 RPM per key with a sliding window |
| Model fallover list | Tries `gemini-3.5-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash`; skips a model globally for 60 s on 503 |
| Parallel `ThreadPoolExecutor` | Scales workers to `5 × num_keys`; respects per-key RPM via `KeyManager` |
| Chain-of-thought field | Forces the model to reason before committing to categorical values, reducing hallucination |
| Image ID normalisation | `_normalize_img_id()` maps tool-generated names (`input_file_0`, `image_0.png`) back to `img_1, img_2, …` |

---

## File Structure

```text
code/
├── main.py               # Entry point — runs inference on dataset/claims.csv
├── orchestrator.py       # Per-claim pipeline: context assembly → VLM call → output row
├── ai_helpers.py         # Gemini client, KeyManager, Pydantic schema, JSON parser
├── prompts.py            # System instruction and user prompt builder
├── constants.py          # Allowed categorical values (claim status, issue types, parts, etc.)
├── common_helpers.py     # CSV read/write helpers
├── requirements.txt      # Python dependencies
├── .env.example          # Template for API key env vars
├── .env                  # Your actual keys (git-ignored)
└── evaluation/
    ├── main.py           # Evaluation entry point — runs on dataset/sample_claims.csv
    └── evaluation_report.md  # Operational analysis (model calls, cost, latency)
```

---

## Setup

### 1. Python version

Python **3.10+** is required (uses `list[str]` type hints and `match`-style patterns).

### 2. Install dependencies

Run from inside the `code/` directory:

```bash
# Windows
cd code
pip install -r requirements.txt

# macOS / Linux
cd code
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|---|---|
| `pandas >= 2.0.0` | CSV I/O |
| `google-genai` | Gemini VLM client |
| `python-dotenv` | Load `.env` file |
| `pydantic` | Structured JSON response validation |
| `tqdm` | Progress bar |

### 3. Configure API keys

Copy the example env file and add your Gemini API key(s):

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_KEY_1=your_first_gemini_api_key_here
GEMINI_KEY_2=your_second_gemini_api_key_here   # optional — adds throughput
GEMINI_KEY_3=your_third_gemini_api_key_here    # optional
```

- You need **at least one** key (`GEMINI_KEY_1`).
- Adding more keys increases parallel throughput (5 RPM per key).
- Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

> **Never commit your `.env` file.** It is already in `.gitignore`.

---

## Running Inference

Run from the `code/` directory:

```bash
# Process all claims (writes to dataset/output.csv)
python main.py
```

To process only the first N claims (useful for testing):

```python
# In main.py, line 14:
ENTRIES = 5   # change from -1 to any positive integer
```

Expected output:

```
Starting 5 parallel agents...

Processing Claims: 100%|████████████| 50/50 [02:14<00:00,  2.68s/it]
Processed 50 claims and wrote to ../dataset/output.csv
```

The output CSV is written to `dataset/output.csv` (one directory above `code/`).

---

## Running Evaluation

Run from the `code/` directory:

```bash
python evaluation/main.py
```

This runs the same pipeline against `dataset/sample_claims.csv` (which includes ground-truth labels) and writes evaluation metrics and a comparison report to `code/evaluation/evaluation_report.md`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No API keys found` | Ensure `.env` exists in `code/` with at least `GEMINI_KEY_1` set |
| `All models exhausted or failed` | All three models hit rate limits or 503; wait ~60 s and retry |
| `Image file not found at …` | Run `main.py` from inside the `code/` directory, not the repo root |
| `Failed to parse JSON response` | Rare schema mismatch; the fallback row is written with `not_enough_information` |
