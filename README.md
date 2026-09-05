TRACK_ID=PS01

# NexusTiQ 24 - Medical Triage Assistant

## Project Overview

A medical intake triage assistant that takes patient descriptions in plain
language, asks targeted follow-up questions, and produces a structured
triage note: urgency level, recommended department, the exact rule that
fired, and what's still unknown — with mandatory escalation to human review
whenever the rules or the patient's own words say so.

**Track:** Healthcare - Patient Intake Triage Assistant (PS01)

## Critical Features

- **Never diagnoses** — a deterministic rule engine (`triage_engine.py`)
  decides urgency/department/escalation; the LLM is only used to phrase
  follow-up questions and summaries in natural language, never to make the
  clinical decision itself.
- **Cites the exact rule** behind every recommendation, plus which keywords
  and red flags matched.
- **Escalates to human review** whenever: a rule mandates it, a red-flag
  symptom is detected, the patient's statements contradict each other, or
  five exchanges pass without a confident match.
- **Handles edge cases** — null/no-complaint cases, conflicting information,
  and negated symptoms ("no fever" doesn't falsely trigger the fever rule).
- **Grounded in documents** — real TF-IDF + cosine similarity retrieval
  (`rag_pipeline.py`) over `triage_rules.json`, with the retrieved rules and
  relevance scores shown live in the UI.
- **Runs with zero API key** — a deterministic mock LLM provider is the
  default, so the app is fully testable offline; set `GEMINI_API_KEY` to
  switch to live Gemini phrasing with no code changes.

## Tech Stack

- **Backend:** Python Flask
- **Frontend:** HTML5 + vanilla JS (no build step)
- **Triage decisions:** deterministic rule engine, not the LLM
- **RAG:** scikit-learn `TfidfVectorizer` + cosine similarity, precomputed at startup
- **LLM (optional, phrasing only):** Google Gemini via `google-generativeai`, or offline mock
- **Persistence:** SQLite (`triages.db`)

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8000`. No `GEMINI_API_KEY` is required — the
app runs fully offline using the built-in mock LLM provider. To use live
Gemini phrasing instead, copy `.env.example` to `.env` and set
`GEMINI_API_KEY`.

## Project Structure

```
├── app.py                 # Flask routes, session state machine, persistence
├── triage_engine.py        # Deterministic rule matching (the safety-critical core)
├── rag_pipeline.py         # TF-IDF + cosine similarity retrieval over triage_rules.json
├── llm_provider.py         # Pluggable LLM: MockProvider (default) / GeminiProvider
├── triage_rules.json        # Source of truth for all triage rules
├── sample_data.json         # Bundled test cases, browsable in the UI
├── requirements.txt
├── templates/index.html     # Web UI (Triage / Rules / Samples / History tabs)
└── triages.db                # SQLite triage history (created at runtime)
```

## How It Works

1. **Session start** — greets the patient, asks for their initial complaint.
2. **Each patient turn** — `triage_engine.evaluate()` re-scores every rule
   against the full accumulated text (keyword + red-flag matching, with
   basic negation handling so "no fever" doesn't misfire). `rag_pipeline`
   retrieves the top-3 most relevant rule documents by cosine similarity for
   display and for grounding the LLM's follow-up phrasing.
3. **Decision point** — a triage note is finalized immediately if a red flag
   or mandatory-escalation rule fires; otherwise after 2+ patient turns once
   a confident (non-`UNCERTAIN`) rule matches, or by turn 5 regardless (at
   which point an unresolved case is force-escalated to `UNCERTAIN_001`
   rather than guessed).
4. **Persistence** — completed triage notes are saved to `triages.db` and
   viewable in the History tab.

## Triage Rules Included

Loaded directly from `triage_rules.json` (single source of truth — not
duplicated in a prompt):

1. Fever (`FEVER_001`) → High → Urgent Care
2. Chest Pain - Acute (`CHEST_PAIN_001`) → Critical → Emergency (mandatory escalation)
3. Difficulty Breathing - Acute (`BREATHING_001`) → Critical → Emergency (mandatory escalation)
4. Traumatic Injury (`INJURY_001`) → High → Emergency
5. Severe Abdominal Pain (`ABDOMINAL_PAIN_001`) → High → Emergency
6. Minor Symptoms (`MILD_SYMPTOMS_001`) → Low → General
7. Insufficient Information (`UNCERTAIN_001`) → Moderate → Urgent Care (mandatory escalation)
8. Null Case (`NULL_001`) — no clear complaint → Routine → General

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/start-triage` | Start a new session, returns `session_id` + greeting |
| POST | `/api/triage` | Send a patient message, get a follow-up or final triage note |
| GET | `/api/triage-history` | Last 25 completed triage sessions |
| GET | `/api/rules` | All triage rules (powers the Rules tab) |
| GET | `/api/sample-cases` | Bundled sample cases (powers the Samples tab) |
| GET | `/api/health` | Liveness check + which LLM provider is active |

## Known Limitations

1. In-memory sessions (single-process demo mode) — restarting the server
   clears active (not yet completed) sessions; completed triage notes
   persist in `triages.db`.
2. Keyword-based matching with light negation handling, not full clinical
   NLP — deliberately conservative: ambiguous phrasing falls through to
   `UNCERTAIN_001` and gets escalated rather than guessed.
3. No external API calls except the optional Gemini provider, which is
   used for phrasing only and never for the clinical decision.

## Author

Built for NexusTiQ 24 Hackathon — Track: Healthcare Innovation (PS01)
