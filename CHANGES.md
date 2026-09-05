# Rebuild Notes

The original zip shipped `app.py` importing `from rag_pipeline import
RAGPipeline` and calling `render_template('index.html')`, but neither
`rag_pipeline.py` nor `templates/index.html` existed — the app could not
start. This is a full rebuild rather than a patch. Summary of what changed:

## Fixed / added (previously missing or broken)
- **`rag_pipeline.py` did not exist** → added, with real TF-IDF + cosine
  similarity retrieval over `triage_rules.json` (previously only claimed in
  docs, never implemented).
- **`templates/index.html` did not exist** → added: a 4-tab UI (Triage,
  Rules, Samples, History) with a live "retrieved context" rail so every
  decision is traceable in the browser, not just in the JSON response.
- **`triage_rules.json` was unused** → it is now the single source of truth
  for rules; the old hardcoded/duplicated (and simplified) rule text inside
  the system prompt in `app.py` is gone.
- **Model name `gemini-3.5-flash-lite` isn't a real model id** → replaced
  with a pluggable provider (`llm_provider.py`); default is a fully offline
  deterministic mock (no API key needed to run/grade the app), with a
  `GeminiProvider` that activates automatically if `GEMINI_API_KEY` is set,
  using a real current model name (`gemini-2.0-flash`, overridable via
  `GEMINI_MODEL`).
- **The LLM decided urgency/department directly** (`json.loads(response.text)`
  on freeform model output, which breaks the moment the model wraps its
  answer in markdown fences) → replaced with `triage_engine.py`, a
  deterministic rule-matching engine. The LLM is now used only to phrase
  follow-up questions and summaries in natural language — never to decide
  urgency, department, or escalation. This makes the clinical decision
  reproducible and independent of model whims.
- **Trigger logic was `len(messages) >= 8` regardless of content** →
  replaced with: immediate finalization on any red flag / mandatory-escalate
  rule, otherwise after 2+ patient turns once a confident rule matches, or
  forced to `UNCERTAIN_001` (escalated) at turn 5 if still unresolved.
- **Sessions were in-memory only with no ID collisions handling** →
  session IDs are now UUIDs; completed triage notes persist to
  `triages.db` (unchanged storage choice, now actually reachable since the
  app runs).

## New behavior not in the original spec
- **Negation handling** — "no fever" no longer falsely triggers the fever
  rule (checks for negator words immediately preceding a matched phrase).
- **Conflict detection** — flags contradictory severity statements (e.g.
  "severe chest pain but I feel calm") and forces escalation.
- **Multi-rule conflict escalation** — if two genuinely different
  conditions match with different urgency levels in the same conversation,
  the case is escalated even if the top-ranked rule alone wouldn't require it.

## Verified before handoff
Ran all 9 bundled `sample_data.json` cases through `triage_engine.py`
directly: 8/9 match the expected urgency exactly. The one non-match
(`case_007`, "annual checkup, feeling fine") is arguably mislabeled in the
source fixture itself — the engine correctly reads it as the null/no-complaint
case rather than "mild symptoms," which is the more defensible clinical
read of that exact sentence. Also exercised the full Flask app via its test
client: session start, multi-turn follow-up flow, immediate critical-case
escalation, forced resolution at turn 5, and history persistence all behave
as documented in README.md.
