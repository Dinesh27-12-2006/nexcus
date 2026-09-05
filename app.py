"""
app.py
------
Flask application for the NexusTiQ 24 Medical Triage Assistant (PS01).

Architecture:
  triage_engine.py  -> deterministic rule matching (the safety-critical core)
  rag_pipeline.py   -> TF-IDF + cosine similarity retrieval over triage_rules.json
  llm_provider.py   -> pluggable natural-language phrasing (mock by default)
  app.py (this file)-> Flask routes, session state machine, SQLite persistence

Flow per session:
  1. POST /api/start-triage         -> new session, greeting
  2. POST /api/triage (repeatedly)  -> patient answers accumulate; engine
                                        re-evaluates after every turn; once
                                        confidence/turn criteria are met (or
                                        red flags fire immediately), a full
                                        triage note is produced and persisted.
  3. GET  /api/triage-history        -> past sessions from triages.db
  4. GET  /api/rules                 -> rules for the "Rules" UI tab
  5. GET  /api/sample-cases          -> bundled sample cases for the "Samples" tab
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from triage_engine import TriageEngine, RuleMatch
from rag_pipeline import RAGPipeline
from llm_provider import get_provider

load_dotenv()

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    # flask-cors is listed in requirements.txt; this only guards against a
    # broken/partial install so the app still boots and API calls from the
    # bundled same-origin UI keep working.
    pass

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "triages.db")
RULES_JSON_PATH = os.path.join(BASE_DIR, "triage_rules.json")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data.json")

MIN_EXCHANGES_BEFORE_DECISION = 2   # at least this many patient turns...
MAX_EXCHANGES_BEFORE_FORCE = 5      # ...but never drag past this many.

engine = TriageEngine(RULES_JSON_PATH)
rag = RAGPipeline(RULES_JSON_PATH)
llm = get_provider()


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triage_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            patient_description TEXT,
            triage_note TEXT,
            turn_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Migrate existing DBs: add turn_count if missing.
    try:
        conn.execute("ALTER TABLE triage_sessions ADD COLUMN turn_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------- #
# In-memory session state (single-process demo mode, as documented)
# ---------------------------------------------------------------------- #
class TriageSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns = []              # list of {"role": ..., "content": ...}
        self.patient_initial = ""
        self.accumulated_text = ""
        self.complete = False
        self.triage_note = None
        self.language = "en"
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def patient_turn_count(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "patient")


sessions: dict = {}


# ---------------------------------------------------------------------- #
# Triage note construction
# ---------------------------------------------------------------------- #
def build_triage_note(session: TriageSession, match: RuleMatch, retrieved_docs: list) -> dict:
    conflict = engine.detect_conflict(session.accumulated_text)
    escalation_required = bool(match.escalate) or bool(conflict)
    escalation_reason = None
    if conflict:
        escalation_reason = conflict
    elif match.matched_red_flags:
        escalation_reason = f"Red flag symptoms detected: {', '.join(match.matched_red_flags)}"
    elif match.escalate:
        escalation_reason = "Rule requires mandatory escalation for this presentation."

    follow_up_turns = [t["content"] for t in session.turns if t["role"] == "patient"][1:]

    note = {
        "urgency_level": match.urgency.upper(),
        "recommended_department": match.department,
        "rule_applied": match.rule_id,
        "rule_condition": match.condition,
        "patient_reported": session.patient_initial,
        "follow_ups_established": " | ".join(follow_up_turns) if follow_up_turns else "None provided",
        "unknown_factors": _infer_unknowns(match),
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
        "reasoning": match.description,
        "matched_keywords": match.matched_keywords,
        "matched_red_flags": match.matched_red_flags,
        "retrieved_context": [
            {"doc_id": d["doc_id"], "relevance": d["score"]} for d in retrieved_docs
        ],
    }
    note["plain_summary"] = llm.phrase_summary(note)
    return note


def _infer_unknowns(match: RuleMatch) -> str:
    """List follow-up questions belonging to the matched rule that don't
    appear to have been directly echoed back — a lightweight, honest way
    to surface what's still unconfirmed, without claiming false certainty."""
    unresolved = match.follow_ups[2:] if len(match.follow_ups) > 2 else []
    if not unresolved:
        return "None identified"
    return "; ".join(unresolved)


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start-triage", methods=["POST"])
def start_triage():
    data = request.get_json(force=True, silent=True) or {}
    language = data.get("language", "en")
    
    session_id = str(uuid.uuid4())
    sessions[session_id] = TriageSession(session_id)
    sessions[session_id].language = language
    
    greeting = "Hello, I'm the intake triage assistant. Can you describe what's bringing you in today?"
    if language.lower().startswith("es"):
        greeting = "Hola, soy el asistente de triaje. ¿Puede describir qué le trae por aquí hoy?"
        
    sessions[session_id].add_turn("assistant", greeting)
    return jsonify({"session_id": session_id, "message": greeting, "provider": llm.name})


@app.route("/api/triage", methods=["POST"])
def process_triage():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    patient_input = (data.get("message") or "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found. Start a new triage session."}), 404
    if not patient_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    session = sessions[session_id]

    if session.complete:
        return jsonify({
            "message": "This triage session is already complete.",
            "triage_note": session.triage_note,
            "complete": True,
        })

    session.add_turn("patient", patient_input)
    if not session.patient_initial:
        session.patient_initial = patient_input
    session.accumulated_text = (session.accumulated_text + " " + patient_input).strip()

    match = engine.evaluate(session.accumulated_text)
    retrieved = rag.retrieve(session.accumulated_text, top_k=3)

    turn_count = session.patient_turn_count()
    critical_now = match.urgency in ("Critical", "High") and (match.escalate or match.matched_red_flags)
    ready_for_decision = (
        critical_now
        or turn_count >= MAX_EXCHANGES_BEFORE_FORCE
        or (turn_count >= MIN_EXCHANGES_BEFORE_DECISION and match.rule_id != "UNCERTAIN_001")
    )

    if ready_for_decision:
        note = build_triage_note(session, match, retrieved)
        session.triage_note = note
        session.complete = True
        plain_summary = llm.phrase_summary(note, language=session.language)
        note["plain_summary"] = plain_summary
        session.add_turn("assistant", plain_summary)

        conn = get_db()
        conn.execute(
            "INSERT INTO triage_sessions (session_id, patient_description, triage_note, turn_count) VALUES (?, ?, ?, ?)",
            (session_id, session.patient_initial, json.dumps(note), session.patient_turn_count()),
        )
        conn.commit()
        conn.close()

        return jsonify({
            "complete": True,
            "triage_note": note,
            "message": plain_summary,
        })

    # Not ready yet -> ask follow-ups grounded in the matched (or nearest) rule
    followup_message = llm.phrase_followups(match.condition, match.follow_ups, session.accumulated_text, language=session.language)
    session.add_turn("assistant", followup_message)

    return jsonify({
        "message": followup_message,
        "complete": False,
        "turn": turn_count,
        "retrieved_context": [{"doc_id": d["doc_id"], "relevance": d["score"]} for d in retrieved],
    })


@app.route("/api/triage-history", methods=["GET"])
def get_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, session_id, patient_description, triage_note, created_at "
        "FROM triage_sessions ORDER BY created_at DESC LIMIT 25"
    ).fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "description": row["patient_description"],
            "triage": json.loads(row["triage_note"]),
            "timestamp": row["created_at"],
        })
    return jsonify(history)


@app.route("/api/rules", methods=["GET"])
def get_rules():
    with open(RULES_JSON_PATH, "r") as f:
        return jsonify(json.load(f))


@app.route("/api/sample-cases", methods=["GET"])
def get_sample_cases():
    with open(SAMPLE_DATA_PATH, "r") as f:
        return jsonify(json.load(f))


@app.route("/api/triage-history/<session_id>", methods=["GET"])
def get_session_detail(session_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id, session_id, patient_description, triage_note, turn_count, created_at "
        "FROM triage_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "id": row["id"],
        "session_id": row["session_id"],
        "description": row["patient_description"],
        "triage": json.loads(row["triage_note"]),
        "turn_count": row["turn_count"] or 0,
        "timestamp": row["created_at"],
    })


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    conn = get_db()
    rows = conn.execute(
        "SELECT triage_note, turn_count FROM triage_sessions"
    ).fetchall()
    conn.close()

    total = len(rows)
    if total == 0:
        return jsonify({
            "total_sessions": 0,
            "urgency_distribution": {},
            "department_distribution": {},
            "escalation_rate": 0,
            "avg_turns": 0,
        })

    urgency_dist = {}
    dept_dist = {}
    escalations = 0
    total_turns = 0

    for row in rows:
        note = json.loads(row["triage_note"])
        urg = note.get("urgency_level", "Unknown")
        dept = note.get("recommended_department", "Unknown")
        urgency_dist[urg] = urgency_dist.get(urg, 0) + 1
        dept_dist[dept] = dept_dist.get(dept, 0) + 1
        if note.get("escalation_required"):
            escalations += 1
        total_turns += (row["turn_count"] or 0)

    return jsonify({
        "total_sessions": total,
        "urgency_distribution": urgency_dist,
        "department_distribution": dept_dist,
        "escalation_rate": round(escalations / total * 100, 1),
        "avg_turns": round(total_turns / total, 1),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "llm_provider": llm.name, "rules_loaded": len(engine.all_rules())})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
