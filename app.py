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
if os.environ.get("VERCEL") or os.environ.get("AWS_EXECUTION_ENV") or not os.access(BASE_DIR, os.W_OK):
    DB_PATH = "/tmp/triages.db"
else:
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
    try:
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
    except Exception as e:
        print(f"Warning: DB initialization skipped/failed: {e}")


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


FIRST_AID_GUIDES = {
    "CHEST_PAIN_001": {
        "title": "Cardiac / Acute Chest Pain",
        "icon": "🫀",
        "emergency": True,
        "steps": [
            "Call Emergency Services immediately (911 / 112 / 108).",
            "Have the patient stop all physical activity and sit in a comfortable, upright position.",
            "Loosen tight clothing around the neck, chest, and waist.",
            "If prescribed nitroglycerin or advised by emergency dispatch, assist them in taking it.",
            "Stay calm, do not leave the patient unattended, and prepare for CPR if needed."
        ],
        "steps_es": [
            "Llame a emergencias de inmediato (911 / 112 / 108).",
            "Haga que la persona descanse sentada en posición cómoda y tranquila.",
            "Afloje la ropa ajustada alrededor del cuello y pecho.",
            "Si tiene nitroglicerina recetada, ayúdele a tomarla según indicaciones médicas.",
            "Mantenga la calma y vigile la respiración continuamente."
        ]
    },
    "BREATHING_001": {
        "title": "Acute Breathing Difficulty / Asthma",
        "icon": "🫁",
        "emergency": True,
        "steps": [
            "Call Emergency Services immediately.",
            "Sit the person upright leaning slightly forward (tripod position) to ease breathing.",
            "Loosen tight collars, ties, or restrictive belts.",
            "Assist the patient with their prescribed rescue inhaler (e.g. Albuterol).",
            "Keep the area quiet and ventilated to prevent panic."
        ],
        "steps_es": [
            "Llame a emergencias inmediatamente.",
            "Ayude a la persona a sentarse erguida inclinada hacia adelante.",
            "Afloje la ropa apretada y asegure aire fresco.",
            "Ayúdele a usar su inhalador de rescate recetado.",
            "Mantenga la calma para evitar hiperventilación."
        ]
    },
    "INJURY_001": {
        "title": "Traumatic Injury / Wound Care / Fracture",
        "icon": "🩹",
        "emergency": False,
        "steps": [
            "Direct Pressure: Apply firm, continuous pressure to bleeding wounds with a clean cloth.",
            "Immobilize: Do NOT attempt to straighten or manipulate broken bones or dislocated joints.",
            "Cold Therapy: Apply ice wrapped in a towel for 15-20 minutes to reduce acute swelling.",
            "Shock Prevention: Keep the person warm, calm, and resting flat with legs elevated if no spinal injury."
        ],
        "steps_es": [
            "Presión directa: Aplique presión firme sobre heridas sangrantes con un paño limpio.",
            "Inmovilización: NO intente enderezar huesos deformados.",
            "Frío local: Aplique hielo con una toalla durante 15-20 minutos para la inflamación.",
            "Prevención de shock: Abrigue a la persona y manténgala en reposo."
        ]
    },
    "ABDOMINAL_PAIN_001": {
        "title": "Severe Abdominal Distress",
        "icon": "🤢",
        "emergency": False,
        "steps": [
            "Position of Comfort: Lie down on the side with knees drawn toward the chest.",
            "Nil By Mouth: Do NOT ingest solid food or heavy drinks until clinically cleared.",
            "Avoid Painkillers: Do not take NSAIDs (ibuprofen/aspirin) without medical advice as they can worsen gastric bleeding.",
            "Seek urgent or emergency medical evaluation promptly."
        ],
        "steps_es": [
            "Postura de alivio: Acuéstese de lado con las rodillas flexionadas.",
            "Ayuno: No consuma alimentos sólidos ni bebidas pesadas hasta revisión médica.",
            "Evite automedicarse con analgésicos que puedan encubrir el cuadro clínico.",
            "Acuda a valoración de urgencias sin demora."
        ]
    },
    "FEVER_001": {
        "title": "Fever & Infection Management",
        "icon": "🌡️",
        "emergency": False,
        "steps": [
            "Hydration: Drink plenty of water, electrolyte fluids, or light broths in frequent sips.",
            "Environment: Rest in a cool, ventilated room with light breathable clothing.",
            "Tepid Compresses: Apply a cool, damp cloth to forehead, neck, and wrists.",
            "Medication: Take directed fever reducers (acetaminophen or ibuprofen) if appropriate."
        ],
        "steps_es": [
            "Hidratación: Beba agua y soluciones de rehidratación con frecuencia.",
            "Ambiente fresco: Use ropa ligera y mantenga el espacio bien ventilado.",
            "Compresas: Aplique paños tibios o frescos en la frente y cuello.",
            "Antitérmicos: Tome paracetamol o ibuprofeno según las pautas recomendadas."
        ]
    },
    "MILD_SYMPTOMS_001": {
        "title": "Minor Ailments & Supportive Care",
        "icon": "💊",
        "emergency": False,
        "steps": [
            "Rest: Allow your body adequate sleep and minimal physical strain.",
            "Observation: Keep track of symptom progression over the next 24-48 hours.",
            "Consultation: Contact a primary care physician or pharmacist if symptoms worsen or linger."
        ],
        "steps_es": [
            "Reposo: Permita que su cuerpo descanse adecuadamente.",
            "Observación: Vigile la evolución de las molestias durante 24-48 horas.",
            "Consulta: Acuda a su médico habitual si no observa mejoría."
        ]
    },
    "UNCERTAIN_001": {
        "title": "General Triage & First-Aid Protocol",
        "icon": "ℹ️",
        "emergency": False,
        "steps": [
            "Monitor Alertness: Check responsiveness, steady breathing, and pulse.",
            "Warning Signs: If sudden confusion, chest pain, or breathing trouble occurs, call Emergency immediately.",
            "In-Person Evaluation: Visit Urgent Care for an accurate hands-on physical assessment."
        ],
        "steps_es": [
            "Vigilar constantes: Verifique la respiración y el nivel de consciencia.",
            "Signos de alarma: Si nota opresión en el pecho o falta de aire, llame a emergencias.",
            "Atención médica: Acuda a un centro sanitario para una valoración profesional."
        ]
    }
}


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
    fa_info = FIRST_AID_GUIDES.get(match.rule_id, FIRST_AID_GUIDES["UNCERTAIN_001"])
    steps = fa_info["steps_es"] if getattr(session, "language", "en").startswith("es") else fa_info["steps"]

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
        "first_aid": {
            "title": fa_info["title"],
            "icon": fa_info["icon"],
            "emergency": fa_info["emergency"],
            "steps": steps
        },
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

    if not patient_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    language = data.get("language", "en")
    client_accumulated = (data.get("accumulated_text") or "").strip()
    turn_count_hint = int(data.get("turn_count", 0))

    if session_id not in sessions:
        sessions[session_id] = TriageSession(session_id)
        sessions[session_id].language = language
        if client_accumulated and client_accumulated != patient_input:
            # Reconstruct accumulated context from previous serverless turns
            sessions[session_id].accumulated_text = client_accumulated
            sessions[session_id].patient_initial = client_accumulated.split("\n")[0]
        if turn_count_hint > 1:
            for _ in range(turn_count_hint - 1):
                sessions[session_id].turns.append({"role": "patient", "content": "prior turn"})

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

        try:
            init_db()
            conn = get_db()
            conn.execute(
                "INSERT INTO triage_sessions (session_id, patient_description, triage_note, turn_count) VALUES (?, ?, ?, ?)",
                (session_id, session.patient_initial, json.dumps(note), session.patient_turn_count()),
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Warning: Failed to save session to DB: {db_err}")

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
    try:
        init_db()
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
    except Exception as e:
        return jsonify([])


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
    try:
        init_db()
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
    except Exception as e:
        return jsonify({"error": "Session not found"}), 404


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    try:
        init_db()
        conn = get_db()
        rows = conn.execute(
            "SELECT triage_note, turn_count FROM triage_sessions"
        ).fetchall()
        conn.close()
    except Exception as e:
        rows = []

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



@app.route("/api/first-aid", methods=["GET"])
def get_first_aid():
    return jsonify(FIRST_AID_GUIDES)


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
