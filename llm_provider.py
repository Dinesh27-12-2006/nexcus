"""
llm_provider.py
---------------
Pluggable LLM layer. The LLM is used ONLY for two things in this app:
  1. Turning a matched rule's structured follow-up questions into a natural,
     empathetic sentence or two to show the patient.
  2. Turning a final triage decision into a short plain-English summary.

It is never used to decide urgency, department, or escalation — that
decision comes entirely from triage_engine.py's deterministic rule matching.
This separation is what makes the clinical output reproducible and auditable
regardless of which model (or no model at all) is behind this layer.

Two providers are implemented:
  - MockProvider: fully offline, deterministic, zero API key required.
    Used by default so the app runs out of the box for grading/testing.
  - GeminiProvider: thin adapter around google-generativeai, activated
    automatically when GEMINI_API_KEY is set in the environment.

To add another provider (OpenAI, Anthropic, etc.), implement the same
two methods (`phrase_followups`, `phrase_summary`) and register it in
get_provider().
"""
import os
import random


class BaseProvider:
    name = "base"

    def phrase_followups(self, condition: str, follow_up_questions: list, patient_text: str, language: str = "en") -> str:
        raise NotImplementedError

    def phrase_summary(self, decision: dict, language: str = "en") -> str:
        raise NotImplementedError


class MockProvider(BaseProvider):
    """
    Deterministic, template-based provider. No network calls, no API key.
    This is the default provider so the app is fully runnable and gradeable
    offline, and so behavior is 100% reproducible in tests/CI.
    """
    name = "mock"

    _EMPATHY_OPENERS = {
        "en": [
            "Thanks for sharing that.",
            "I understand — let's get a clearer picture.",
            "Okay, noted.",
            "Got it, thank you.",
        ],
        "es": [
            "Gracias por compartir esto.",
            "Entiendo — vamos a obtener una imagen más clara.",
            "De acuerdo, anotado.",
            "Entendido, gracias.",
        ]
    }

    def phrase_followups(self, condition: str, follow_up_questions: list, patient_text: str, language: str = "en") -> str:
        lang_code = language.lower()[:2]
        openers = self._EMPATHY_OPENERS.get(lang_code, self._EMPATHY_OPENERS["en"])
        opener = random.Random(len(patient_text)).choice(openers)
        
        if not follow_up_questions:
            msg = "Can you tell me a bit more about your main concern?" if lang_code == "en" else "¿Puede contarme un poco más sobre su problema principal?"
            return f"{opener} {msg}"
            
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(follow_up_questions[:3]))
        lead = f"I need a few more details about your {condition.lower()}:" if lang_code == "en" else f"Necesito algunos detalles más sobre su condición:"
        return f"{opener} {lead}\n{numbered}"

    def phrase_summary(self, decision: dict, language: str = "en") -> str:
        lang_code = language.lower()[:2]
        urgency = decision["urgency_level"]
        dept = decision["recommended_department"]
        rule = decision["rule_applied"]
        
        esc_en = " This case has been flagged for human clinician review." if decision["escalation_required"] else ""
        esc_es = " Este caso ha sido marcado para revisión humana por un médico." if decision["escalation_required"] else ""
        
        if lang_code == "en":
            return f"Based on the information gathered, this has been triaged as **{urgency}** priority, routed to **{dept}**, per rule {rule}.{esc_en}"
        else:
            return f"Según la información recopilada, esto ha sido clasificado como prioridad **{urgency}**, dirigido a **{dept}**, según la regla {rule}.{esc_es}"


class GeminiProvider(BaseProvider):
    """
    Adapter around google-generativeai. Only activated when GEMINI_API_KEY
    is present. Falls back to MockProvider behavior on any API error so a
    transient network/API issue never breaks the triage flow.
    """
    name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self._fallback = MockProvider()

    def phrase_followups(self, condition: str, follow_up_questions: list, patient_text: str, language: str = "en") -> str:
        try:
            prompt = (
                "You are a calm, empathetic hospital intake assistant. "
                "You do NOT diagnose. Rephrase the following clinical follow-up "
                "questions into a short, warm, plain-English message (max 3 sentences "
                "of lead-in, then the questions as a numbered list). "
                f"Translate your final response to this language: {language}. "
                f"Patient's condition category: {condition}. "
                f"Patient said: {patient_text!r}. "
                f"Questions to ask: {follow_up_questions}"
            )
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return self._fallback.phrase_followups(condition, follow_up_questions, patient_text, language)

    def phrase_summary(self, decision: dict, language: str = "en") -> str:
        try:
            prompt = (
                "You are a hospital intake assistant. Do NOT diagnose. "
                "Write a 2-3 sentence plain-English summary of this triage decision "
                f"for the patient: {decision}. "
                f"Translate your final response to this language: {language}."
            )
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return self._fallback.phrase_summary(decision, language)


def get_provider() -> BaseProvider:
    """
    Provider selection: Gemini if GEMINI_API_KEY is set and the SDK is
    importable, otherwise the offline Mock provider. This keeps the app
    runnable with zero configuration while remaining a one-line swap to
    go live with a real key.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            return GeminiProvider(api_key=api_key, model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        except Exception:
            pass
    return MockProvider()
