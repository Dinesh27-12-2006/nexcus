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

    Supported language codes (BCP-47 prefix):
        en  English   |  es  Español    |  fr  Français
        hi  हिन्दी      |  de  Deutsch    |  ar  العربية
        pt  Português  |  zh  中文
    """
    name = "mock"

    # ---------- Empathy openers ----------------------------------------
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
        ],
        "fr": [
            "Merci de partager cela.",
            "Je comprends — essayons d'y voir plus clair.",
            "D'accord, noté.",
            "Bien reçu, merci.",
        ],
        "hi": [
            "यह साझा करने के लिए धन्यवाद।",
            "मैं समझता हूँ — आइए स्थिति को और स्पष्ट करें।",
            "ठीक है, नोट किया।",
            "समझ गया, धन्यवाद।",
        ],
        "de": [
            "Vielen Dank, dass Sie das mitteilen.",
            "Ich verstehe — lassen Sie uns ein klareres Bild gewinnen.",
            "In Ordnung, notiert.",
            "Alles klar, danke.",
        ],
        "ar": [
            "شكراً لمشاركة ذلك.",
            "أفهم — دعنا نحصل على صورة أوضح.",
            "حسناً، تمّ الملاحظة.",
            "فهمت، شكراً لك.",
        ],
        "pt": [
            "Obrigado por compartilhar isso.",
            "Entendo — vamos esclarecer melhor a situação.",
            "Certo, anotado.",
            "Entendido, obrigado.",
        ],
        "zh": [
            "感谢您分享这些信息。",
            "我明白了，让我们进一步了解情况。",
            "好的，已记录。",
            "收到，谢谢。",
        ],
        "ta": [
            "பகிர்ந்ததற்கு நன்றி.",
            "புரிகிறேன் — தெளிவான பட்சத்தை பெற முயலலாம்.",
            "சரி, குறித்துக்கொண்டேன்.",
            "புரிந்தது, நன்றி.",
        ],
    }

    # ---------- Follow-up lead-in sentences ----------------------------
    _FOLLOWUP_LEAD = {
        "en": "I need a few more details about your {condition}:",
        "es": "Necesito algunos detalles más sobre su condición:",
        "fr": "J'ai besoin de quelques précisions supplémentaires sur votre état :",
        "hi": "आपकी स्थिति के बारे में मुझे कुछ और जानकारी चाहिए:",
        "de": "Ich benötige noch einige Details zu Ihrem Zustand:",
        "ar": "أحتاج إلى مزيد من التفاصيل حول حالتك:",
        "pt": "Preciso de mais alguns detalhes sobre a sua condição:",
        "zh": "我需要了解更多关于您病情的详情：",
        "ta": "உங்கள் நிலைமைப்பற்றி இன்னும் சில விவரங்கள் தேவைய்ப்படுகின்றன:",
    }

    # ---------- Fallback "tell me more" question -----------------------
    _TELL_MORE = {
        "en": "Can you tell me a bit more about your main concern?",
        "es": "¿Puede contarme un poco más sobre su problema principal?",
        "fr": "Pouvez-vous me dire un peu plus sur votre problème principal ?",
        "hi": "क्या आप अपनी मुख्य समस्या के बारे में थोड़ा और बता सकते हैं?",
        "de": "Können Sie mir etwas mehr über Ihr Hauptanliegen erzählen?",
        "ar": "هل يمكنك إخباري بمزيد من التفاصيل حول مشكلتك الرئيسية؟",
        "pt": "Pode me contar um pouco mais sobre o seu principal problema?",
        "zh": "您能告诉我更多关于您主要问题的情况吗？",
        "ta": "உங்கள் முக்கிய பிரச்சினைப்பற்றி கொஞ்சம் கூடுதலாக சொல்ல முடியுமா?",
    }

    # ---------- Triage summary templates --------------------------------
    _SUMMARY = {
        "en": "Based on the information gathered, this has been triaged as **{urgency}** priority, routed to **{dept}**, per rule {rule}.{esc}",
        "es": "Según la información recopilada, esto ha sido clasificado como prioridad **{urgency}**, dirigido a **{dept}**, según la regla {rule}.{esc}",
        "fr": "D'après les informations recueillies, ce cas a été trié en priorité **{urgency}**, orienté vers **{dept}**, conformément à la règle {rule}.{esc}",
        "hi": "एकत्र की गई जानकारी के आधार पर, इसे **{urgency}** प्राथमिकता के रूप में वर्गीकृत किया गया है, नियम {rule} के अनुसार **{dept}** को भेजा गया है।{esc}",
        "de": "Basierend auf den gesammelten Informationen wurde dieser Fall als Priorität **{urgency}** eingestuft und gemäß Regel {rule} an **{dept}** weitergeleitet.{esc}",
        "ar": "بناءً على المعلومات التي تم جمعها، تمّ تصنيف هذه الحالة على أنّها ذات أولوية **{urgency}**، وتمّ توجيهها إلى **{dept}**، وفقاً للقاعدة {rule}.{esc}",
        "pt": "Com base nas informações coletadas, este caso foi triado com prioridade **{urgency}**, encaminhado para **{dept}**, de acordo com a regra {rule}.{esc}",
        "zh": "根据收集的信息，该病例已被分诊为 **{urgency}** 优先级，按规则 {rule} 转至 **{dept}**。{esc}",
        "ta": "சேகரிக்கப்பட்ட தகவல்களின் படி, இந்த வழக்கு **{urgency}** முன்னுரிமையாக வகைப்படுத்தப்பட்டு, விதி {rule} ப்படி **{dept}** க்கு அனுப்பப்பட்டது.{esc}",
    }

    # ---------- Escalation suffix strings --------------------------------
    _ESCALATION = {
        "en": " This case has been flagged for human clinician review.",
        "es": " Este caso ha sido marcado para revisión humana por un médico.",
        "fr": " Ce cas a été signalé pour examen par un clinicien humain.",
        "hi": " इस मामले को मानव चिकित्सक की समीक्षा के लिए चिह्नित किया गया है।",
        "de": " Dieser Fall wurde zur Überprüfung durch einen menschlichen Kliniker markiert.",
        "ar": " تمّت الإشارة إلى هذه الحالة لمراجعة طبيب بشري.",
        "pt": " Este caso foi sinalizado para revisão por um clínico humano.",
        "zh": " 此病例已被标记，供医护人员进行人工审核。",
        "ta": " இந்த வழக்கு மனித மருத்துவரின் மதிப்பாய்வுக்காக கொடியப்பட்டுள்ளது.",
    }

    # -------------------------------------------------------------------

    def phrase_followups(
        self,
        condition: str,
        follow_up_questions: list,
        patient_text: str,
        language: str = "en",
    ) -> str:
        lang = language.lower()[:2]
        openers = self._EMPATHY_OPENERS.get(lang, self._EMPATHY_OPENERS["en"])
        opener = random.Random(len(patient_text)).choice(openers)

        if not follow_up_questions:
            msg = self._TELL_MORE.get(lang, self._TELL_MORE["en"])
            return f"{opener} {msg}"

        lead_tpl = self._FOLLOWUP_LEAD.get(lang, self._FOLLOWUP_LEAD["en"])
        lead = lead_tpl.format(condition=condition.lower())
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(follow_up_questions[:3]))
        return f"{opener} {lead}\n{numbered}"

    def phrase_summary(self, decision: dict, language: str = "en") -> str:
        lang = language.lower()[:2]
        urgency = decision["urgency_level"]
        dept    = decision["recommended_department"]
        rule    = decision["rule_applied"]
        esc     = self._ESCALATION.get(lang, self._ESCALATION["en"]) if decision["escalation_required"] else ""

        tpl = self._SUMMARY.get(lang, self._SUMMARY["en"])
        return tpl.format(urgency=urgency, dept=dept, rule=rule, esc=esc)


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
