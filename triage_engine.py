"""
triage_engine.py
-----------------
Deterministic, rule-driven triage decision engine.

This is the safety-critical core of the system. It does NOT depend on an LLM
to decide urgency/department/escalation — those decisions come from explicit,
auditable keyword + criteria matching against triage_rules.json. The LLM layer
(llm_provider.py) is only used for natural-language follow-up question
generation and for turning the final decision into human-readable prose.

Why deterministic instead of "ask the LLM to decide"?
  - Reproducible: same input -> same urgency level, every time.
  - Auditable: every decision names the exact rule id that fired.
  - Safe by default: if nothing matches confidently, we fall back to the
    UNCERTAIN rule and escalate, rather than guessing.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


RULES_PATH = os.path.join(os.path.dirname(__file__), "triage_rules.json")

# Urgency ranking used to pick the single most severe matching rule when
# several rules match at once (e.g. "chest pain" + "vomiting").
URGENCY_RANK = {"Critical": 4, "High": 3, "Moderate": 2, "Low": 1, "Routine": 0}


@dataclass
class RuleMatch:
    rule_id: str
    condition: str
    urgency: str
    department: str
    description: str
    follow_ups: list
    red_flags: list
    escalate: bool
    matched_keywords: list = field(default_factory=list)
    matched_red_flags: list = field(default_factory=list)
    score: float = 0.0


class TriageEngine:
    def __init__(self, rules_path: str = RULES_PATH):
        with open(rules_path, "r") as f:
            data = json.load(f)
        self.rules = data["rules"]
        self.null_case = data["null_case"]
        # Precompute keyword sets per rule for fast matching.
        self._keyword_map = self._build_keyword_map()

    # ------------------------------------------------------------------ #
    # Keyword extraction
    # ------------------------------------------------------------------ #
    def _build_keyword_map(self):
        """
        Derive matchable keywords/phrases for each rule from its condition
        name and criteria, plus a small hand-curated synonym table. This
        keeps the rule file itself the single source of truth while still
        giving reasonable recall on everyday patient language.
        """
        synonym_table = {
            "FEVER_001": ["fever", "temperature", "high temp", "chills", "sweats", "hot and cold"],
            "CHEST_PAIN_001": ["chest pain", "chest pressure", "chest tightness", "chest hurts",
                                "pain in my chest", "pressure in my chest"],
            "BREATHING_001": ["difficulty breathing", "can't breathe", "cant breathe", "shortness of breath",
                               "trouble breathing", "wheezing", "gasping", "not getting enough air",
                               "can't catch my breath", "cant catch my breath"],
            "INJURY_001": ["fell", "fall", "accident", "injury", "injured", "broke", "broken",
                            "sprain", "twisted", "hit my", "hurt my arm", "hurt my leg", "swollen",
                            "bicycle", "car accident"],
            "ABDOMINAL_PAIN_001": ["stomach", "abdomen", "abdominal", "belly", "gut pain",
                                     "stomach hurts", "vomit", "vomiting", "threw up", "nausea"],
            "MILD_SYMPTOMS_001": ["mild", "slight", "minor", "small cut", "little pain", "not too bad",
                                    "small scrape", "little scratch", "minor cut", "small bruise"],
            "UNCERTAIN_001": [],  # fallback only, no direct keywords
        }
        keyword_map = {}
        for rule in self.rules:
            rid = rule["id"]
            words = set()
            words.add(rule["condition"].lower())
            words.update(w.lower() for w in synonym_table.get(rid, []))
            keyword_map[rid] = sorted(words)
        return keyword_map

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    NEGATION_WINDOW_CHARS = 12  # how far back before a phrase we check for a negator
    NEGATORS = ("no ", "not ", "denies ", "denied ", "without ", "never had ", "don't have ",
                "dont have ", "doesn't have ", "doesnt have ")

    def _is_negated(self, text: str, phrase: str) -> bool:
        """
        Cheap negation check: look at the text immediately preceding the
        matched phrase for a negator word (e.g. "no fever", "denies chest
        pain"). This is a heuristic, not full NLP negation scope resolution
        — it exists to stop the most common false positive ("no fever that
        I know of" incorrectly triggering the FEVER rule), not to be a
        clinical-grade negation detector. Genuinely ambiguous or negated-but-
        still-relevant statements are exactly the kind of case that should
        fall through to UNCERTAIN_001 and get escalated for human review.
        """
        idx = text.find(phrase)
        if idx == -1:
            return False
        window_start = max(0, idx - self.NEGATION_WINDOW_CHARS - 1)
        preceding = text[window_start:idx]
        return any(neg in preceding for neg in self.NEGATORS)

    def _text_has_any(self, text: str, phrases: list) -> list:
        hits = []
        for p in phrases:
            if p and p in text and not self._is_negated(text, p):
                hits.append(p)
        return hits

    def evaluate(self, full_text: str) -> RuleMatch:
        """
        Evaluate the full accumulated patient text (initial complaint +
        all follow-up answers) against every rule. Returns the single
        highest-urgency matching rule, with red flags escalating regardless
        of which rule wins.
        """
        text = full_text.lower()
        candidates = []

        for rule in self.rules:
            rid = rule["id"]
            kw_hits = self._text_has_any(text, self._keyword_map.get(rid, []))
            rf_hits = self._text_has_any(text, [r.lower() for r in rule.get("red_flags", [])])

            if not kw_hits and not rf_hits:
                continue

            score = len(kw_hits) + 2 * len(rf_hits)  # red flags weigh more
            escalate = bool(rule.get("escalate", False)) or bool(rf_hits)

            candidates.append(RuleMatch(
                rule_id=rid,
                condition=rule["condition"],
                urgency=rule["urgency"],
                department=rule["department"],
                description=rule["description"],
                follow_ups=rule.get("follow_ups", []),
                red_flags=rule.get("red_flags", []),
                escalate=escalate,
                matched_keywords=kw_hits,
                matched_red_flags=rf_hits,
                score=score,
            ))

        if not candidates:
            # Nothing matched at all.
            if len(text.strip()) < 8 or self._is_no_complaint(text):
                nc = self.null_case
                return RuleMatch(
                    rule_id=nc["id"], condition=nc["condition"], urgency=nc["urgency"],
                    department=nc["department"], description=nc["description"],
                    follow_ups=["Can you tell me more about why you're here today?"],
                    red_flags=[], escalate=False,
                )
            return self._uncertain_match()

        # Pick highest urgency first, then highest score as tiebreak.
        candidates.sort(key=lambda c: (URGENCY_RANK.get(c.urgency, 0), c.score), reverse=True)
        best = candidates[0]

        # If multiple *different* conditions matched with meaningfully
        # different urgency/department, that's conflicting information —
        # force escalation even if the top rule alone wouldn't.
        distinct_conditions = {c.rule_id for c in candidates}
        if len(distinct_conditions) > 1:
            urgencies = {c.urgency for c in candidates}
            if len(urgencies) > 1:
                best.escalate = True

        return best

    def _uncertain_match(self) -> RuleMatch:
        rule = next(r for r in self.rules if r["id"] == "UNCERTAIN_001")
        return RuleMatch(
            rule_id=rule["id"], condition=rule["condition"], urgency=rule["urgency"],
            department=rule["department"], description=rule["description"],
            follow_ups=rule.get("follow_ups", []), red_flags=[],
            escalate=True,
        )

    @staticmethod
    def _is_no_complaint(text: str) -> bool:
        no_complaint_phrases = [
            "feeling fine", "just here for", "routine checkup", "no symptoms",
            "nothing wrong", "just a checkup", "annual checkup", "just here to",
        ]
        return any(p in text for p in no_complaint_phrases)

    # ------------------------------------------------------------------ #
    # Conflict detection (used to flag contradictory patient statements)
    # ------------------------------------------------------------------ #
    CONTRADICTION_PAIRS = [
        (r"\bsevere\b", r"\b(feel fine|feel calm|not that bad|no big deal|fine actually)\b"),
        (r"\b(can'?t breathe|difficulty breathing)\b", r"\b(breathing (is )?fine|breathing normally)\b"),
        (r"\b10/10|10 out of 10\b", r"\b(mild|barely hurts|not painful)\b"),
    ]

    def detect_conflict(self, full_text: str) -> Optional[str]:
        text = full_text.lower()
        for a, b in self.CONTRADICTION_PAIRS:
            if re.search(a, text) and re.search(b, text):
                return "Patient description contains contradictory severity statements."
        return None

    def all_rules(self):
        return self.rules
