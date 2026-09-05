"""
rag_pipeline.py
---------------
Local-only RAG pipeline: chunks triage_rules.json into retrievable documents,
embeds them with TF-IDF (scikit-learn), stores vectors in-memory (numpy), and
retrieves the top-k most relevant rule documents for a given query via cosine
similarity. No external vector DB, no hosted embedding service.

This module is intentionally decoupled from the actual triage DECISION
(see triage_engine.py) — it exists purely to ground the LLM's natural
language output (follow-up questions, plain-English explanations) in the
real rule text, and to give the UI a "Retrieved Context" trace so every
answer is auditable back to source documents.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RULES_PATH = os.path.join(os.path.dirname(__file__), "triage_rules.json")


@dataclass
class Document:
    doc_id: str
    doc_type: str
    content: str


class RAGPipeline:
    def __init__(self, rules_path: str = RULES_PATH):
        self.documents: list[Document] = self._load_documents(rules_path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [d.content for d in self.documents]
        # Precompute all vectors once at startup (not per-request) to meet
        # the latency budget.
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def _load_documents(self, rules_path: str) -> list:
        with open(rules_path, "r") as f:
            data = json.load(f)

        docs = []
        for rule in data["rules"]:
            content = (
                f"TRIAGE RULE {rule['id']}: {rule['condition']}\n"
                f"Urgency: {rule['urgency']} | Department: {rule['department']}\n"
                f"Criteria: {rule['criteria'].get('primary', '')}. "
                f"{', '.join(rule['criteria'].get('secondary', []))}\n"
                f"Description: {rule['description']}\n"
                f"Follow-up questions: {' | '.join(rule.get('follow_ups', []))}\n"
                f"Red flags requiring escalation: {', '.join(rule.get('red_flags', []))}"
            )
            docs.append(Document(doc_id=rule["id"], doc_type="triage_rule", content=content))

        nc = data["null_case"]
        docs.append(Document(
            doc_id=nc["id"],
            doc_type="null_case",
            content=(
                f"NULL CASE {nc['id']}: {nc['condition']}\n"
                f"Description: {nc['description']}\n"
                f"Action: {nc['action']} | Department: {nc['department']} | Urgency: {nc['urgency']}"
            ),
        ))
        return docs

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return the top_k documents most similar to query, each with a score."""
        if not query or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in ranked_idx:
            if sims[i] <= 0:
                continue
            results.append({
                "doc_id": self.documents[i].doc_id,
                "doc_type": self.documents[i].doc_type,
                "content": self.documents[i].content,
                "score": round(float(sims[i]), 4),
            })
        return results

    def retrieve_as_text(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No closely matching triage rules found in the knowledge base."
        blocks = [f"[{r['doc_id']} | relevance {r['score']}]\n{r['content']}" for r in results]
        return "\n\n".join(blocks)
