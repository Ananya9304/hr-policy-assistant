
import logging
import re

from pydantic import ValidationError

from app import config
from app.llm import SYSTEM_PROMPT, LLMClient, build_user_prompt
from app.models import (AskResponse, Citation, LLMAnswer, RefusalReason, RetrievedChunk, Status)
from app.retrieval import Retriever

log = logging.getLogger("rag")

REFUSAL_TEXT = ("I don't have enough information in the uploaded HR policies to answer that. "
                f"Please contact {config.HR_CONTACT}.")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower().replace("’", "'"))


def quote_is_supported(quote: str, source_text: str) -> bool:
    """True if `quote` really comes from `source_text`.

    Exact match after normalising case/punctuation, or (for minor LLM copy noise)
    at least 90% of the quote's words appear in the source and every NUMBER in
    the quote appears in the source. Numbers are what employees act on, so we
    never accept a quote whose numbers we cannot find.
    """
    q, s = _words(quote), _words(source_text)
    if len(q) < 2:
        return False
    if " ".join(q) in " ".join(s):
        return True
    s_set = set(s)
    numbers_ok = all(w in s_set for w in q if any(ch.isdigit() for ch in w))
    overlap = sum(w in s_set for w in q) / len(q)
    return numbers_ok and overlap >= 0.9


_TYPOGRAPHY = str.maketrans({"\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ",
                             "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2019": "'"})


def clean_text(text: str) -> str:
    return text.translate(_TYPOGRAPHY).strip()


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def answer_numbers_supported(answer: str, cited_texts: list[str], question: str = "") -> bool:
    """Every number in the answer (days, rupees, limits) must appear in a cited chunk
    (body or section heading, e.g. "§4.1") or in the question itself ("for a 5 day leave")."""
    allowed = set().union(_numbers(question), *(_numbers(t) for t in cited_texts))
    return _numbers(answer) <= allowed


def refuse(reason: RefusalReason, top_score: float | None = None, text: str = REFUSAL_TEXT) -> AskResponse:
    return AskResponse(status=Status.refused, answer=text, citations=[], reason=reason, top_score=top_score)


class RAGService:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def ask(self, question: str) -> AskResponse:
        question = (question or "").strip()
        if not question:
            return refuse(RefusalReason.empty_query, text="Please type a question about an HR policy.")

        hits = self.retriever.search(question)
        if not hits:
            return refuse(RefusalReason.no_documents,
                          text="No HR policies have been uploaded yet. Please contact " + config.HR_CONTACT + ".")

       
        top_score = max(h.dense_score for h in hits)
        if top_score < config.MIN_DENSE_SCORE:
            log.info("gate refused q=%r top=%.3f", question, top_score)
            return refuse(RefusalReason.low_retrieval_score, round(top_score, 3))

      
        labelled = {f"S{i + 1}": h for i, h in enumerate(hits)}
        prompt = build_user_prompt(
            question, [(label, f"{h.chunk.document} > {h.chunk.section_path}", h.chunk.text)
                       for label, h in labelled.items()])
        parsed = self._call_llm(prompt)
        if parsed is None:
            return AskResponse(status=Status.error, reason=RefusalReason.llm_unavailable, top_score=round(top_score, 3),
                               answer="The assistant is temporarily unavailable. Please try again, or contact "
                                      + config.HR_CONTACT + ".")
        if not parsed.answerable or not parsed.answer.strip():
            return refuse(RefusalReason.model_declined, round(top_score, 3))

      
        citations, cited_texts = self._verified_citations(parsed, labelled)
        if not citations or not answer_numbers_supported(parsed.answer, cited_texts, question):
            log.warning("verification failed q=%r raw=%s", question, parsed.model_dump())
            return refuse(RefusalReason.unverified_citations, round(top_score, 3))

        return AskResponse(status=Status.answered, answer=clean_text(parsed.answer),
                           citations=citations, top_score=round(top_score, 3))

    def _call_llm(self, prompt: str) -> LLMAnswer | None:
        for attempt in range(2):          # one retry for malformed JSON
            try:
                return LLMAnswer.model_validate(self.llm.complete_json(SYSTEM_PROMPT, prompt))
            except (ValueError, ValidationError) as exc:       # bad JSON / wrong shape
                log.warning("LLM output invalid (attempt %d): %s", attempt + 1, exc)
            except Exception as exc:                            # network, quota, auth ...
                log.error("LLM call failed: %s", exc)
                return None
        return None

    @staticmethod
    def _verified_citations(parsed: LLMAnswer, labelled: dict[str, RetrievedChunk]):
        out, texts, seen = [], [], set()
        for c in parsed.citations:
            hit = labelled.get(c.source_id.strip().strip("[]"))
            if hit is None or not quote_is_supported(c.quote, hit.chunk.text):
                continue
            key = (hit.chunk.chunk_id, c.quote)
            if key in seen:
                continue
            seen.add(key)
            texts.append(f"{hit.chunk.section_path}\n{hit.chunk.text}")
            out.append(Citation(document=hit.chunk.document, filename=hit.chunk.filename,
                                section=hit.chunk.section, section_path=hit.chunk.section_path,
                                quote=clean_text(c.quote)))
        return out, texts
