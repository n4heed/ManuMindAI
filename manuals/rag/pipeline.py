"""Ties retrieval + generation together and persists the Question/Citation records."""

from __future__ import annotations

import logging
import time

from django.contrib.auth.models import AbstractUser

from ..models import Citation, GroqCitation, GroqClaim, Machine, Question
from ..services import RetrievedChunk, retrieve_chunks
from .groq_provider import GroqAnswer, GroqConfigurationError, GroqProviderError, get_groq_provider
from .providers import get_llm_provider

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    """Conservative normalization for exact-quote matching: collapses any
    run of whitespace to a single space and strips the ends. Does not fix
    wording, punctuation, or case - a quote must still be a real substring."""
    return " ".join(text.split())

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available manuals for this machine do not contain enough information to answer this question."
)


def answer_question(user: AbstractUser, machine: Machine, question_text: str) -> Question:
    # Retrieval happens exactly once here. Both the RAG-only generator below
    # and the Groq generator (_attach_groq_answer) are given this same
    # `retrieved` list - neither one ever calls retrieve_chunks() itself.
    retrieved = retrieve_chunks(machine, question_text)

    if not retrieved:
        # Insufficient evidence: neither generator runs. Groq is never
        # called, and its status/answer reflect the same abstention as the
        # RAG-only path - not a separate "unavailable" or an attempted call.
        question = Question.objects.create(
            user=user,
            machine=machine,
            question_text=question_text,
            answer_text=INSUFFICIENT_EVIDENCE_MESSAGE,
            status=Question.Status.INSUFFICIENT_EVIDENCE,
            llm_provider="none",
            groq_status=Question.GroqStatus.INSUFFICIENT_EVIDENCE,
            groq_answer_text=INSUFFICIENT_EVIDENCE_MESSAGE,
        )
        return question

    llm = get_llm_provider()
    context_chunks = [
        {
            "text": r.chunk.text,
            "document_id": r.chunk.manual.document_id,
            "page_number": r.chunk.page.page_number,
        }
        for r in retrieved
    ]
    answer_text = llm.generate(question_text, context_chunks)

    question = Question.objects.create(
        user=user,
        machine=machine,
        question_text=question_text,
        answer_text=answer_text,
        status=Question.Status.ANSWERED,
        llm_provider=llm.name,
    )
    Citation.objects.bulk_create(
        [
            Citation(
                question=question,
                manual=r.chunk.manual,
                page_number=r.chunk.page.page_number,
                excerpt=r.chunk.text[:400],
                score=r.score,
            )
            for r in retrieved
        ]
    )

    # The RAG-only answer above is fully created and saved before this call.
    # Whatever happens inside _attach_groq_answer - disabled, misconfigured,
    # a network/API failure, an unexpected bug - it can only ever update
    # this question's groq_* fields afterwards; it can never undo or block
    # the RAG-only answer already persisted above.
    _attach_groq_answer(question, question_text, retrieved)

    return question


def _attach_groq_answer(question: Question, question_text: str, retrieved: list[RetrievedChunk]) -> None:
    """
    Builds [S#] source identifiers from the SAME retrieved chunks used for
    the RAG-only answer (no second retrieval), calls Groq, validates every
    identifier Groq's response cites against that exact set, and persists
    only citations the backend itself resolved to a real Manual/page - never
    an identifier the model invented.
    """
    source_lookup: dict[str, RetrievedChunk] = {}
    sources_payload = []
    for i, r in enumerate(retrieved):
        sid = f"S{i + 1}"
        source_lookup[sid] = r
        sources_payload.append(
            {
                "id": sid,
                "document_id": r.chunk.manual.document_id,
                "page_number": r.chunk.page.page_number,
                "text": r.chunk.text,
            }
        )

    start = time.monotonic()
    try:
        provider = get_groq_provider()
    except GroqConfigurationError as exc:
        # No API call was made, and none will be retried - this is a
        # configuration state, not a transient failure.
        question.groq_status = Question.GroqStatus.UNAVAILABLE
        question.groq_error_message = str(exc)
        question.save(update_fields=["groq_status", "groq_error_message"])
        return

    def _fail(message: str, exc_info: bool = False) -> None:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if exc_info:
            logger.exception("Groq generation failed for question %s", question.id)
        else:
            logger.warning("Groq generation failed for question %s: %s", question.id, message)
        question.groq_status = Question.GroqStatus.FAILED
        question.groq_model = provider.model
        question.groq_error_message = message
        question.groq_generation_time_ms = elapsed_ms
        question.save(
            update_fields=["groq_status", "groq_model", "groq_error_message", "groq_generation_time_ms"]
        )

    try:
        result = provider.generate(question_text, sources_payload)
    except GroqProviderError as exc:
        _fail(str(exc))
        return
    except Exception:  # noqa: BLE001 - a bug here must never break the RAG-only answer
        _fail("An unexpected error occurred generating the Groq answer.", exc_info=True)
        return

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if result.status == "insufficient_evidence" or not result.claims:
        # The model itself judged the evidence insufficient (distinct from
        # the "no retrieval at all" case above, which never calls Groq).
        question.groq_status = Question.GroqStatus.INSUFFICIENT_EVIDENCE
        question.groq_answer_text = INSUFFICIENT_EVIDENCE_MESSAGE
        question.groq_model = provider.model
        question.groq_error_message = ""
        question.groq_generation_time_ms = elapsed_ms
        question.save(
            update_fields=[
                "groq_status", "groq_answer_text", "groq_model", "groq_error_message", "groq_generation_time_ms",
            ]
        )
        return

    validated_claims = _validate_claims(result, source_lookup, question.machine_id)

    if not validated_claims:
        # Citation correctness is mandatory: an answer where no claim has a
        # real, quote-verified supporting source is not presented as cited.
        question.groq_status = Question.GroqStatus.FAILED
        question.groq_model = provider.model
        question.groq_error_message = "Groq's response did not reference any valid, quote-verified supporting citation."
        question.groq_generation_time_ms = elapsed_ms
        question.save(
            update_fields=["groq_status", "groq_model", "groq_error_message", "groq_generation_time_ms"]
        )
        return

    question.groq_status = Question.GroqStatus.COMPLETED
    question.groq_answer_text = " ".join(claim_text for claim_text, _ in validated_claims)
    question.groq_model = provider.model
    question.groq_error_message = ""
    question.groq_generation_time_ms = elapsed_ms
    question.save(
        update_fields=[
            "groq_status",
            "groq_answer_text",
            "groq_model",
            "groq_error_message",
            "groq_generation_time_ms",
        ]
    )
    for order, (claim_text, valid_sources) in enumerate(validated_claims):
        claim = GroqClaim.objects.create(question=question, order=order, text=claim_text)
        GroqCitation.objects.bulk_create(
            [
                GroqCitation(
                    question=question,
                    claim=claim,
                    manual=r.chunk.manual,
                    page_number=r.chunk.page.page_number,
                    excerpt=r.chunk.text[:400],
                    quote=quote,
                    score=r.score,
                    source_id=sid,
                )
                for sid, r, quote in valid_sources
            ]
        )


def _validate_claims(
    result: GroqAnswer,
    source_lookup: dict[str, RetrievedChunk],
    machine_id: int,
) -> list[tuple[str, list[tuple[str, RetrievedChunk, str]]]]:
    """
    For every claim Groq returned, keep only the sources that: (1) use a
    source_id we actually sent, (2) belong to the question's own machine
    (defense in depth - source_lookup is already machine-scoped by
    construction, since it is built only from `retrieved`, but this is
    checked explicitly rather than assumed), and (3) give a supporting
    quotation that is a real, whitespace-normalized substring of that
    source's actual chunk text - never trusting a quotation Groq invented.
    A claim with zero surviving sources is dropped entirely, never shown
    unsupported. Returns [(claim_text, [(source_id, chunk, quote), ...]), ...].
    """
    validated: list[tuple[str, list[tuple[str, RetrievedChunk, str]]]] = []
    for claim in result.claims:
        if not claim.text or not claim.text.strip():
            continue
        valid_sources: list[tuple[str, RetrievedChunk, str]] = []
        for source in claim.sources:
            sid = source.get("source_id")
            quote = source.get("supporting_quote") or ""
            r = source_lookup.get(sid)
            if r is None:
                continue  # unknown source identifier - never trusted
            if r.chunk.machine_id != machine_id:
                continue  # defense in depth: source must belong to the selected machine
            if not quote.strip():
                continue
            if _normalize_whitespace(quote) not in _normalize_whitespace(r.chunk.text):
                continue  # fabricated/altered quotation - not an actual substring of the source
            valid_sources.append((sid, r, quote))
        if valid_sources:
            validated.append((claim.text.strip(), valid_sources))
    return validated
