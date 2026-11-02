"""Ties retrieval + generation together and persists the Question/Citation records."""

from __future__ import annotations

from django.contrib.auth.models import AbstractUser

from ..models import Citation, Machine, Question
from ..services import retrieve_chunks
from .providers import get_llm_provider

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available manuals for this machine do not contain enough information to answer this question."
)


def answer_question(user: AbstractUser, machine: Machine, question_text: str) -> Question:
    retrieved = retrieve_chunks(machine, question_text)

    if not retrieved:
        question = Question.objects.create(
            user=user,
            machine=machine,
            question_text=question_text,
            answer_text=INSUFFICIENT_EVIDENCE_MESSAGE,
            status=Question.Status.INSUFFICIENT_EVIDENCE,
            llm_provider="none",
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
    return question
