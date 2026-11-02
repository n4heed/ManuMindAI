"""
PDF ingestion and retrieval services.

Ingestion: PDF file -> ManualPage rows (1-based page numbers preserved) ->
Chunk rows with embeddings, scoped to the manual's machine.

Retrieval: question text -> embed -> cosine similarity against Chunk rows
for ONE machine only -> top-K chunks above RAG_SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Chunk, Machine, Manual, ManualPage
from .rag.providers import cosine_similarity, get_embedding_provider

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024
CHUNK_TARGET_CHARS = 600
CHUNK_OVERLAP_CHARS = 100


class IngestionError(Exception):
    """Raised for problems the caller should show to the uploading Administrator."""


def validate_pdf_upload(uploaded_file) -> None:
    name = uploaded_file.name or ""
    if not name.lower().endswith(".pdf"):
        raise IngestionError("Only .pdf files are accepted.")
    if uploaded_file.size == 0:
        raise IngestionError("The uploaded file is empty.")
    if uploaded_file.size > MAX_PDF_SIZE_BYTES:
        raise IngestionError("The uploaded file exceeds the 25 MB limit.")


def extract_pages(file_path: str) -> list[tuple[int, str]]:
    """Return [(one_based_page_number, text), ...]. Raises IngestionError on bad PDFs."""
    try:
        import fitz  # PyMuPDF - imported lazily so the rest of the app still
        # works (server, API, UI, non-ingestion tests) in an environment where
        # PyMuPDF's native extension can't load (e.g. blocked by a Windows
        # Application Control / Smart App Control policy). See README
        # "Known blockers" for details and how to fix the environment.
    except ImportError as exc:
        raise IngestionError(f"PyMuPDF (fitz) is not available in this environment: {exc}") from exc

    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # malformed / not a real PDF
        raise IngestionError(f"Could not open PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            raise IngestionError("The PDF is password-protected/encrypted; encrypted PDFs are not supported.")
        if doc.page_count == 0:
            raise IngestionError("The PDF has no pages.")

        pages: list[tuple[int, str]] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text").strip()
            pages.append((index + 1, text))  # 1-based page number

        if all(not text for _, text in pages):
            raise IngestionError(
                "No extractable text was found on any page. "
                "This looks like a scanned/image-only PDF, which is not supported (OCR is out of scope)."
            )
        return pages
    finally:
        doc.close()


def chunk_page_text(text: str) -> list[str]:
    """Split one page's text into overlapping chunks on paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= CHUNK_TARGET_CHARS:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= CHUNK_TARGET_CHARS:
            buffer = paragraph
        else:
            # Paragraph itself is long; split on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            buffer = ""
            for sentence in sentences:
                candidate = f"{buffer} {sentence}".strip() if buffer else sentence
                if len(candidate) <= CHUNK_TARGET_CHARS:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = sentence
    if buffer:
        chunks.append(buffer)

    # Add small overlap between consecutive chunks so evidence near a boundary isn't lost.
    # Trim to a word boundary so the overlap doesn't start mid-word (e.g. "nt filter...").
    if len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-CHUNK_OVERLAP_CHARS:]
            tail = tail.split(" ", 1)[-1] if " " in tail else tail
            overlapped.append(f"{tail} {chunks[i]}".strip())
        chunks = overlapped

    return chunks


@transaction.atomic
def ingest_manual(manual: Manual) -> None:
    """
    Extract text, chunk it, embed it, and persist ManualPage/Chunk rows for
    `manual`. Updates manual.status/error_message/page_count/processed_at.
    Never raises on expected failure modes - records them on the manual instead,
    so the Administrator sees a clear status rather than a 500 error.
    """
    manual.status = Manual.Status.PROCESSING
    manual.error_message = ""
    manual.save(update_fields=["status", "error_message"])

    try:
        pages = extract_pages(manual.file.path)
    except IngestionError as exc:
        # A failed (re-)ingestion must not leave this manual's *previous*
        # successful extraction looking current: drop its old pages/chunks
        # and page_count, so a "failed" manual never has stale content that
        # could still be silently retrieved. Found via a real-PDF test.
        manual.pages.all().delete()
        manual.status = Manual.Status.FAILED
        manual.error_message = str(exc)
        manual.page_count = 0
        manual.processed_at = None
        manual.save(update_fields=["status", "error_message", "page_count", "processed_at"])
        return

    manual.pages.all().delete()  # re-ingestion: replace, don't duplicate
    embedder = get_embedding_provider()

    page_objs = []
    all_chunk_texts = []
    chunk_owner_pages = []
    for page_number, text in pages:
        page_obj = ManualPage.objects.create(manual=manual, page_number=page_number, text=text)
        page_objs.append(page_obj)
        for chunk_text in chunk_page_text(text):
            all_chunk_texts.append(chunk_text)
            chunk_owner_pages.append(page_obj)

    if all_chunk_texts:
        vectors = embedder.embed(all_chunk_texts)
        chunk_objs = []
        per_page_index: dict[int, int] = {}
        for chunk_text, page_obj, vector in zip(all_chunk_texts, chunk_owner_pages, vectors):
            idx = per_page_index.get(page_obj.id, 0)
            per_page_index[page_obj.id] = idx + 1
            chunk_objs.append(
                Chunk(
                    manual=manual,
                    machine=manual.machine,
                    page=page_obj,
                    chunk_index=idx,
                    text=chunk_text,
                    embedding=vector,
                    embedding_provider=embedder.name,
                )
            )
        Chunk.objects.bulk_create(chunk_objs)

    manual.status = Manual.Status.COMPLETED
    manual.page_count = len(pages)
    manual.processed_at = timezone.now()
    manual.save(update_fields=["status", "page_count", "processed_at"])


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve_chunks(machine: Machine, question_text: str) -> list[RetrievedChunk]:
    """Retrieve the top-K chunks for ONE machine only, above the similarity threshold.

    At most one chunk per (manual, page) is kept - a page longer than
    CHUNK_TARGET_CHARS is split into multiple chunks with a deliberate
    character overlap (see chunk_page_text), so two chunks from the same
    page are frequently near-duplicates of each other. Without this, a
    single relevant page could show up as two or three almost-identical
    citations. Keeping only the best-scoring chunk per page trades away the
    rare case of one long page holding two genuinely distinct answers, which
    this hackathon's 3-page manuals don't exhibit.
    """
    embedder = get_embedding_provider()
    [question_vector] = embedder.embed([question_text])

    candidates = Chunk.objects.filter(machine=machine, embedding_provider=embedder.name).select_related(
        "manual", "page"
    )
    scored = [
        RetrievedChunk(chunk=chunk, score=cosine_similarity(question_vector, chunk.embedding))
        for chunk in candidates
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    above_threshold = [r for r in scored if r.score >= settings.RAG_SIMILARITY_THRESHOLD]

    deduped: list[RetrievedChunk] = []
    seen_pages: set[tuple[int, int]] = set()
    for r in above_threshold:
        page_key = (r.chunk.manual_id, r.chunk.page_id)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        deduped.append(r)

    return deduped[: settings.RAG_TOP_K]
