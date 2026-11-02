"""
PDF ingestion and retrieval services.

Ingestion: PDF file -> ManualPage rows (1-based page numbers preserved) ->
Chunk rows with embeddings, scoped to the manual's machine.

Retrieval: question text -> embed -> cosine similarity against Chunk rows
for ONE machine only -> top-K chunks above RAG_SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

import os
import re
import tempfile
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


_MACHINE_LABEL_RE = re.compile(r"Machine:\s*([^\n|]+)", re.IGNORECASE)


def check_machine_mismatch(uploaded_file, selected_machine: Machine) -> str | None:
    """
    Advisory-only safeguard for the Administrator upload form: looks at just
    the PDF's first page and checks whether it seems to belong to a
    different registered machine than the one selected. Never blocks, never
    reassigns anything itself - returns a human-readable warning string, or
    None if nothing looks wrong (including whenever the check itself can't
    be completed, e.g. PyMuPDF unavailable, encrypted/corrupt file - a
    failed *check* must never block a real upload that would otherwise
    succeed or fail on its own merits).

    Two-tier detection:
    1. If the page has an explicit "Machine: <id>" label (as this corpus's
       manuals do), compare it directly - high precision.
    2. Otherwise, fall back to a soft heuristic: does the page mention a
       *different* registered machine's code while never mentioning the
       selected one at all.
    """
    try:
        import fitz
    except ImportError:
        return None

    tmp_path = tempfile.mktemp(suffix=".pdf")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        try:
            doc = fitz.open(tmp_path)
        except Exception:
            return None
        try:
            if doc.is_encrypted or doc.page_count == 0:
                return None
            first_page_text = doc.load_page(0).get_text("text")
        finally:
            doc.close()
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    selected_code = selected_machine.machine_id.upper()
    label_match = _MACHINE_LABEL_RE.search(first_page_text)
    if label_match:
        declared = label_match.group(1).strip()
        if selected_code not in declared.upper():
            return (
                f'This document\'s first page declares "Machine: {declared}", which does not '
                f"appear to match the selected machine ({selected_machine.machine_id}). "
                "Confirm this is the correct machine before uploading."
            )
        return None

    text_upper = first_page_text.upper()
    if selected_code in text_upper:
        return None
    others = [
        m.machine_id
        for m in Machine.objects.exclude(id=selected_machine.id)
        if m.machine_id.upper() in text_upper
    ]
    if others:
        return (
            f"This document's first page mentions {', '.join(others)} but not the selected "
            f"machine ({selected_machine.machine_id}). Confirm this is the correct machine "
            "before uploading."
        )
    return None


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


_ALARM_CODE_RE = re.compile(r"\b[Ee]\d{1,3}\b")
_FILTER_CODE_RE = re.compile(r"\b([A-Z]{2}-\d{1,3})\b")
_RECORD_ESCALATION_RE = re.compile(r"record[\s\S]{0,60}escalat|escalat[\s\S]{0,60}record", re.IGNORECASE)


def generate_recommended_questions(machine: Machine, limit: int = 6) -> list[str]:
    """
    Deterministic, LLM-free question suggestions built only from this
    machine's own successfully-indexed chunks - the exact same trusted,
    machine-scoped query (`Chunk.objects.filter(machine=machine)`) that
    retrieval itself uses, so a suggestion can never be derived from
    another machine's manuals. Never calls Groq - this must not add LLM
    cost or latency to a dashboard load. Returns [] if nothing suitable
    was found in the indexed text, rather than padding the list with
    generic, unsupported questions.

    Pattern-based, not content-generating: every rule here only turns an
    already-present textual pattern (an alarm code, a filter identifier, a
    recognized section topic) into a natural question to *ask* - it never
    asserts a fact of its own, so there is nothing for it to invent.
    """
    texts = list(Chunk.objects.filter(machine=machine).values_list("text", flat=True))
    if not texts:
        return []
    full_text = "\n".join(texts)
    # Document IDs (e.g. "HP-MNT-01") share the same [A-Z]+-\d+ shape as real
    # filter/part codes (e.g. "AF-10") and appear on nearly every page as a
    # header, often near the word "filter" purely by chunking coincidence -
    # exclude anything that's actually a fragment of this machine's own
    # document IDs so a heading never gets mistaken for a part code.
    document_ids_upper = {
        d.upper() for d in Manual.objects.filter(machine=machine).values_list("document_id", flat=True)
    }
    # The machine's own id/model (e.g. "AC-90") also match the same code
    # shape and turn up next to "filter" in cross-reference sentences like
    # "The HP-200 RF-20 filter is not the AC-90 filter" - exclude those too,
    # alongside document-ID fragments, so only a genuine part code is used.
    excluded_codes = document_ids_upper | {machine.machine_id.upper(), machine.model.upper()}

    questions: list[str] = []

    alarm_codes: list[str] = []
    for code in _ALARM_CODE_RE.findall(full_text):
        code = code.upper()
        if code not in alarm_codes:
            alarm_codes.append(code)
    for code in alarm_codes[:2]:
        questions.append(f"What does {code} indicate?")

    for chunk_text in texts:
        lowered = chunk_text.lower()
        if "filter" not in lowered:
            continue
        found_code = None
        for match in _FILTER_CODE_RE.finditer(chunk_text):
            code = match.group(1)
            if any(code.upper() in excluded for excluded in excluded_codes):
                continue  # a document-ID or machine-model fragment, not a real part code
            # Only trust a code that actually appears close to the word
            # "filter" in the text, not merely somewhere in the same chunk.
            window_start = max(0, match.start() - 25)
            window_end = min(len(chunk_text), match.end() + 25)
            if "filter" not in chunk_text[window_start:window_end].lower():
                continue
            found_code = code
            break
        if found_code:
            question = f"How should the {found_code} filter be inspected?"
            if question not in questions:
                questions.append(question)
            break

    if re.search(r"before each shift|inspection schedule", full_text, re.IGNORECASE):
        questions.append("What maintenance checks are required before each shift?")

    if _RECORD_ESCALATION_RE.search(full_text):
        questions.append("What should be recorded before escalation?")

    if re.search(r"\bREADY\b", full_text) and re.search(r"\bHOLD\b", full_text):
        questions.append("What do the machine's operating states mean?")

    if re.search(r"troubleshooting", full_text, re.IGNORECASE):
        questions.append("What troubleshooting steps are documented for this machine?")

    if re.search(r"maintenance schedule", full_text, re.IGNORECASE):
        questions.append("What is the maintenance schedule for this machine?")

    seen: set[str] = set()
    deduped: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:limit]
