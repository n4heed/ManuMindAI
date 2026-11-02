# Manufacturing Manual Assistant (RAG)

Phase 1 submission for the Round 1 Student Challenge: *Manufacturing Manual Assistant using Python Full Stack and RAG*.

A Django + DRF + PostgreSQL web app where an Administrator uploads machine manuals (PDF),
an Engineer asks maintenance questions about a selected machine and gets an answer grounded
in retrieved manual passages (with page citations), and a Supervisor reviews question history
read-only. Retrieval is strictly scoped to the selected machine.

## 1. Setup and launch

Requirements: Python 3.11 (the repo's `.venv` already uses it), PostgreSQL running locally.

```bash
cd Project
.venv\Scripts\python.exe -m pip install -r requirements.txt   # already installed in .venv
```

Copy `.env.example` to `.env` and fill in your local Postgres credentials (an app role/database
is already created on this dev machine as `manual_app` / database `manual_assistant` - see
below for the exact commands used).

```bash
.venv\Scripts\python.exe manage.py migrate
.venv\Scripts\python.exe manage.py seed_demo_users      # creates admin1/engineer1/supervisor1
.venv\Scripts\python.exe manage.py runserver
```

Open http://127.0.0.1:8000/ and log in with one of the demo accounts below.

### Demo credentials (local/synthetic - not real secrets)

| Username | Password | Role |
|---|---|---|
| admin1 | DemoPass123! | Administrator |
| engineer1 | DemoPass123! | Engineer |
| supervisor1 | DemoPass123! | Supervisor |

Django admin (`/admin/`) superuser: `superadmin` / `SuperDemo123!` - useful for inspecting raw
data (chunks, embeddings, citations) during Q&A with judges.

### Loading the official sample manuals quickly

```bash
.venv\Scripts\python.exe manage.py ingest_sample_manuals "<path to 'Round 1 Student Pack'>"
```

This registers `M-HP200` and `M-AC90` from `machine_manual_index.csv` and ingests all three
PDFs directly from disk (equivalent to doing it through the Administrator upload UI, just
faster for repeated local testing).

## 2. Architecture

```
Browser (Bootstrap + vanilla JS fetch)
   -> Django views (login, role-routed dashboard pages)
   -> Django REST Framework API (/api/...)
        -> manuals/services.py      (PDF extraction + chunking, PyMuPDF)
        -> manuals/rag/providers.py (pluggable embedding + LLM providers)
        -> manuals/rag/pipeline.py  (retrieve -> generate -> persist)
   -> PostgreSQL (machines, manuals, pages, chunks+embeddings, questions, citations)
```

- **Models** (`manuals/models.py`): `Profile` (role), `Machine`, `Manual`, `ManualPage`
  (1-based page number preserved), `Chunk` (text + embedding, scoped to a `machine`),
  `Question`, `Citation`.
- **Ingestion** (`manuals/services.py`): validates the upload, extracts text per page with
  PyMuPDF, chunks each page (~600 chars, paragraph/sentence-aware, with word-boundary-safe
  overlap), embeds each chunk, and stores everything tied to the manual's machine. Failures
  (encrypted PDF, no extractable text, corrupt file) set the manual to `failed` with a message
  instead of raising a 500.
- **Retrieval** (`manuals/services.py: retrieve_chunks`): embeds the question, computes cosine
  similarity only against `Chunk` rows for the **selected machine** (`Chunk.objects.filter(machine=...)`),
  keeps results above `RAG_SIMILARITY_THRESHOLD`, returns top `RAG_TOP_K`.
- **Generation** (`manuals/rag/pipeline.py`): if no chunk clears the threshold, the question is
  recorded as `insufficient_evidence` with no citations - no model is ever called with
  no-machine-match context. Otherwise the LLM provider is given only the retrieved
  machine-scoped chunks and must not use outside knowledge.
- **Roles**: `manuals/permissions.py` + DRF permission classes enforce, server-side, that only
  Administrators can create machines/upload manuals, only Engineers can ask questions, and
  Engineers only ever see their own question history while Supervisors/Administrators see all
  of it (read-only for Supervisors - no ask/upload endpoints are granted to that role).

## 3. RAG provider configuration (important - read this)

**No organizer-issued embedding/LLM credentials were available while building this**, per
`START_HERE.txt`: *"Model access and college rules for coding assistance are announced
separately. No credentials or solution code are included."* To keep the app fully functional
and testable regardless, `EMBEDDING_PROVIDER` / `LLM_PROVIDER` in `.env` select a provider
implementation (`manuals/rag/providers.py`):

- **`local` (current default)**: a dependency-free, deterministic fallback.
  - Embedding: exact-term sparse bag-of-words (stopword-filtered term frequency, L2-normalized,
    cosine similarity over shared tokens). No download, no network call, fully deterministic.
  - Generation: extractive - returns the retrieved sentences that share the most keywords with
    the question. **This is explicitly not an AI-generated answer.** Every `Question` row
    records `llm_provider`, and the UI shows a "source: local" line so this is never
    misrepresented as a real LLM's output.
- **`openai`**: `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation. Needs
  `OPENAI_API_KEY` in `.env`.
- **`anthropic`** (LLM only - Anthropic has no public embeddings API, so embeddings still need
  `EMBEDDING_PROVIDER=local` or `openai`): Claude for generation. Needs `ANTHROPIC_API_KEY`.

To switch once real access is issued: set `EMBEDDING_PROVIDER`/`LLM_PROVIDER` and the API key
in `.env`, restart the server. No other code changes needed - and existing `local`-embedded
chunks are automatically ignored by `retrieve_chunks` (it filters by `embedding_provider`), so
re-ingest manuals after switching providers rather than mixing embedding spaces.

### Retrieval threshold - how 0.12 was chosen

This was **measured, not guessed** (see git history / this report for the raw numbers). With
the hashed fixed-size embedding originally tried, unrelated passages scored *higher* than
related ones (hash collisions), which silently broke machine isolation and abstention. Switched
to exact-term sparse vectors and re-measured on realistic question/passage pairs from this
corpus:

- Genuinely related pairs: **0.17 - 0.67** cosine similarity.
- Unrelated pairs (different topic, different machine): **0.0** (no shared non-stopword terms).

`RAG_SIMILARITY_THRESHOLD=0.12` sits comfortably between those two clusters. It's a setting in
`.env`, not a magic number buried in code, so it can be re-tuned if a different embedding
provider changes the score distribution.

### What the citation "score" is - and is not

The number shown next to each citation (e.g. "score 0.43") is **cosine similarity between the
question's and the passage's term-frequency vectors** - a measure of *how much vocabulary the
question and the passage share*, normalized to [0, 1] for this provider (it can range further
negative/positive for other embedding providers, since cosine similarity's mathematical range is
[-1, 1]; this implementation's non-negative term-count vectors just never produce a negative
score). Concretely:

- It is **not** a probability or confidence level. A score of 0.43 does not mean "43% likely to
  be correct" or "43% confident" - there is no calibration step that would make that claim
  meaningful, and we have not built one.
- It is **not semantic similarity**. With the `local` provider (current default - see section 3),
  two passages about the same concept in different words (e.g. "pressure" vs. "PSI", or "the
  machine stopped" vs. "operation halted") score 0 if they share no literal terms. The light
  stemming added in this milestone only bridges exact word-form differences (e.g.
  "checked"/"check"), not synonyms or paraphrases.
- It **is** useful for relative ranking within one question: a higher score means more shared,
  meaningful vocabulary with the question, which is exactly what the threshold and top-K cutoff
  rely on. Treat it as "how lexically relevant", not "how right."
- Switching to a real embedding provider (`openai`) would produce an actual semantic-similarity
  score from a trained model, which is a materially different number with its own distribution -
  `RAG_SIMILARITY_THRESHOLD` would need to be re-measured for that provider rather than reused.

## 4. Known limitations (honest, as of this report)

- **PyMuPDF was blocked by Windows Smart App Control on the dev machine for part of this
  session** (`CodeIntegrity` event: *"did not meet the Enterprise signing level requirements"*).
  It resolved itself once Windows finished an async reputation check (confirmed working:
  successfully extracted and ingested all three official sample PDFs, 3 pages each). The `fitz`
  import is lazy (`manuals/services.py: extract_pages`) specifically so the rest of the app
  (server, API, UI, most tests) stays usable even if this blocks again on another machine;
  ingestion itself would show `status: failed` with a clear error rather than crashing.
- **`local` LLM provider output is extractive, not fluent.** It quotes retrieved sentences
  rather than synthesizing prose. Occasionally produces a near-duplicate sentence when chunk
  overlap causes the same fact to appear in two adjacent chunks. This goes away entirely once
  a real `openai`/`anthropic` key is configured. The UI labels this clearly (a yellow
  "Not AI-generated" banner) on every dashboard that shows an answer.
- **No OCR.** Scanned/image-only PDFs are explicitly out of scope per the brief and are
  rejected with a clear "no extractable text" error (verified against a real generated
  image-only PDF, not just a mock) rather than silently producing nothing or falsely claiming
  OCR support.
- **Synchronous ingestion.** Given three short PDFs (~3 pages each), ingestion runs in the
  request/response cycle (a few seconds) rather than via Celery/Redis, which the brief does not
  require for Phase 1.
- **No password reset / registration flow.** Accounts and roles are created via
  `seed_demo_users` or Django admin, matching "Django admin may be used for administration"
  in the official brief.
- **Citation/relevance score is lexical, not semantic** - see "What the citation 'score' is -
  and is not" above. This is the most significant quality limitation until a real embedding
  provider is configured.
- **Fixed, found and corrected in this milestone**: a failed *re-ingestion* of a manual used to
  leave the previous successful ingestion's pages/chunks in the database untouched (only
  `status` flipped to `failed`), so stale content could still be silently retrieved under a
  manual that looked broken in the UI. Found via a real-PDF test (`test_real_pdf_ingestion.py`),
  not a mock. Fixed in `manuals/services.py: ingest_manual` - a failed (re-)ingestion now always
  clears that manual's pages/chunks/page_count.

## 5. Requirements compliance matrix

| Requirement (brief) | Status | Where |
|---|---|---|
| Django + DRF + Postgres | Done | `config/settings.py`, verified: real `migrate` against PostgreSQL 18 |
| Bootstrap/HTML/CSS/JS frontend | Done | `templates/manuals/*.html` |
| PyMuPDF extraction, 1-based page numbers preserved | Done | `manuals/services.py: extract_pages` |
| Embeddings + vector store + LLM, pluggable | Done | `manuals/rag/providers.py`, `manuals/services.py` |
| Retrieval restricted to selected machine | Done, tested | `retrieve_chunks` filters by `machine`; `tests/test_pipeline.py::MachineIsolationRetrievalTests` |
| Citations: passage, document id/name, revision, page, link | Done | `CitationSerializer`, engineer dashboard UI |
| Abstains when evidence insufficient | Done, tested | `rag/pipeline.py`, `tests/test_pipeline.py::AnswerQuestionTests` |
| Persist question/answer/machine/timestamp/sources/status | Done | `Question`, `Citation` models |
| Server-enforced roles (not just hidden buttons) | Done, tested | `manuals/permissions.py`; `tests/test_api_permissions.py` |
| Check: Engineers cannot upload manuals | Done | `tests/test_api_permissions.py::UploadRestrictionTests` |
| Check: retrieval excludes other machine's passages | Done | `tests/test_pipeline.py::MachineIsolationRetrievalTests` |
| Question history (Engineer: own; Supervisor: all, read-only) | Done, tested | `QuestionHistoryView`; `tests/test_api_permissions.py::QuestionHistoryScopeTests` |

## 6. Tests

```bash
.venv\Scripts\python.exe manage.py test manuals
```

**57/57 passing** (last run: see milestone report). Covers: chunking edge cases, embedding
determinism and true/false-match separation (including light stemming), ingestion
success/failure/re-ingestion (both mocked for speed AND against real temporary PDF files -
valid, corrupt, encrypted, and image-only, built with PyMuPDF itself - see
`tests/test_real_pdf_ingestion.py` and `tests/pdf_fixtures.py`), RAG abstention, machine
isolation (retrieval layer, end-to-end, and the "identical alarm code means something different
on each machine" scenario in `tests/test_retrieval_regression.py`), citation-to-chunk accuracy,
per-page citation dedup, chunk-boundary word-safety, retrieval threshold/top-K edge cases, and
all role-based API permission/isolation checks.

## 7. Demo script (maps to the official checklist)

1. Log in as `admin1` -> register machine(s) (or they're already registered if you ran
   `ingest_sample_manuals`) -> upload a PDF -> watch status go `processing` -> `completed`
   with a page count.
2. Log in as `engineer1`, select **M-HP200**, ask *"What does E17 mean?"* -> answered, with a
   citation to `HP-OPS-01` and a page-open link.
3. Ask *"What should be recorded before escalation?"* on M-HP200 -> answer draws from
   `HP-MNT-01` (the second, complementary HP-200 manual) - demonstrates multi-document citation.
4. Select **M-AC90**, ask *"What does E17 mean?"* -> a **different**, correct answer
   (AC-90's E17 is unrelated to HP-200's) - proves machine isolation, not just a stock reply.
5. Ask an unsupported question (e.g. warranty/engine questions not in any manual) -> explicit
   "insufficient evidence" status, no fabrication.
6. Log in as `engineer1` and try `POST /api/manuals/` (e.g. via browser devtools or curl) ->
   403 Forbidden - restricted upload access, enforced server-side.
7. Log in as `supervisor1` -> read-only history table across all engineers/machines, with an
   "unanswered only" filter.

## 8. Likely judge questions

- **"Why isn't this using a real LLM?"** No organizer embedding/LLM access had been issued
  as of submission (per `START_HERE.txt`). The provider interface in
  `manuals/rag/providers.py` is built so switching to OpenAI/Anthropic is a `.env` change, no
  code change - and the fallback is explicitly labeled, never passed off as AI-generated.
- **"How do you know retrieval is actually isolated, not just filtered in the UI?"** It's
  enforced in the DB query itself (`Chunk.objects.filter(machine=machine)`) before any
  similarity scoring happens, and covered by an automated test that asserts a chunk from
  machine B is never returned for a machine A query even when the text would otherwise match.
- **"What happens with a bad PDF?"** `extract_pages` raises a typed `IngestionError` for
  encrypted/empty/unreadable PDFs; `ingest_manual` catches it and marks the manual `failed`
  with the message, never silently succeeding or crashing the request.
- **"How did you pick the similarity threshold?"** Measured on this corpus, not guessed - see
  section 3 above.
