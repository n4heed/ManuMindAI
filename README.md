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

## 4. Dual-Answer RAG (RAG-only + Groq)

Every eligible question now produces **two independent answers**, shown side by side on the
Engineer and Supervisor dashboards:

- **RAG-ONLY** - exactly the existing deterministic extractive answer described in section 3
  above. Unchanged by this feature: same retrieval, same threshold, same generator, same
  `Citation` rows.
- **RAG + GROQ AI** - the *same retrieved chunks* sent to Groq's OpenAI-compatible chat API
  (`https://api.groq.com/openai/v1`, default model `openai/gpt-oss-120b`), which generates a
  fuller, grounded answer. Independently stored and independently fails without affecting the
  RAG-only answer.

### Why two answers, and how retrieval is shared

`manuals/rag/pipeline.py: answer_question()` calls `retrieve_chunks()` **exactly once**. Both
generators receive that same list - the RAG-only generator as before, and
`_attach_groq_answer()` afterwards, which builds `[S1]`, `[S2]`, ... identifiers from those same
chunks and nothing else. There is no second retrieval call, and Groq is never shown a full page
or a full PDF - only the already-chunked (~600 char) excerpts already used for the RAG-only
citations.

### Groq integration architecture

- `manuals/rag/groq_provider.py` - a provider independent of Django views/templates and of the
  DB layer. It takes a question string and a list of plain `{"id","document_id","page_number","text"}`
  dicts in, and returns a plain `GroqAnswer(text, cited_source_ids)` out.
- **HTTP client: `requests`, not the stdlib `urllib`, and not the `openai` package.**
  `pip show openai` confirmed the `openai` package is not installed in this project (and never
  has been - the existing `openai`/`anthropic` RAG-only provider classes in
  `manuals/rag/providers.py` only ever imported it lazily, inside a method body, and have never
  actually been run). The first implementation of `groq_provider.py` used only the standard
  library (`urllib.request`) for zero new dependencies. **During live verification against the
  real Groq API, that implementation consistently got HTTP 403 "error code: 1010" from
  Cloudflare** (the WAF in front of `api.groq.com`) - a client *network-fingerprint* block, not
  a credentials or model-permission problem. Confirmed by testing the exact same request,
  same key, same model, with `requests` instead: it succeeded immediately with a real 200
  response. `requests` (already present in this project's virtual environment, and now
  declared in `requirements.txt`) is used for exactly that reason, not for API ergonomics.
  Swapping to the real `openai` package later (pointed at Groq's base URL) would only require
  changing this one file.
- **Prompting and grounding**: the system prompt (`SYSTEM_PROMPT` in `groq_provider.py`)
  explicitly forbids general knowledge, invented measurements/intervals/ratings, and
  unsupported safety-critical instructions, and requires every claim to be cited with its
  `[S#]` identifier. The retrieved passages are explicitly described to the model as untrusted
  data, not instructions.
- **Strict Structured Outputs, not prose with `[S#]` hints.** `groq_provider.py` sends Groq's
  Chat Completions `response_format: {"type": "json_schema", "json_schema": {"strict": true,
  "schema": ...}}`, forcing the model's own output to be JSON matching
  `{"status": "answered"|"insufficient_evidence", "claims": [{"text", "sources": [{"source_id",
  "supporting_quote"}]}]}` - every property required, every object `additionalProperties:
  false`, as Groq's strict mode requires. This replaced an earlier version that only asked the
  model to mention `[S#]` somewhere in free-text prose, which a real live test showed the model
  does not reliably do for multi-part questions - the structured schema makes citing a source
  per claim part of the response *shape itself*, not a instruction the model can skip.
- **Citation validation is mandatory, server-side, and per-claim (not just per-answer).**
  Structured Outputs guarantees response *shape*, not factual truth, so
  `manuals/rag/pipeline.py: _validate_claims()` independently checks, for every claim's every
  source: (1) the `source_id` is one of the exact identifiers sent to Groq - never trusted
  otherwise; (2) the source's chunk belongs to the question's own machine (defense in depth -
  retrieval is already machine-scoped, so this can never actually differ, but it's checked
  explicitly rather than assumed); (3) the `supporting_quote` is a real, whitespace-normalized
  substring of that chunk's actual text - a fabricated or altered quotation is rejected even
  when the `source_id` is real. A source that fails any check is dropped; a claim with zero
  surviving sources is dropped entirely (never shown unsupported); if **every** claim is
  dropped, the whole Groq answer is marked `failed` rather than shown as properly cited. Every
  surviving `GroqCitation` row's manual, page number, and PDF link are resolved from the
  database, not from anything Groq said - and the PDF link is the same authenticated
  `ManualFileView` endpoint used everywhere else; no public `/media/` URL is ever produced.

### Default model and how to change it

Default: `openai/gpt-oss-120b`. Configurable via `GROQ_MODEL` in `.env` - no code change needed.

### Environment variables

Add these to your own `.env` (never commit `.env` - it stays gitignored; `.env.example` only
has blank/placeholder values for these):

```
LLM_ENABLED=False
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b
LLM_TIMEOUT=30
```

These are deliberately separate from `EMBEDDING_PROVIDER`/`LLM_PROVIDER` (section 3), which
still control the RAG-only generator and remain `local` - this feature does not touch them.

### How to manually add your Groq API key

1. Open your local `.env` file (not `.env.example`) in the project root.
2. Set `LLM_ENABLED=True`.
3. Set `GROQ_API_KEY=<your real key>`.
4. Optionally adjust `GROQ_MODEL` / `GROQ_BASE_URL` / `LLM_TIMEOUT`.
5. Restart the Django dev server so the new settings are read.

No code change is required. The RAG-only answer does **not** depend on this key at all - it
works identically with or without it, both before and after you add one.

### How the platform behaves when Groq is unavailable or fails

- **`LLM_ENABLED=False` (the default) or no `GROQ_API_KEY` set**: Groq is never called - not
  even once, no retry loop - and both dashboards show the RAG + Groq AI card as clearly
  **"Not configured"**, right next to the normal RAG-only answer.
- **Insufficient evidence**: if retrieval finds nothing, Groq is never called either - both
  answers show the same abstention message, consistent with the existing RAG-only behaviour.
- **A real API failure** (invalid key, rate limit, timeout, network error, malformed/empty
  response, invalid citations): the Groq card shows **"Failed"** with a short, sanitized
  message (never a raw stack trace, filesystem path, or the API key) - the RAG-only card next
  to it is completely unaffected, because the RAG-only answer is already saved to the database
  before Groq is ever called.
- Retryable failures (HTTP 429, HTTP 5xx, network/timeout errors) get a small bounded number of
  retries (`GroqLLMProvider.MAX_ATTEMPTS = 2`); invalid credentials (401/403) fail immediately
  with no retry.

### Running the automated tests (no API key needed)

```bash
.venv\Scripts\python.exe manage.py test manuals.tests.test_groq_provider
.venv\Scripts\python.exe manage.py test manuals.tests.test_dual_answer_rag
```

Every HTTP call Groq would make is mocked (`unittest.mock`) - these tests make **no real network
call** and need no API key, per the project requirement to implement and test this feature
before a real key is available.

### Running the optional live integration smoke test (only after you add a real key)

```bash
.venv\Scripts\python.exe manage.py groq_smoke_test
```

This makes one **real** call to the configured Groq API with a sample question, to confirm the
integration genuinely works end to end. It is never run automatically by this project's test
suite or by any command in this milestone's implementation - run it yourself, by hand, only
after completing the steps in "How to manually add your Groq API key" above.

## 5. Known limitations (honest, as of this report)

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
- **Live Groq integration has been verified against the real API**, twice: first with
  free-text `[S#]` prompting (see below for why that was replaced), and again after switching
  to Strict Structured Outputs. The structured version was live-tested with 4 real scenarios
  (simple, multi-part, multi-source-single-claim, insufficient-evidence) via
  `manage.py groq_smoke_test`, all exactly as expected, and with the actual previously-failing
  question through the real Engineer dashboard - it now answers correctly with two separate,
  quote-verified claims.
- **Fixed, found via a real live test, not a mock: free-text `[S#]` citation prompting was not
  reliable for multi-part questions.** The question *"What does E17 mean and what should be
  recorded before escalation?"* was asked live and Groq's answer didn't include `[S#]` markers
  for every claim, so the backend correctly rejected it (`groq_status: "failed"`) rather than
  show an uncited answer - the validation worked as designed, but the free-text prompting
  approach it was validating against wasn't reliable enough. Fixed by switching to Groq Strict
  Structured Outputs (see "Dual-Answer RAG" above), which was then live-verified to answer the
  exact same question correctly, with each part as its own cited, quote-verified claim.
- **Fixed, found via the same live test: a manual uploaded under the wrong machine.** The
  live screenshot that surfaced the citation-format issue also showed a citation to
  `AP-MNT-01` (a manual whose own text says "Machine: M-AP500/AP-500" on every page) for a
  question asked on M-AC90. Investigation confirmed retrieval and machine isolation were never
  broken (`Chunk.objects.filter(machine=machine)` was always correctly scoped) - the manual had
  simply been uploaded with the wrong machine selected at upload time, so its chunks correctly
  inherited that wrong machine tag. Fixed by reassigning the manual to its real machine and
  re-ingesting it (a data correction, not a code or architecture change); verified afterward
  that the exact same question's retrieval no longer includes that manual at all.
- **Fixed, found and corrected in this milestone**: a failed *re-ingestion* of a manual used to
  leave the previous successful ingestion's pages/chunks in the database untouched (only
  `status` flipped to `failed`), so stale content could still be silently retrieved under a
  manual that looked broken in the UI. Found via a real-PDF test (`test_real_pdf_ingestion.py`),
  not a mock. Fixed in `manuals/services.py: ingest_manual` - a failed (re-)ingestion now always
  clears that manual's pages/chunks/page_count.

## 6. Requirements compliance matrix

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

## 7. Tests

```bash
.venv\Scripts\python.exe manage.py test manuals
```

**130/130 passing** (last run: see milestone report). Covers everything described in earlier
milestones (chunking edge cases, embedding determinism and true/false-match separation,
ingestion success/failure/re-ingestion against both mocks and real temporary PDF files, RAG
abstention, machine isolation, citation-to-chunk accuracy, per-page citation dedup,
chunk-boundary word-safety, retrieval threshold/top-K edge cases, role-based API
permission/isolation checks, the security audit regression suite, and secure PDF delivery),
plus this milestone's additions:

- `tests/test_groq_provider.py` (19 tests) - the Groq provider in isolation: configuration
  (disabled/no key), request payload shape, response parsing, retry/backoff behaviour, and that
  no error message ever leaks the API key. Every HTTP call is mocked; no real network call.
- `tests/test_dual_answer_rag.py` (19 tests) - the dual-answer pipeline end to end: retrieval
  happens exactly once and both generators receive it; insufficient evidence prevents any Groq
  call; a Groq failure never destroys the RAG-only answer; invalid/partially-invalid citation
  identifiers are rejected and only backend-validated citations are ever stored; machine
  isolation holds for Groq citations too; the API response exposes both answer objects with the
  authenticated PDF link; the API key never appears in any response; a Supervisor can read both
  answers but still cannot ask a question; and a question created before this feature existed
  still serializes correctly (`groq_status: "unavailable"`, no crash).

## 8. Demo script (maps to the official checklist)

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

## 9. Likely judge questions

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
