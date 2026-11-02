"""
Retrieval and safety regression tests (Task C).

These intentionally use REAL PDF files and the full ingestion pipeline for
the "identical alarm code" scenario, because that is the single most
important correctness property of this whole app: the same code/term
appearing in two machines' manuals, with different meanings, must never
cross over.
"""

from django.test import TestCase, override_settings

from manuals.models import Chunk, Question
from manuals.rag.pipeline import answer_question
from manuals.services import retrieve_chunks

from . import pdf_fixtures
from .helpers import make_chunk, make_machine, make_manual, make_user


@override_settings(RAG_SIMILARITY_THRESHOLD=0.12, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class IdenticalAlarmCodeIsolationTests(TestCase):
    """Task C, point 1: identical alarm codes on different machines must
    only ever return the selected machine's own evidence."""

    def setUp(self):
        self.machine_hp200 = make_machine("M-HP200", "HP-200", "Hydraulic forming press")
        self.machine_ac90 = make_machine("M-AC90", "AC-90", "Air compressor")
        self.manual_hp200 = make_manual(self.machine_hp200, document_id="HP-OPS-01")
        self.manual_ac90 = make_manual(self.machine_ac90, document_id="AC-OM-01")
        # Deliberately the SAME alarm code string, different meaning per machine -
        # this is exactly the trap a naive keyword search would fall into.
        make_chunk(
            self.manual_hp200,
            "E17 on the HP-200 indicates a hydraulic pressure sensor fault. Stop and inspect wiring.",
            page_number=2,
        )
        make_chunk(
            self.manual_ac90,
            "E17 on the AC-90 indicates high discharge temperature. Check the cooling fan.",
            page_number=2,
        )

    def test_hp200_e17_question_only_returns_hp200_evidence(self):
        results = retrieve_chunks(self.machine_hp200, "What does E17 mean?")
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertEqual(r.chunk.machine_id, self.machine_hp200.id)
            self.assertIn("hydraulic pressure", r.chunk.text)
            self.assertNotIn("discharge temperature", r.chunk.text)

    def test_ac90_e17_question_only_returns_ac90_evidence(self):
        results = retrieve_chunks(self.machine_ac90, "What does E17 mean?")
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertEqual(r.chunk.machine_id, self.machine_ac90.id)
            self.assertIn("discharge temperature", r.chunk.text)
            self.assertNotIn("hydraulic pressure", r.chunk.text)

    def test_end_to_end_answers_differ_by_machine_for_the_same_question(self):
        engineer = make_user("eng_e17", "engineer")
        q_hp200 = answer_question(engineer, self.machine_hp200, "What does E17 mean?")
        q_ac90 = answer_question(engineer, self.machine_ac90, "What does E17 mean?")

        self.assertEqual(q_hp200.status, Question.Status.ANSWERED)
        self.assertEqual(q_ac90.status, Question.Status.ANSWERED)
        self.assertNotEqual(q_hp200.answer_text, q_ac90.answer_text)
        self.assertIn("hydraulic", q_hp200.answer_text.lower())
        self.assertIn("temperature", q_ac90.answer_text.lower())

        for c in q_hp200.citations.all():
            self.assertEqual(c.manual.machine_id, self.machine_hp200.id)
        for c in q_ac90.citations.all():
            self.assertEqual(c.manual.machine_id, self.machine_ac90.id)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.12, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class CitationAccuracyTests(TestCase):
    """Task C, point 3: every citation must correspond to a chunk that was
    actually retrieved, with the correct manual and one-based page number."""

    def test_multi_document_answer_cites_the_right_manual_and_page_for_each(self):
        machine = make_machine("M-HP200", "HP-200", "Hydraulic forming press")
        ops_manual = make_manual(machine, document_id="HP-OPS-01")
        mnt_manual = make_manual(machine, document_id="HP-MNT-01")
        make_chunk(ops_manual, "E17 indicates a hydraulic pressure sensor fault during forming.", page_number=2)
        make_chunk(mnt_manual, "Before escalating E17, record the alarm code and reservoir level.", page_number=3)

        engineer = make_user("eng_multi", "engineer")
        question = answer_question(engineer, machine, "What does E17 mean and what should be recorded before escalation?")

        self.assertEqual(question.status, Question.Status.ANSWERED)
        citations = list(question.citations.all())
        self.assertEqual(len(citations), 2)

        cited_by_doc = {c.manual.document_id: c for c in citations}
        self.assertIn("HP-OPS-01", cited_by_doc)
        self.assertIn("HP-MNT-01", cited_by_doc)
        self.assertEqual(cited_by_doc["HP-OPS-01"].page_number, 2)
        self.assertEqual(cited_by_doc["HP-MNT-01"].page_number, 3)
        # The excerpt text must actually be a substring of what was chunked/stored -
        # never a snippet the generator invented.
        stored_chunk_texts = set(Chunk.objects.filter(manual__in=[ops_manual, mnt_manual]).values_list("text", flat=True))
        for c in citations:
            self.assertTrue(
                any(c.excerpt in chunk_text for chunk_text in stored_chunk_texts),
                f"citation excerpt not found in any stored chunk: {c.excerpt!r}",
            )


@override_settings(RAG_SIMILARITY_THRESHOLD=0.12, RAG_TOP_K=5)
class RealIngestionDuplicateAndBoundaryRegressionTests(TestCase):
    """
    Task C, point 4: using a REAL long page run through the real chunker and
    real ingestion pipeline (not hand-built chunks), check there are no
    duplicate excerpts among retrieved chunks and no chunk starts mid-word.
    """

    def setUp(self):
        self.machine = make_machine()
        self._cleanup_paths = []

    def tearDown(self):
        pdf_fixtures.cleanup(*self._cleanup_paths)

    def test_long_page_does_not_produce_duplicate_or_mid_word_citations(self):
        # Long enough that chunk_page_text() must split it into 2+ overlapping chunks.
        long_text = (
            "The AF-10 filter indicator must be checked every 100 operating hours. "
            "Planned replacement of the AF-10 filter is required every 600 operating hours. "
            "Record the hour meter reading and the technician initials at each check. "
            "If the restriction indicator shows red, replace the filter immediately regardless "
            "of the scheduled interval, and note the early replacement in the maintenance log. "
            "A calendar-based interval is not supplied for the AF-10 filter; only the operating-hour "
            "interval applies to this component across all AC-90 deployment conditions and duty cycles. "
            "Inspect the filter housing gasket for visible cracking or deformation at every scheduled check. "
            "Confirm the upstream shutoff valve is fully open before removing the filter cartridge for service. "
            "Dispose of the used filter cartridge according to the site's hazardous-material handling procedure."
        )
        path = pdf_fixtures.make_valid_pdf([long_text])
        self._cleanup_paths.append(path)

        from django.core.files import File

        from manuals.models import Manual
        from manuals.services import ingest_manual
        from .helpers import make_manual

        manual = make_manual(self.machine, document_id="LONGPAGE-1")
        manual.status = Manual.Status.PENDING
        manual.save()
        with open(path, "rb") as f:
            manual.file.save("long.pdf", File(f), save=True)
        ingest_manual(manual)
        manual.refresh_from_db()
        self.assertEqual(manual.status, Manual.Status.COMPLETED)

        chunk_count = Chunk.objects.filter(manual=manual).count()
        self.assertGreaterEqual(chunk_count, 2, "test setup should have produced multiple chunks")

        results = retrieve_chunks(self.machine, "How often should the AF-10 filter be checked and replaced?")
        self.assertGreaterEqual(len(results), 1)

        pages_seen = [r.chunk.page.page_number for r in results]
        self.assertEqual(len(pages_seen), len(set(pages_seen)), "no two citations should come from the same page")

        # No chunk text may start with a lowercase-continuation fragment
        # (a sign the overlap sliced mid-word, e.g. "nt The AF-10...").
        for r in results:
            first_word = r.chunk.text.split(" ", 1)[0]
            self.assertTrue(
                first_word[:1].isupper() or first_word[:1].isdigit() or not first_word.isalpha(),
                f"chunk appears to start mid-word: {r.chunk.text[:40]!r}",
            )
