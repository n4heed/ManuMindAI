from unittest import mock

from django.test import TestCase, override_settings

from manuals.models import Question
from manuals.rag.pipeline import answer_question, build_insufficient_evidence_message
from manuals.services import retrieve_chunks

from .helpers import make_chunk, make_machine, make_manual, make_user


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class AnswerQuestionTests(TestCase):
    def setUp(self):
        self.engineer = make_user("eng1", "engineer")
        self.machine = make_machine("M-AC90", "AC-90", "Air compressor")
        self.manual = make_manual(self.machine, document_id="AC-OM-01")
        make_chunk(
            self.manual,
            "The AF-10 filter indicator should be checked weekly, with planned replacement every six months.",
            page_number=2,
        )

    def test_abstains_when_machine_has_no_manuals(self):
        empty_machine = make_machine("M-EMPTY", "EMPTY-1", "Unindexed machine")
        question = answer_question(self.engineer, empty_machine, "How often is the filter checked?")

        self.assertEqual(question.status, Question.Status.INSUFFICIENT_EVIDENCE)
        self.assertEqual(question.answer_text, build_insufficient_evidence_message(empty_machine))
        self.assertIn("M-EMPTY", question.answer_text)
        self.assertEqual(question.citations.count(), 0)

    def test_answers_with_citation_when_evidence_present(self):
        question = answer_question(self.engineer, self.machine, "How often is the AF-10 filter checked?")

        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertEqual(question.citations.count(), 1)
        citation = question.citations.first()
        self.assertEqual(citation.manual, self.manual)
        self.assertEqual(citation.page_number, 2)

    def test_unrelated_question_abstains_even_with_manuals_indexed(self):
        question = answer_question(self.engineer, self.machine, "What is the warranty claim procedure for engines?")
        self.assertEqual(question.status, Question.Status.INSUFFICIENT_EVIDENCE)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class MachineIsolationRetrievalTests(TestCase):
    """The two required automated checks from the brief include: 'retrieval excludes
    passages belonging to another machine'. This directly tests that at the
    retrieval layer, independent of any view/permission logic."""

    def setUp(self):
        self.machine_a = make_machine("M-HP200", "HP-200", "Hydraulic forming press")
        self.machine_b = make_machine("M-AC90", "AC-90", "Air compressor")
        self.manual_a = make_manual(self.machine_a, document_id="HP-OPS-01")
        self.manual_b = make_manual(self.machine_b, document_id="AC-OM-01")
        make_chunk(self.manual_a, "E17 indicates a hydraulic pressure sensor fault on the HP-200 press.")
        make_chunk(self.manual_b, "The AF-10 filter indicator should be checked weekly on the AC-90 compressor.")

    def test_machine_a_question_never_retrieves_machine_b_chunks(self):
        results = retrieve_chunks(self.machine_a, "How often should the AF-10 filter indicator be checked?")
        for r in results:
            self.assertEqual(r.chunk.machine_id, self.machine_a.id)
        # The only matching evidence belongs to machine B, so machine A should get nothing.
        self.assertEqual(len(results), 0)

    def test_machine_b_question_retrieves_its_own_matching_chunk(self):
        results = retrieve_chunks(self.machine_b, "How often should the AF-10 filter indicator be checked?")
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(r.chunk.machine_id == self.machine_b.id for r in results))

    def test_end_to_end_answer_respects_machine_isolation(self):
        engineer = make_user("eng2", "engineer")
        question = answer_question(engineer, self.machine_a, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.INSUFFICIENT_EVIDENCE)
        for citation in question.citations.all():
            self.assertEqual(citation.manual.machine_id, self.machine_a.id)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.12, RAG_TOP_K=5)
class RetrievalDedupAndThresholdTests(TestCase):
    """Covers the duplicate-citation fix: a long page split into overlapping
    chunks should surface as ONE citation, not one per chunk."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine)

    def test_at_most_one_citation_per_page(self):
        # Two chunks that both clearly match the question, from the SAME page -
        # simulating the overlap between adjacent chunks of one long page.
        make_chunk(self.manual, "The AF-10 filter indicator should be checked weekly.", page_number=3, chunk_index=0)
        make_chunk(
            self.manual,
            "checked weekly. Filter replacement for the AF-10 indicator is due every six months.",
            page_number=3,
            chunk_index=1,
        )
        results = retrieve_chunks(self.machine, "How often is the AF-10 filter indicator checked?")
        pages = [r.chunk.page.page_number for r in results]
        self.assertEqual(len(pages), len(set(pages)), "expected at most one chunk per page")

    def test_distinct_pages_are_not_deduped(self):
        make_chunk(self.manual, "The AF-10 filter indicator should be checked weekly.", page_number=2)
        make_chunk(self.manual, "The AF-10 filter is replaced every six months per the maintenance schedule.", page_number=3)
        results = retrieve_chunks(self.machine, "How often is the AF-10 filter checked or replaced?")
        pages = {r.chunk.page.page_number for r in results}
        self.assertEqual(pages, {2, 3})

    def test_threshold_boundary_is_inclusive(self):
        make_chunk(self.manual, "irrelevant placeholder text", page_number=1)
        with mock.patch("manuals.services.cosine_similarity", return_value=0.12):
            results = retrieve_chunks(self.machine, "anything")
        self.assertEqual(len(results), 1)  # score == threshold must be included (>=, not >)

    def test_top_k_is_respected(self):
        for i in range(10):
            make_chunk(self.manual, f"Filter check item number {i} for the AF-10 filter.", page_number=i + 1)
        with override_settings(RAG_TOP_K=3):
            results = retrieve_chunks(self.machine, "AF-10 filter check")
        self.assertLessEqual(len(results), 3)
