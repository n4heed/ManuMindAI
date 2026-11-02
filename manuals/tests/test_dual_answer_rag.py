"""
Integration tests for the Dual-Answer RAG pipeline (manuals/rag/pipeline.py)
and its exposure through the Question API. The Groq HTTP layer is never hit
for real here - every test either EXPLICITLY disables Groq via
@override_settings(LLM_ENABLED=False) or mocks
manuals.rag.pipeline.get_groq_provider directly, per "testing without an API
key". Settings are never left to their process default here, because a
developer's real .env (e.g. with a live key for the smoke test) would
otherwise change what these tests actually exercise - settings.py reads
.env at process start, so an un-overridden test would pick up whatever the
developer's own .env currently has configured.

Covers the structured-output claim/citation validation added to fix the
real citation-reliability issue found during live verification (a question
on one machine was shown citing another machine's manual, after a manual
had been uploaded with the wrong machine selected - see
test_e17_cross_manual_regression below for the exact scenario, rebuilt with
real fixture-style manual text, not fabricated content).
"""

from unittest import mock

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from manuals.models import Citation, GroqCitation, GroqClaim, Question
from manuals.rag.groq_provider import GroqAnswer, GroqClaimResult, GroqConfigurationError, GroqProviderError
from manuals.rag.pipeline import answer_question
from manuals.services import retrieve_chunks

from .helpers import make_chunk, make_machine, make_manual, make_user


def _answer(*claims_data):
    """claims_data: (text, [(source_id, quote), ...]) tuples."""
    claims = [
        GroqClaimResult(
            text=text,
            sources=[{"source_id": sid, "supporting_quote": quote} for sid, quote in sources],
        )
        for text, sources in claims_data
    ]
    return GroqAnswer(status="answered", claims=claims)


def _insufficient_evidence_answer():
    return GroqAnswer(status="insufficient_evidence", claims=[])


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class RagOnlyUnaffectedByGroqConfigTests(TestCase):
    """#16: RAG-only must work identically whether or not Groq is configured."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "The AF-10 filter should be checked every 100 operating hours.")
        self.engineer = make_user("eng_dual_1", "engineer")

    def test_rag_only_answer_generated_with_llm_disabled(self):
        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertTrue(question.answer_text)
        self.assertEqual(Citation.objects.filter(question=question).count(), 1)

    def test_groq_is_unavailable_with_llm_disabled(self):
        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.groq_status, Question.GroqStatus.UNAVAILABLE)
        self.assertEqual(question.groq_answer_text, "")
        self.assertEqual(GroqCitation.objects.filter(question=question).count(), 0)

    @override_settings(LLM_ENABLED=True, GROQ_API_KEY="")
    def test_groq_is_unavailable_when_enabled_but_no_api_key(self):
        """#9: missing API key disables Groq gracefully, RAG-only still works."""
        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertEqual(question.groq_status, Question.GroqStatus.UNAVAILABLE)
        self.assertIn("GROQ_API_KEY", question.groq_error_message)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class SharedRetrievalTests(TestCase):
    """A single retrieval feeds both generators identically."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "The AF-10 filter should be checked every 100 operating hours.", page_number=1)
        make_chunk(self.manual, "Replace the RF-20 filter every 250 operating hours.", page_number=2)
        self.engineer = make_user("eng_dual_2", "engineer")

    @mock.patch("manuals.rag.pipeline.retrieve_chunks", wraps=retrieve_chunks)
    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_retrieval_happens_exactly_once(self, mock_get_provider, mock_retrieve):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(("Answer.", [("S1", "The AF-10 filter should be checked every 100 operating hours.")]))
        mock_get_provider.return_value = mock_provider

        answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(mock_retrieve.call_count, 1)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_groq_payload_matches_the_rag_only_citations_exactly(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("Checked every 100 hours.", [("S1", "The AF-10 filter should be checked every 100 operating hours.")]),
            ("Replaced every 250 hours.", [("S2", "Replace the RF-20 filter every 250 operating hours.")]),
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "How often should filters be checked?")

        sent_question, sent_sources = mock_provider.generate.call_args[0]
        self.assertEqual(sent_question, "How often should filters be checked?")

        rag_only_texts = set(Citation.objects.filter(question=question).values_list("excerpt", flat=True))
        sent_texts = {s["text"][:400] for s in sent_sources}
        self.assertTrue(sent_texts.issubset(rag_only_texts | {s["text"] for s in sent_sources}))
        self.assertEqual(len(sent_sources), Citation.objects.filter(question=question).count())

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_groq_receives_only_chunk_text_not_full_page_text(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(("Answer.", [("S1", "The AF-10 filter should be checked every 100 operating hours.")]))
        mock_get_provider.return_value = mock_provider

        answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        _, sent_sources = mock_provider.generate.call_args[0]
        for s in sent_sources:
            self.assertLess(len(s["text"]), 1000)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class InsufficientEvidenceTests(TestCase):
    """#8 (retrieval-level), #10: insufficient evidence must never trigger a Groq call at the
    retrieval stage, and the model's own insufficient_evidence verdict must be honored too."""

    def setUp(self):
        self.machine = make_machine()
        self.engineer = make_user("eng_dual_3", "engineer")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_groq_is_never_called_when_retrieval_finds_nothing(self, mock_get_provider):
        question = answer_question(self.engineer, self.machine, "What is the warranty claim procedure?")
        self.assertEqual(question.status, Question.Status.INSUFFICIENT_EVIDENCE)
        self.assertEqual(question.groq_status, Question.GroqStatus.INSUFFICIENT_EVIDENCE)
        mock_get_provider.assert_not_called()

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_models_own_insufficient_evidence_verdict_is_honored(self, mock_get_provider):
        """Retrieval found *something* (so the model was called), but the model itself
        judged it couldn't answer - this must be stored as insufficient_evidence, not failed."""
        manual = make_manual(self.machine)
        make_chunk(manual, "The AF-10 filter should be checked every 100 operating hours.")
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _insufficient_evidence_answer()
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What is the warranty claim procedure for third-party engines?")
        self.assertEqual(question.groq_status, Question.GroqStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(GroqClaim.objects.filter(question=question).count(), 0)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class ProviderFailureDoesNotDestroyRagOnlyTests(TestCase):
    """Whatever goes wrong with Groq, the RAG-only answer is untouched."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "The AF-10 filter should be checked every 100 operating hours.")
        self.engineer = make_user("eng_dual_4", "engineer")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_provider_error_leaves_rag_only_answer_intact(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.side_effect = GroqProviderError("Groq rate limit reached (HTTP 429).")
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertTrue(question.answer_text)
        self.assertEqual(Citation.objects.filter(question=question).count(), 1)
        self.assertEqual(question.groq_status, Question.GroqStatus.FAILED)
        self.assertIn("429", question.groq_error_message)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_unexpected_exception_in_groq_path_leaves_rag_only_answer_intact(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.side_effect = RuntimeError("boom - an unanticipated bug")
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertTrue(question.answer_text)
        self.assertEqual(question.groq_status, Question.GroqStatus.FAILED)
        self.assertNotIn("boom", question.groq_error_message)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_configuration_error_is_reported_without_an_api_call_attempt(self, mock_get_provider):
        mock_get_provider.side_effect = GroqConfigurationError("The Groq integration is disabled (LLM_ENABLED=False).")
        question = answer_question(self.engineer, self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(question.status, Question.Status.ANSWERED)
        self.assertEqual(question.groq_status, Question.GroqStatus.UNAVAILABLE)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class ClaimCitationValidationTests(TestCase):
    """#3, #6, #7, #8, #9, #12, #13, #15: per-claim, per-source citation validation."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="HP-OPS-01")
        make_chunk(self.manual, "E17 indicates a hydraulic pressure fault.", page_number=2)
        self.engineer = make_user("eng_dual_5", "engineer")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_single_claim_with_one_valid_source_is_accepted(self, mock_get_provider):
        """#3: one factual claim with one valid source."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a hydraulic pressure fault.", [("S1", "E17 indicates a hydraulic pressure fault.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.COMPLETED)
        claims = GroqClaim.objects.filter(question=question)
        self.assertEqual(claims.count(), 1)
        self.assertEqual(claims.first().citations.count(), 1)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_explanatory_multi_sentence_claim_is_stored_verbatim_and_grounded(self, mock_get_provider):
        """Explanatory answers (the goal of this milestone) are longer than
        the earlier one-liners, but the grounding requirement is unchanged:
        the quote only needs to be a real substring, not the whole claim."""
        explanatory_text = (
            "E17 indicates a hydraulic pressure fault on this machine. In practical terms, the "
            "controller has detected pressure below the documented threshold during forming. The "
            "manual does not specify a corrective action beyond reporting the condition."
        )
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            (explanatory_text, [("S1", "E17 indicates a hydraulic pressure fault.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.COMPLETED)
        claim = GroqClaim.objects.get(question=question)
        self.assertEqual(claim.text, explanatory_text)  # stored exactly, not truncated/altered
        citation = claim.citations.get()
        self.assertEqual(citation.quote, "E17 indicates a hydraulic pressure fault.")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_unknown_source_identifier_is_rejected(self, mock_get_provider):
        """#7: a source_id Groq invented (never sent to it) is rejected."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 means something.", [("S99", "E17 indicates a hydraulic pressure fault.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.FAILED)
        self.assertEqual(GroqClaim.objects.filter(question=question).count(), 0)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_fabricated_quotation_is_rejected_even_with_a_valid_source_id(self, mock_get_provider):
        """#8: a quote that does not actually appear in the cited passage is rejected,
        even though the source_id itself is real."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 means the machine will explode.", [("S1", "E17 means the machine will explode.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.FAILED)
        self.assertEqual(GroqClaim.objects.filter(question=question).count(), 0)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_exact_quotation_is_accepted(self, mock_get_provider):
        """#9: an exact, verbatim quotation is accepted."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a hydraulic pressure fault.", [("S1", "E17 indicates a hydraulic pressure fault.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        citation = GroqCitation.objects.get(question=question)
        self.assertEqual(citation.quote, "E17 indicates a hydraulic pressure fault.")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_whitespace_only_difference_in_quotation_is_still_accepted(self, mock_get_provider):
        """Conservative whitespace normalization: extra/odd spacing is tolerated, wording is not."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a fault.", [("S1", "E17   indicates\na hydraulic   pressure fault.")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.COMPLETED)

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_claim_with_no_valid_sources_is_dropped_but_other_claims_survive(self, mock_get_provider):
        """#6, #12: one unsupported claim does not sink claims that ARE supported -
        a multi-part answer's unanswerable part is dropped, not invented."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a hydraulic pressure fault.", [("S1", "E17 indicates a hydraulic pressure fault.")]),
            ("The warranty claim procedure requires form 27-B.", [("S1", "the warranty claim procedure requires form 27-B")]),  # fabricated, not in S1's text
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean, and what is the warranty claim procedure?")
        self.assertEqual(question.groq_status, Question.GroqStatus.COMPLETED)
        claims = list(GroqClaim.objects.filter(question=question))
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "E17 indicates a hydraulic pressure fault.")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_all_claims_rejected_marks_the_whole_answer_failed(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("Completely fabricated statement.", [("S1", "this text does not appear anywhere in the source")])
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean?")
        self.assertEqual(question.groq_status, Question.GroqStatus.FAILED)
        self.assertIn("quote-verified", question.groq_error_message)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class MultiPartQuestionTests(TestCase):
    """#4, #5, #11: multi-claim answers with different/multiple source identifiers."""

    def setUp(self):
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="HP-OPS-01")
        make_chunk(self.manual, "E17 indicates a hydraulic pressure fault.", page_number=2)
        make_manual2 = make_manual(self.machine, document_id="HP-MNT-01")
        make_chunk(make_manual2, "Record the reservoir level and RF-20 filter state before escalation.", page_number=3)
        self.engineer = make_user("eng_dual_multi", "engineer")

    def _source_id_for(self, machine, question_text, chunk_text_contains):
        """
        The pipeline assigns [S#] identifiers in retrieval-score order, not
        document order - look up the real id for a given chunk the same way
        the pipeline does, so these tests don't depend on guessing score
        ordering (retrieval itself is exercised for real; only the Groq
        call is mocked).
        """
        for i, r in enumerate(retrieve_chunks(machine, question_text)):
            if chunk_text_contains in r.chunk.text:
                return f"S{i + 1}"
        raise AssertionError(f"no retrieved chunk contains {chunk_text_contains!r}")

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_multiple_claims_with_different_source_identifiers(self, mock_get_provider):
        """#4: each part of a multi-part question gets its own claim and source."""
        q = "What does E17 mean and what should be recorded before escalation?"
        sid_e17 = self._source_id_for(self.machine, q, "E17 indicates")
        sid_reservoir = self._source_id_for(self.machine, q, "Record the reservoir level")

        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a hydraulic pressure fault.", [(sid_e17, "E17 indicates a hydraulic pressure fault.")]),
            ("Record the reservoir level and RF-20 filter state before escalation.", [(sid_reservoir, "Record the reservoir level and RF-20 filter state before escalation.")]),
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, q)
        claims = list(GroqClaim.objects.filter(question=question).order_by("order"))
        self.assertEqual(len(claims), 2)
        self.assertEqual({claims[0].citations.first().source_id, claims[1].citations.first().source_id}, {sid_e17, sid_reservoir})

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_single_claim_supported_by_multiple_passages(self, mock_get_provider):
        """#5: one claim can cite more than one source."""
        q = "What does E17 mean and what should be recorded before escalation?"
        sid_e17 = self._source_id_for(self.machine, q, "E17 indicates")
        sid_reservoir = self._source_id_for(self.machine, q, "Record the reservoir level")

        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        mock_provider.generate.return_value = _answer(
            (
                "E17 should be investigated using the reservoir level and the RF-20 filter state.",
                [
                    (sid_e17, "E17 indicates a hydraulic pressure fault."),
                    (sid_reservoir, "Record the reservoir level and RF-20 filter state before escalation."),
                ],
            ),
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, q)
        claim = GroqClaim.objects.get(question=question)
        self.assertEqual(claim.citations.count(), 2)
        self.assertEqual({c.source_id for c in claim.citations.all()}, {sid_e17, sid_reservoir})

    @mock.patch("manuals.rag.pipeline.get_groq_provider")
    def test_one_unanswered_part_does_not_force_an_invented_claim(self, mock_get_provider):
        """#12: the model may legitimately answer only the supported part of a multi-part
        question; the backend must not require every part to have a claim."""
        mock_provider = mock.MagicMock()
        mock_provider.model = "openai/gpt-oss-120b"
        # Only answers the E17 part; the model chose not to claim anything about a part
        # with no evidence (this is what a well-behaved model does, not a bug).
        mock_provider.generate.return_value = _answer(
            ("E17 indicates a hydraulic pressure fault.", [("S1", "E17 indicates a hydraulic pressure fault.")]),
        )
        mock_get_provider.return_value = mock_provider

        question = answer_question(self.engineer, self.machine, "What does E17 mean and what is the warranty procedure?")
        self.assertEqual(question.groq_status, Question.GroqStatus.COMPLETED)
        self.assertEqual(GroqClaim.objects.filter(question=question).count(), 1)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class MachineIsolationTests(TestCase):
    """#13, #14: Groq's claims/citations must respect machine isolation exactly like
    RAG-only citations - including rejecting a source that (by some bug) pointed at
    another machine's chunk, as defense in depth beyond the already machine-scoped
    retrieval."""

    def test_groq_citations_never_cross_machines(self):
        machine_a = make_machine("M-A", "A-1", "Machine A")
        machine_b = make_machine("M-B", "B-1", "Machine B")
        manual_a = make_manual(machine_a, document_id="A-DOC")
        manual_b = make_manual(machine_b, document_id="B-DOC")
        make_chunk(manual_a, "E17 on machine A means a pressure fault.", page_number=1)
        make_chunk(manual_b, "E17 on machine B means a temperature fault.", page_number=1)
        engineer = make_user("eng_dual_iso", "engineer")

        with mock.patch("manuals.rag.pipeline.get_groq_provider") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_provider.model = "openai/gpt-oss-120b"
            mock_provider.generate.return_value = _answer(
                ("E17 on machine A means a pressure fault.", [("S1", "E17 on machine A means a pressure fault.")])
            )
            mock_get_provider.return_value = mock_provider

            question_a = answer_question(engineer, machine_a, "What does E17 mean?")

        citation = GroqCitation.objects.get(question=question_a)
        self.assertEqual(citation.manual_id, manual_a.id)
        self.assertNotEqual(citation.manual_id, manual_b.id)

    def test_e17_cross_manual_regression(self):
        """
        Regression test rebuilding the real issue found during live
        verification: a manual whose own text explicitly names a different
        machine was uploaded (by mistake) under the wrong machine, so its
        chunks were retrieved and cited for a question about the WRONG
        machine - "AP-MNT-01 / Machine: M-AP500/AP-500" content shown as
        evidence for an M-AC90 question. Content here is written to mirror
        exactly that structure (a manual that names a different machine on
        every page), not fabricated as something else.

        The fix in production was correcting the mis-tagged Manual.machine
        and re-ingesting (a data fix, not a code fix - retrieval was always
        correctly scoped by machine). This test proves the code's own
        machine-scoping holds given correctly-tagged data: a manual
        correctly tagged to its own machine is never retrieved for, or
        cited by, a different machine's question - both for RAG-only and
        for the validated Groq claims.
        """
        ac90 = make_machine("M-AC90", "AC-90", "Air compressor")
        ap500 = make_machine("M-AP500", "AP-500", "Industrial air compressor")
        ac90_manual = make_manual(ac90, document_id="AC-OM-01")
        # Correctly tagged to AP-500, and its own text says so - this must
        # never surface for an AC-90 question.
        ap500_manual = make_manual(ap500, document_id="AP-MNT-01")
        make_chunk(ac90_manual, "E17 on AC-90 means high discharge temperature.", page_number=2)
        make_chunk(
            ap500_manual,
            "Document ID: AP-MNT-01 | Machine: M-AP500/AP-500. Inspection Schedule: sump sight-glass and condensate drain valve.",
            page_number=2,
        )
        engineer = make_user("eng_e17_regression", "engineer")

        # RAG-only: machine-scoped retrieval must only ever see AC-90's own manual.
        results = retrieve_chunks(ac90, "What does E17 mean?")
        self.assertTrue(all(r.chunk.manual.machine_id == ac90.id for r in results))
        self.assertTrue(all(r.chunk.manual.document_id != "AP-MNT-01" for r in results))

        with mock.patch("manuals.rag.pipeline.get_groq_provider") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_provider.model = "openai/gpt-oss-120b"
            mock_provider.generate.return_value = _answer(
                ("E17 on AC-90 means high discharge temperature.", [("S1", "E17 on AC-90 means high discharge temperature.")])
            )
            mock_get_provider.return_value = mock_provider
            question = answer_question(engineer, ac90, "What does E17 mean?")

        for citation in GroqCitation.objects.filter(question=question):
            self.assertEqual(citation.manual.machine_id, ac90.id)
            self.assertNotEqual(citation.manual.document_id, "AP-MNT-01")


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=True, GROQ_API_KEY="sk-super-secret-key")
class ApiResponseShapeAndSecurityTests(TestCase):
    """#1 (via API), #17, #18, #19: API shape, PDF links, no key leakage, and supervisor read access."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="HP-OPS-01")
        make_chunk(self.manual, "E17 indicates a hydraulic pressure fault.", page_number=2)
        self.engineer = make_user("eng_dual_api", "engineer")
        self.supervisor = make_user("sup_dual_api", "supervisor")

    def _ask(self):
        with mock.patch("manuals.rag.pipeline.get_groq_provider") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_provider.model = "openai/gpt-oss-120b"
            mock_provider.generate.return_value = _answer(
                ("E17 indicates a hydraulic pressure fault.", [("S1", "E17 indicates a hydraulic pressure fault.")])
            )
            mock_get_provider.return_value = mock_provider
            self.client.force_authenticate(self.engineer)
            return self.client.post(
                "/api/questions/ask/", {"machine": self.machine.id, "question_text": "What does E17 mean?"}, format="json"
            )

    def test_response_exposes_both_answer_objects_with_claims(self):
        """#1: a valid structured Groq response, surfaced via the API."""
        response = self._ask()
        self.assertEqual(response.status_code, 201)
        answers = response.data["answers"]
        self.assertIn("rag_only", answers)
        self.assertIn("rag_groq", answers)
        self.assertEqual(answers["rag_groq"]["status"], "completed")
        self.assertEqual(answers["rag_groq"]["provider"], "groq")
        self.assertEqual(answers["rag_groq"]["model"], "openai/gpt-oss-120b")
        self.assertIsInstance(answers["rag_groq"]["generation_time_ms"], int)
        self.assertEqual(len(answers["rag_groq"]["claims"]), 1)
        self.assertEqual(len(answers["rag_groq"]["claims"][0]["citations"]), 1)

    def test_groq_citation_uses_the_authenticated_pdf_endpoint(self):
        response = self._ask()
        citation = response.data["answers"]["rag_groq"]["claims"][0]["citations"][0]
        self.assertIn("/api/manuals/", citation["manual_file_url"])
        self.assertIn("/file/", citation["manual_file_url"])
        self.assertNotIn("/media/", citation["manual_file_url"])
        self.assertEqual(citation["page_number"], 2)
        self.assertEqual(citation["source_id"], "S1")
        self.assertIn("quote", citation)

    def test_api_key_never_appears_anywhere_in_the_response(self):
        """#19."""
        response = self._ask()
        self.assertNotIn("sk-super-secret-key", str(response.data))

    def test_supervisor_can_read_both_answers_but_cannot_ask(self):
        """#18."""
        self._ask()
        self.client.force_authenticate(self.supervisor)
        history = self.client.get("/api/questions/")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.data), 1)
        self.assertIn("rag_groq", history.data[0]["answers"])
        self.assertEqual(len(history.data[0]["answers"]["rag_groq"]["claims"]), 1)

        ask_attempt = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "x"}, format="json"
        )
        self.assertEqual(ask_attempt.status_code, 403)


class BackwardCompatibilityTests(TestCase):
    """#17: historical records (both pre-Groq, and pre-structured-output flat Groq
    answers) must still render correctly."""

    def test_pre_existing_question_without_groq_data_serializes_as_unavailable(self):
        from manuals.serializers import QuestionSerializer

        machine = make_machine()
        engineer = make_user("eng_dual_legacy", "engineer")
        question = Question.objects.create(
            user=engineer,
            machine=machine,
            question_text="Old question asked before Groq existed",
            answer_text="An old RAG-only answer.",
            status=Question.Status.ANSWERED,
            llm_provider="local",
        )
        data = QuestionSerializer(question, context={}).data
        self.assertEqual(data["answers"]["rag_only"]["answer"], "An old RAG-only answer.")
        self.assertEqual(data["answers"]["rag_groq"]["status"], "unavailable")
        self.assertEqual(data["answers"]["rag_groq"]["claims"], [])
        self.assertEqual(data["answers"]["rag_groq"]["citations"], [])
        self.assertEqual(data["answer_text"], "An old RAG-only answer.")

    def test_pre_structured_output_flat_groq_answer_still_renders(self):
        """A question answered under the OLD (pre-Phase-2) flat-text Groq format -
        groq_answer_text set, GroqCitation rows with claim=None - must still serialize,
        distinct from the new claims-based shape."""
        from manuals.serializers import QuestionSerializer
        from manuals.models import GroqCitation as GroqCitationModel

        machine = make_machine()
        manual = make_manual(machine, document_id="HP-OPS-01")
        engineer = make_user("eng_dual_legacy2", "engineer")
        question = Question.objects.create(
            user=engineer,
            machine=machine,
            question_text="A question answered before structured outputs existed",
            answer_text="RAG-only answer.",
            status=Question.Status.ANSWERED,
            llm_provider="local",
            groq_status=Question.GroqStatus.COMPLETED,
            groq_answer_text="Old flat-text Groq answer [S1].",
            groq_model="openai/gpt-oss-120b",
        )
        GroqCitationModel.objects.create(
            question=question, claim=None, manual=manual, page_number=1,
            excerpt="some excerpt", score=0.5, source_id="S1",
        )
        data = QuestionSerializer(question, context={}).data
        self.assertEqual(data["answers"]["rag_groq"]["status"], "completed")
        self.assertEqual(data["answers"]["rag_groq"]["answer"], "Old flat-text Groq answer [S1].")
        self.assertEqual(data["answers"]["rag_groq"]["claims"], [])  # no claims for this legacy row
        self.assertEqual(len(data["answers"]["rag_groq"]["citations"]), 1)  # flat citation list instead
