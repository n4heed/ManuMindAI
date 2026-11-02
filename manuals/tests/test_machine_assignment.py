"""
Regression tests for the machine/manual assignment investigation:
- manuals/services.py: check_machine_mismatch (the Administrator-upload
  advisory safeguard)
- manuals/views_api.py: ManualMachineCheckView (its permission boundary)
- manuals/rag/pipeline.py: build_insufficient_evidence_message (machine-
  specific wording) and re-ingestion after a confirmed reassignment

Frontend-only behaviours (the Engineer dashboard clearing a stale answer
when the machine selector changes, and sending the currently-selected
machine ID on submit) are covered by live browser verification, documented
in the milestone report - they have no Django-testable surface of their
own since no backend code implements them.
"""

from django.core.files import File
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from manuals.models import Chunk, Manual, Question
from manuals.rag.pipeline import build_insufficient_evidence_message
from manuals.services import check_machine_mismatch, ingest_manual, retrieve_chunks

from . import pdf_fixtures
from .helpers import make_chunk, make_machine, make_manual, make_user


class CheckMachineMismatchTests(TestCase):
    """#1, #2: the advisory check correctly distinguishes a matching manual
    from one whose own declared machine conflicts with the selection."""

    def setUp(self):
        self.ac90 = make_machine("M-AC90", "AC-90", "Air compressor")
        self.ap500 = make_machine("M-AP500", "AP-500", "Industrial air compressor")
        self._cleanup_paths = []

    def tearDown(self):
        pdf_fixtures.cleanup(*self._cleanup_paths)

    def _upload(self, text):
        path = pdf_fixtures.make_valid_pdf([text])
        self._cleanup_paths.append(path)
        with open(path, "rb") as f:
            return SimpleUploadedFile("manual.pdf", f.read(), content_type="application/pdf")

    def test_matching_declared_machine_produces_no_warning(self):
        """#1: a manual correctly uploaded for the machine it declares."""
        uploaded = self._upload("AC-90 Maintenance Manual\nMachine: M-AC90\nRevision 1.0")
        warning = check_machine_mismatch(uploaded, self.ac90)
        self.assertIsNone(warning)

    def test_conflicting_declared_machine_produces_a_warning(self):
        """#2: the exact real-world scenario - a manual that explicitly
        names a different machine than the one selected for upload."""
        uploaded = self._upload("AP 500 Maintenance Manual\nMachine: M-AP500/AP-500\nRevision 1.0")
        warning = check_machine_mismatch(uploaded, self.ac90)
        self.assertIsNotNone(warning)
        self.assertIn("M-AP500", warning)
        self.assertIn("M-AC90", warning)

    def test_no_machine_label_but_mentions_another_machine_code(self):
        """Fallback heuristic: no explicit 'Machine:' label, but the page
        mentions a different registered machine and never the selected one."""
        uploaded = self._upload("Troubleshooting reference. See M-AP500 wiring diagram for details.")
        warning = check_machine_mismatch(uploaded, self.ac90)
        self.assertIsNotNone(warning)
        self.assertIn("M-AP500", warning)

    def test_ambiguous_text_with_no_machine_reference_is_not_blocked(self):
        """If the document's machine identity can't be determined at all,
        the check must allow the existing workflow (no warning)."""
        uploaded = self._upload("General safety notes. No machine-specific content on this page.")
        warning = check_machine_mismatch(uploaded, self.ac90)
        self.assertIsNone(warning)

    def test_corrupt_file_never_blocks_the_real_upload_path(self):
        """A failed *check* (e.g. an unreadable file) must return None, not
        raise - the real upload endpoint is what should reject a bad file."""
        bad_file = SimpleUploadedFile("bad.pdf", b"not a real pdf", content_type="application/pdf")
        warning = check_machine_mismatch(bad_file, self.ac90)
        self.assertIsNone(warning)


class ManualMachineCheckViewPermissionTests(TestCase):
    """#12: the advisory check endpoint is Administrator-only, like the real upload."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()

    def test_anonymous_cannot_use_the_check_endpoint(self):
        response = self.client.post("/api/manuals/check-machine/", {"machine": self.machine.id})
        self.assertIn(response.status_code, (401, 403))

    def test_engineer_cannot_use_the_check_endpoint(self):
        engineer = make_user("eng_machine_check", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.post("/api/manuals/check-machine/", {"machine": self.machine.id})
        self.assertEqual(response.status_code, 403)

    def test_admin_can_use_the_check_endpoint_and_it_writes_nothing(self):
        admin = make_user("admin_machine_check", "admin")
        self.client.force_authenticate(admin)
        path = pdf_fixtures.make_valid_pdf(["Generic content with no machine reference."])
        try:
            with open(path, "rb") as f:
                response = self.client.post(
                    "/api/manuals/check-machine/",
                    {"machine": self.machine.id, "file": SimpleUploadedFile("m.pdf", f.read(), content_type="application/pdf")},
                    format="multipart",
                )
            self.assertEqual(response.status_code, 200)
            self.assertIn("warning", response.data)
            self.assertEqual(Manual.objects.count(), 0)  # purely advisory - no record created
        finally:
            pdf_fixtures.cleanup(path)


class InsufficientEvidenceMessageTests(TestCase):
    """#9: the message names the actual selected machine and points at the real cause."""

    def test_message_names_the_selected_machine(self):
        machine = make_machine("M-AC90", "AC-90", "Air compressor")
        message = build_insufficient_evidence_message(machine)
        self.assertIn("M-AC90", message)
        self.assertIn("uploaded and assigned to this machine", message)

    def test_different_machines_get_different_messages(self):
        a = make_machine("M-A", "A-1", "Machine A")
        b = make_machine("M-B", "B-1", "Machine B")
        self.assertNotEqual(build_insufficient_evidence_message(a), build_insufficient_evidence_message(b))


class ReassignmentAndReingestionTests(TestCase):
    """#3, #6, #7, #8: machine isolation holds before AND after a confirmed
    reassignment + re-ingestion, and an already-created Question's machine
    is never retroactively changed."""

    def setUp(self):
        self.ac90 = make_machine("M-AC90", "AC-90", "Air compressor")
        self.ap500 = make_machine("M-AP500", "AP-500", "Industrial air compressor")
        self._cleanup_paths = []

    def tearDown(self):
        pdf_fixtures.cleanup(*self._cleanup_paths)

    def test_reassigning_a_mis_tagged_manual_and_reingesting_fixes_retrieval(self):
        """Rebuilds the exact real scenario found during investigation: a
        manual uploaded under the wrong machine, then corrected."""
        path = pdf_fixtures.make_valid_pdf(["AP 500 manual. Machine: M-AP500/AP-500. Alarm E12 means high pressure."])
        self._cleanup_paths.append(path)

        # Step 1: mis-tagged at upload time (machine=AC90, content is AP500's).
        manual = make_manual(self.ac90, document_id="AP-MNT-01")
        with open(path, "rb") as f:
            manual.file.save("ap-mnt-01.pdf", File(f), save=True)
        ingest_manual(manual)
        manual.refresh_from_db()
        self.assertEqual(Chunk.objects.filter(manual=manual).first().machine_id, self.ac90.id)

        # Step 2: an Engineer asks a question on AC90 - the mis-tagged
        # manual's unrelated content must not leak into a DIFFERENT
        # question's machine-isolated retrieval (machine isolation itself
        # was never broken - the data was simply wrong).
        results_before = retrieve_chunks(self.ac90, "What does E12 mean?")
        self.assertTrue(any(r.chunk.manual_id == manual.id for r in results_before))  # confirms the bug existed

        # Step 3: Administrator confirms the real machine and corrects it.
        manual.machine = self.ap500
        manual.save(update_fields=["machine"])
        ingest_manual(manual)  # re-ingest so chunks/embeddings get the correct machine
        manual.refresh_from_db()

        self.assertEqual(manual.machine_id, self.ap500.id)
        for chunk in Chunk.objects.filter(manual=manual):
            self.assertEqual(chunk.machine_id, self.ap500.id)

        # Step 4: AC90 retrieval no longer sees this manual at all.
        results_after = retrieve_chunks(self.ac90, "What does E12 mean?")
        self.assertFalse(any(r.chunk.manual_id == manual.id for r in results_after))

        # Step 5: AP500 retrieval now correctly finds it.
        results_ap500 = retrieve_chunks(self.ap500, "What does E12 mean?")
        self.assertTrue(any(r.chunk.manual_id == manual.id for r in results_ap500))

    def test_existing_question_history_is_preserved_across_a_reassignment(self):
        """#6, #11: reassigning/re-ingesting a manual must not touch
        already-created Question records or their original machine."""
        machine = make_machine()
        manual = make_manual(machine, document_id="SOME-DOC")
        make_chunk(manual, "Some content.", page_number=1)
        engineer = make_user("eng_reassign", "engineer")
        question = Question.objects.create(
            user=engineer, machine=machine, question_text="old question",
            answer_text="old answer", status=Question.Status.ANSWERED, llm_provider="local",
        )
        original_machine_id = question.machine_id

        other_machine = make_machine("M-OTHER", "OTHER-1", "Other machine")
        manual.machine = other_machine
        manual.save(update_fields=["machine"])

        question.refresh_from_db()
        self.assertEqual(question.machine_id, original_machine_id)  # never retroactively changed
        self.assertTrue(Question.objects.filter(id=question.id).exists())
