"""
Tests for Administrator machine/manual deletion and archival
(manuals/views_api.py: MachineDeleteView, ManualDeleteView).

Django TestCase wraps each test in a transaction that is rolled back, not
committed, so transaction.on_commit() callbacks (used for the actual
storage-file deletion, deliberately scheduled only after a successful DB
commit) never fire on their own in a test. self.captureOnCommitCallbacks()
is used wherever a test needs to prove the file was actually removed.
"""

from django.core.files.base import ContentFile
from django.test import TestCase
from rest_framework.test import APIClient

from manuals.models import Chunk, Citation, GroqCitation, Machine, ManualPage, Question
from manuals.services import retrieve_chunks

from .helpers import make_chunk, make_machine, make_manual, make_user


class MachineDeletionTests(TestCase):
    """#8, #10, #13, #14."""

    def setUp(self):
        self.client = APIClient()
        self.admin = make_user("admin_del", "admin")

    def test_admin_can_delete_a_machine_with_no_history(self):
        """#8: an eligible machine (no questions) is permanently deleted."""
        machine = make_machine("M-DEL", "DEL-1", "Deletable machine")
        manual = make_manual(machine)
        make_chunk(manual, "Some content.", page_number=1)

        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.delete(f"/api/machines/{machine.id}/")
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Machine.objects.filter(id=machine.id).exists())

    def test_deleting_a_machine_removes_its_manuals_and_chunks(self):
        """#10: dependent manuals/chunks are handled (cascade-removed) correctly."""
        machine = make_machine("M-DEL2", "DEL-2", "Deletable machine")
        manual = make_manual(machine)
        chunk = make_chunk(manual, "Some content.", page_number=1)

        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.delete(f"/api/machines/{machine.id}/")

        self.assertFalse(Chunk.objects.filter(id=chunk.id).exists())
        self.assertEqual(ManualPage.objects.filter(manual_id=manual.id).count(), 0)

    def test_deleting_a_machine_removes_its_manuals_pdf_files(self):
        machine = make_machine("M-DEL3", "DEL-3", "Deletable machine")
        manual = make_manual(machine)
        storage = manual.file.storage
        file_name = manual.file.name
        self.assertTrue(storage.exists(file_name))

        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.delete(f"/api/machines/{machine.id}/")

        self.assertFalse(storage.exists(file_name))

    def test_engineer_cannot_delete_a_machine(self):
        """#13."""
        machine = make_machine()
        engineer = make_user("eng_del", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.delete(f"/api/machines/{machine.id}/")
        self.assertEqual(response.status_code, 403)
        self.assertTrue(Machine.objects.filter(id=machine.id).exists())

    def test_supervisor_cannot_delete_a_machine(self):
        """#14."""
        machine = make_machine()
        supervisor = make_user("sup_del", "supervisor")
        self.client.force_authenticate(supervisor)
        response = self.client.delete(f"/api/machines/{machine.id}/")
        self.assertEqual(response.status_code, 403)
        self.assertTrue(Machine.objects.filter(id=machine.id).exists())

    def test_anonymous_cannot_delete_a_machine(self):
        machine = make_machine()
        response = self.client.delete(f"/api/machines/{machine.id}/")
        self.assertIn(response.status_code, (401, 403))

    def test_deleting_an_unknown_machine_returns_404(self):
        self.client.force_authenticate(self.admin)
        response = self.client.delete("/api/machines/999999/")
        self.assertEqual(response.status_code, 404)


class MachineArchivalTests(TestCase):
    """#9, #11, #12: a machine with question history is archived, not deleted."""

    def setUp(self):
        self.client = APIClient()
        self.admin = make_user("admin_archive", "admin")
        self.machine = make_machine("M-HIST", "HIST-1", "Machine with history")
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "Some content.", page_number=1)
        self.engineer = make_user("eng_hist", "engineer")
        self.question = Question.objects.create(
            user=self.engineer, machine=self.machine, question_text="old question",
            answer_text="old answer", status=Question.Status.ANSWERED, llm_provider="local",
        )

    def test_machine_with_history_is_archived_not_deleted(self):
        """#9."""
        self.client.force_authenticate(self.admin)
        response = self.client.delete(f"/api/machines/{self.machine.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["archived"])
        self.machine.refresh_from_db()
        self.assertTrue(self.machine.is_archived)
        # Nothing was destroyed.
        self.assertTrue(Question.objects.filter(id=self.question.id).exists())
        self.assertTrue(self.manual.__class__.objects.filter(id=self.manual.id).exists())

    def test_archived_machine_is_excluded_from_new_question_submission(self):
        """#11: the ask-form/serializer boundary, not just a hidden <select> option."""
        self.machine.is_archived = True
        self.machine.save(update_fields=["is_archived"])
        self.client.force_authenticate(self.engineer)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "new question"}, format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_archived_machine_still_appears_in_machine_list_with_flag(self):
        self.machine.is_archived = True
        self.machine.save(update_fields=["is_archived"])
        self.client.force_authenticate(self.admin)
        response = self.client.get("/api/machines/")
        matching = [m for m in response.data if m["id"] == self.machine.id]
        self.assertEqual(len(matching), 1)
        self.assertTrue(matching[0]["is_archived"])

    def test_historical_questions_remain_accessible_after_archiving(self):
        """#12."""
        self.client.force_authenticate(self.admin)
        self.client.delete(f"/api/machines/{self.machine.id}/")  # archives (has history)

        self.client.force_authenticate(self.engineer)
        response = self.client.get("/api/questions/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(q["id"] == self.question.id for q in response.data))

    def test_machine_list_annotates_manual_and_question_counts(self):
        self.client.force_authenticate(self.admin)
        response = self.client.get("/api/machines/")
        matching = [m for m in response.data if m["id"] == self.machine.id][0]
        self.assertEqual(matching["manual_count"], 1)
        self.assertEqual(matching["question_count"], 1)

    def test_already_archived_machine_delete_is_idempotent(self):
        self.machine.is_archived = True
        self.machine.save(update_fields=["is_archived"])
        self.client.force_authenticate(self.admin)
        response = self.client.delete(f"/api/machines/{self.machine.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["archived"])


class MachineUnarchiveTests(TestCase):
    """MachineUnarchiveView: reversing an archive back to Active."""

    def setUp(self):
        self.client = APIClient()
        self.admin = make_user("admin_unarchive", "admin")
        self.machine = make_machine("M-UNARCH", "UNARCH-1", "Archived machine")
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "Some content.", page_number=1)
        self.engineer = make_user("eng_unarch", "engineer")
        self.question = Question.objects.create(
            user=self.engineer, machine=self.machine, question_text="old question",
            answer_text="old answer", status=Question.Status.ANSWERED, llm_provider="local",
        )
        self.machine.is_archived = True
        self.machine.save(update_fields=["is_archived"])

    def test_admin_can_unarchive_a_machine(self):
        self.client.force_authenticate(self.admin)
        response = self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["archived"])
        self.machine.refresh_from_db()
        self.assertFalse(self.machine.is_archived)

    def test_unarchived_machine_accepts_new_questions_again(self):
        self.client.force_authenticate(self.admin)
        self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.client.force_authenticate(self.engineer)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "new question"}, format="json"
        )
        self.assertNotEqual(response.status_code, 400)

    def test_unarchiving_preserves_history(self):
        self.client.force_authenticate(self.admin)
        self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertTrue(Question.objects.filter(id=self.question.id).exists())
        self.assertTrue(self.manual.__class__.objects.filter(id=self.manual.id).exists())

    def test_unarchiving_a_non_archived_machine_is_idempotent(self):
        self.machine.is_archived = False
        self.machine.save(update_fields=["is_archived"])
        self.client.force_authenticate(self.admin)
        response = self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["archived"])

    def test_engineer_cannot_unarchive_a_machine(self):
        self.client.force_authenticate(self.engineer)
        response = self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertEqual(response.status_code, 403)
        self.machine.refresh_from_db()
        self.assertTrue(self.machine.is_archived)

    def test_supervisor_cannot_unarchive_a_machine(self):
        supervisor = make_user("sup_unarch", "supervisor")
        self.client.force_authenticate(supervisor)
        response = self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertEqual(response.status_code, 403)

    def test_anonymous_cannot_unarchive_a_machine(self):
        response = self.client.post(f"/api/machines/{self.machine.id}/unarchive/")
        self.assertIn(response.status_code, (401, 403))

    def test_unarchiving_an_unknown_machine_returns_404(self):
        self.client.force_authenticate(self.admin)
        response = self.client.post("/api/machines/999999/unarchive/")
        self.assertEqual(response.status_code, 404)


class ManualDeletionTests(TestCase):
    """#15, #16, #17, #18, #19, #20, #21, #22, #23, #24."""

    def setUp(self):
        self.client = APIClient()
        self.admin = make_user("admin_mandel", "admin")
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="DOC-DEL")
        make_chunk(self.manual, "The AF-10 filter should be checked weekly.", page_number=1)

    def test_admin_can_delete_a_manual(self):
        """#15."""
        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(response.status_code, 204)
        self.assertFalse(self.manual.__class__.objects.filter(id=self.manual.id).exists())

    def test_deleting_a_manual_removes_its_chunks(self):
        """#16: chunks/embeddings removed from active retrieval."""
        chunk_id = Chunk.objects.get(manual=self.manual).id
        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertFalse(Chunk.objects.filter(id=chunk_id).exists())

    def test_deleting_a_manual_removes_its_pdf_file(self):
        """#17."""
        storage = self.manual.file.storage
        file_name = self.manual.file.name
        self.assertTrue(storage.exists(file_name))
        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertFalse(storage.exists(file_name))

    def test_deleted_manual_cannot_be_retrieved_by_rag(self):
        """#18."""
        with_results = retrieve_chunks(self.machine, "How often should the AF-10 filter be checked?")
        self.assertGreaterEqual(len(with_results), 1)

        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.delete(f"/api/manuals/{self.manual.id}/")

        after_results = retrieve_chunks(self.machine, "How often should the AF-10 filter be checked?")
        self.assertEqual(len(after_results), 0)

    def test_historical_answer_text_and_citations_survive_manual_deletion(self):
        """#19, #20, #21: the Question's answer text and citation snapshot survive;
        the citation clearly indicates the manual is gone and the PDF link is disabled."""
        engineer = make_user("eng_mandel", "engineer")
        question = Question.objects.create(
            user=engineer, machine=self.machine, question_text="What is the check interval?",
            answer_text="The AF-10 filter should be checked weekly.", status=Question.Status.ANSWERED,
            llm_provider="local",
        )
        citation = Citation.objects.create(
            question=question, manual=self.manual, manual_document_id=self.manual.document_id,
            manual_revision=self.manual.revision, page_number=1,
            excerpt="The AF-10 filter should be checked weekly.", score=0.5,
        )

        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(response.status_code, 204)

        question.refresh_from_db()
        self.assertEqual(question.answer_text, "The AF-10 filter should be checked weekly.")  # #19 unchanged

        citation.refresh_from_db()
        self.assertIsNone(citation.manual_id)  # SET_NULL, not cascaded away
        self.assertEqual(citation.manual_document_id, "DOC-DEL")  # #20 snapshot survives
        self.assertEqual(citation.page_number, 1)
        self.assertEqual(citation.excerpt, "The AF-10 filter should be checked weekly.")

        from manuals.serializers import CitationSerializer
        data = CitationSerializer(citation, context={}).data
        self.assertTrue(data["manual_deleted"])  # #21
        self.assertIsNone(data["manual_file_url"])  # #21: never a broken link
        self.assertEqual(data["document_id"], "DOC-DEL")

    def test_engineer_cannot_delete_a_manual(self):
        """#22."""
        engineer = make_user("eng_mandel2", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(response.status_code, 403)
        self.assertTrue(self.manual.__class__.objects.filter(id=self.manual.id).exists())

    def test_supervisor_cannot_delete_a_manual(self):
        """#22."""
        supervisor = make_user("sup_mandel", "supervisor")
        self.client.force_authenticate(supervisor)
        response = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(response.status_code, 403)

    def test_anonymous_cannot_delete_a_manual(self):
        """#22."""
        response = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertIn(response.status_code, (401, 403))

    def test_deleting_an_unknown_manual_returns_404_cleanly(self):
        """#23: no internal error/path exposure for a bad id."""
        self.client.force_authenticate(self.admin)
        response = self.client.delete("/api/manuals/999999/")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(":\\", str(response.content))
        self.assertNotIn("Traceback", str(response.content))

    def test_repeated_deletion_request_is_handled_cleanly(self):
        """#24: deleting the same manual twice never crashes or double-deletes a file."""
        self.client.force_authenticate(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            first = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(first.status_code, 204)

        second = self.client.delete(f"/api/manuals/{self.manual.id}/")
        self.assertEqual(second.status_code, 404)
