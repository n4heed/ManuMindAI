from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from manuals.models import Manual, Question

from .helpers import MINIMAL_PDF_BYTES, make_chunk, make_machine, make_manual, make_user


class UploadRestrictionTests(TestCase):
    """Required check #1: Engineers cannot upload manuals."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()

    def _upload_payload(self):
        return {
            "machine": self.machine.id,
            "document_id": "DOC-1",
            "revision": "1.0",
            "file": SimpleUploadedFile("doc.pdf", MINIMAL_PDF_BYTES, content_type="application/pdf"),
        }

    def test_engineer_cannot_upload_manual(self):
        engineer = make_user("eng_upload", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.post("/api/manuals/", self._upload_payload(), format="multipart")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(Manual.objects.count(), 0)

    def test_supervisor_cannot_upload_manual(self):
        supervisor = make_user("sup_upload", "supervisor")
        self.client.force_authenticate(supervisor)
        response = self.client.post("/api/manuals/", self._upload_payload(), format="multipart")
        self.assertEqual(response.status_code, 403)

    def test_unauthenticated_user_cannot_upload_manual(self):
        response = self.client.post("/api/manuals/", self._upload_payload(), format="multipart")
        self.assertIn(response.status_code, (401, 403))

    def test_admin_can_upload_manual(self):
        admin = make_user("admin_upload", "admin")
        self.client.force_authenticate(admin)
        response = self.client.post("/api/manuals/", self._upload_payload(), format="multipart")
        # Ingestion itself will fail in this environment if PyMuPDF can't load, but the
        # request must be *accepted* (not blocked by role permission) and persisted.
        self.assertIn(response.status_code, (201,))
        self.assertEqual(Manual.objects.count(), 1)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class QuestionRolePermissionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()

    def test_engineer_can_ask_question(self):
        engineer = make_user("eng_ask", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "What is E17?"}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Question.objects.count(), 1)

    def test_admin_cannot_ask_question(self):
        admin = make_user("admin_ask", "admin")
        self.client.force_authenticate(admin)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "What is E17?"}, format="json"
        )
        self.assertEqual(response.status_code, 403)

    def test_supervisor_cannot_ask_question(self):
        supervisor = make_user("sup_ask", "supervisor")
        self.client.force_authenticate(supervisor)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "What is E17?"}, format="json"
        )
        self.assertEqual(response.status_code, 403)


@override_settings(RAG_SIMILARITY_THRESHOLD=0.08, RAG_TOP_K=5, LLM_ENABLED=False, GROQ_API_KEY="")
class QuestionHistoryScopeTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()
        self.manual = make_manual(self.machine)
        make_chunk(self.manual, "E17 indicates a hydraulic pressure sensor fault.")

        self.engineer_a = make_user("eng_a", "engineer")
        self.engineer_b = make_user("eng_b", "engineer")
        self.supervisor = make_user("sup_history", "supervisor")

        self.client.force_authenticate(self.engineer_a)
        self.client.post("/api/questions/ask/", {"machine": self.machine.id, "question_text": "What is E17?"}, format="json")

        self.client.force_authenticate(self.engineer_b)
        self.client.post("/api/questions/ask/", {"machine": self.machine.id, "question_text": "What is E17?"}, format="json")

    def test_engineer_sees_only_their_own_history(self):
        self.client.force_authenticate(self.engineer_a)
        response = self.client.get("/api/questions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["asked_by"], "eng_a")

    def test_supervisor_sees_all_history_read_only(self):
        self.client.force_authenticate(self.supervisor)
        response = self.client.get("/api/questions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_supervisor_cannot_upload_or_ask(self):
        self.client.force_authenticate(self.supervisor)
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "x"}, format="json"
        )
        self.assertEqual(response.status_code, 403)


class MachineRegistrationPermissionTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_admin_can_register_machine(self):
        admin = make_user("admin_machine", "admin")
        self.client.force_authenticate(admin)
        response = self.client.post(
            "/api/machines/", {"machine_id": "M-NEW", "model": "NEW-1", "name": "New machine"}, format="json"
        )
        self.assertEqual(response.status_code, 201)

    def test_engineer_cannot_register_machine(self):
        engineer = make_user("eng_machine", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.post(
            "/api/machines/", {"machine_id": "M-NEW2", "model": "NEW-2", "name": "New machine 2"}, format="json"
        )
        self.assertEqual(response.status_code, 403)

    def test_any_authenticated_user_can_list_machines(self):
        make_machine("M-LIST", "LIST-1", "Listable machine")
        engineer = make_user("eng_list", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get("/api/machines/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
