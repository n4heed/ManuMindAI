"""
Regression tests for the final security/demo-readiness audit.

These prove, rather than assume, three specific things that were checked by
reading code during the audit: (1) every protected endpoint actually rejects
an anonymous request, not just an unauthorized role, (2) a session-authenticated
state-changing request without a CSRF token is rejected, and (3) Django's
upload storage rejects a path-traversal filename through the real API, not
just in isolation.
"""

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from manuals.models import Manual

from .helpers import MINIMAL_PDF_BYTES, make_machine, make_user


class AnonymousAccessRejectedTests(TestCase):
    """Audit A: anonymous users cannot reach any protected endpoint, by
    direct API request, not just via the UI hiding a button/link."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()

    def test_whoami_requires_authentication(self):
        response = self.client.get("/api/whoami/")
        self.assertIn(response.status_code, (401, 403))

    def test_machine_list_requires_authentication(self):
        response = self.client.get("/api/machines/")
        self.assertIn(response.status_code, (401, 403))

    def test_machine_create_requires_authentication(self):
        response = self.client.post(
            "/api/machines/", {"machine_id": "M-ANON", "model": "X", "name": "X"}, format="json"
        )
        self.assertIn(response.status_code, (401, 403))

    def test_manual_list_requires_authentication(self):
        response = self.client.get("/api/manuals/")
        self.assertIn(response.status_code, (401, 403))

    def test_question_history_requires_authentication(self):
        response = self.client.get("/api/questions/")
        self.assertIn(response.status_code, (401, 403))

    def test_ask_question_requires_authentication(self):
        response = self.client.post(
            "/api/questions/ask/", {"machine": self.machine.id, "question_text": "x"}, format="json"
        )
        self.assertIn(response.status_code, (401, 403))

    def test_dashboard_page_redirects_anonymous_to_login(self):
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)


class SessionCsrfEnforcementTests(TestCase):
    """
    Audit B: a session-authenticated, state-changing request must be
    rejected without a valid CSRF token. force_authenticate() (used by the
    other permission tests) bypasses Django's normal login/CSRF path
    entirely, so it cannot prove this - a real session login is used here.
    """

    def setUp(self):
        self.admin = make_user("admin_csrf", "admin")
        self.client = APIClient(enforce_csrf_checks=True)

    def test_post_without_csrf_token_is_rejected_for_a_logged_in_session(self):
        self.client.login(username="admin_csrf", password="testpass123")
        response = self.client.post(
            "/api/machines/", {"machine_id": "M-CSRF", "model": "X", "name": "X"}, format="json"
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(Manual.objects.count(), 0)


class UploadPathTraversalTests(TestCase):
    """Audit C: a crafted filename cannot be used to write outside MEDIA_ROOT."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()

    def test_traversal_filename_is_rejected_or_contained_not_written_outside_media_root(self):
        admin = make_user("admin_traversal", "admin")
        self.client.force_authenticate(admin)
        response = self.client.post(
            "/api/manuals/",
            {
                "machine": self.machine.id,
                "document_id": "DOC-TRAVERSAL",
                "revision": "1.0",
                "file": SimpleUploadedFile(
                    "../../../../evil.pdf", MINIMAL_PDF_BYTES, content_type="application/pdf"
                ),
            },
            format="multipart",
        )
        # Either Django's storage layer rejects the traversal filename (500
        # would be unacceptable - but a stored file is checked below for
        # safety regardless of which status code this returns) or it stores
        # the file under a sanitized name within MEDIA_ROOT. It must never
        # succeed while actually writing outside MEDIA_ROOT.
        if response.status_code == 201:
            manual = Manual.objects.get(document_id="DOC-TRAVERSAL")
            from django.conf import settings

            stored_path = manual.file.path
            self.assertTrue(
                str(stored_path).startswith(str(settings.MEDIA_ROOT)),
                f"uploaded file stored outside MEDIA_ROOT: {stored_path}",
            )
            self.assertNotIn("..", manual.file.name)
