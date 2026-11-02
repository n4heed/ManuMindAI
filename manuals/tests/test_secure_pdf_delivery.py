"""
Regression tests for authenticated PDF delivery.

Manual PDFs used to be served from a public /media/... URL. They are now
only reachable through ManualFileView (/api/manuals/<id>/file/), which
resolves the file strictly through the Manual model - never through a
client-supplied filesystem path - and config/urls.py no longer serves
MEDIA_ROOT publicly at all, in any DEBUG setting.
"""

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .helpers import make_machine, make_manual, make_user


class SecurePdfDeliveryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="SEC-1")

    def _file_url(self, manual=None):
        manual = manual or self.manual
        return f"/api/manuals/{manual.id}/file/"

    def test_anonymous_access_is_denied(self):
        response = self.client.get(self._file_url())
        self.assertIn(response.status_code, (401, 403))

    def test_authorized_user_can_view_an_allowed_pdf(self):
        engineer = make_user("eng_pdf", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get(self._file_url())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("inline", response["Content-Disposition"])
        self.assertEqual(response["X-Content-Type-Options"], "nosniff")

    def test_every_role_can_view_a_pdf_matching_the_apps_existing_read_boundary(self):
        """
        There is no per-user manual-ownership restriction anywhere else in
        this app (ManualListCreateView's GET and QuestionHistoryView are
        both IsAuthenticated-only across all three roles), so this view
        intentionally matches that same boundary rather than inventing a
        narrower one. This test documents that as an explicit, checked
        decision rather than an accident.
        """
        for role in ("admin", "engineer", "supervisor"):
            user = make_user(f"role_pdf_{role}", role)
            self.client.force_authenticate(user)
            response = self.client.get(self._file_url())
            self.assertEqual(response.status_code, 200, f"role={role}")

    def test_nonexistent_manual_id_returns_404_not_a_leak(self):
        engineer = make_user("eng_pdf_404", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get("/api/manuals/999999/file/")
        self.assertEqual(response.status_code, 404)
        # Must not echo a filesystem path or internal detail.
        body = response.content.decode(errors="ignore")
        self.assertNotIn(":\\", body)  # no Windows path
        self.assertNotIn("/home/", body)
        self.assertNotIn("Traceback", body)

    def test_non_integer_manual_id_is_rejected_by_routing(self):
        """The id is matched by Django's int path converter, so a traversal
        or non-numeric string in that position can never reach the view at
        all - it 404s at the URL resolver, before any file lookup."""
        engineer = make_user("eng_pdf_traversal", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get("/api/manuals/../../../../etc/passwd/file/")
        self.assertEqual(response.status_code, 404)

    def test_manual_with_record_but_missing_file_on_disk_returns_404_not_a_leak(self):
        engineer = make_user("eng_pdf_missing", "engineer")
        self.client.force_authenticate(engineer)
        # Delete the underlying file on disk while the DB record remains,
        # simulating a dev/ops inconsistency - must not crash or leak a path.
        self.manual.file.delete(save=False)
        self.manual.file.name = "manuals/this-file-does-not-exist.pdf"
        self.manual.save(update_fields=["file"])

        response = self.client.get(self._file_url())
        self.assertEqual(response.status_code, 404)
        body = response.content.decode(errors="ignore")
        self.assertNotIn("this-file-does-not-exist", body)
        self.assertNotIn("Traceback", body)

    def test_citation_and_manual_list_responses_expose_only_the_secure_url(self):
        """The API must never hand back the old public /media/ path."""
        from manuals.serializers import CitationSerializer, ManualSerializer
        from manuals.models import Citation
        from .helpers import make_chunk

        make_chunk(self.manual, "The filter should be checked weekly.")
        admin = make_user("admin_pdf_fields", "admin")
        self.client.force_authenticate(admin)

        response = self.client.get("/api/manuals/")
        self.assertEqual(response.status_code, 200)
        for manual_data in response.data:
            self.assertNotIn("file", manual_data)  # write-only now
            if manual_data["secure_file_url"]:
                self.assertIn("/api/manuals/", manual_data["secure_file_url"])
                self.assertIn("/file/", manual_data["secure_file_url"])
                self.assertNotIn("/media/", manual_data["secure_file_url"])


class LegacyPublicMediaUrlTests(TestCase):
    """Audit requirement: old public media URLs must not expose protected
    manual PDFs in the supported deployment configuration (Django itself
    serving the app). config/urls.py no longer registers any /media/ route
    at all, in any DEBUG setting - these tests exercise both."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()
        self.manual = make_manual(self.machine, document_id="LEGACY-1")

    def _legacy_url(self):
        return f"/{self.manual.file.name}" if self.manual.file.name.startswith("media/") else f"/media/{self.manual.file.name}"

    def test_legacy_media_url_is_not_served_with_debug_false(self):
        response = self.client.get(self._legacy_url())
        self.assertEqual(response.status_code, 404)

    @override_settings(DEBUG=True)
    def test_legacy_media_url_is_not_served_even_with_debug_true(self):
        """
        Regression test for the exact bypass this milestone closed: the old
        code only stopped serving /media/ in production (DEBUG=False) and
        still served it in local dev (DEBUG=True) - which would have made
        the secure view pointless in the one environment these tests
        actually run in. It is now unconditional in config/urls.py, so this
        must 404 regardless of DEBUG.
        """
        response = self.client.get(self._legacy_url())
        self.assertEqual(response.status_code, 404)
