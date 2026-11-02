"""
Ingestion tests against REAL PDF files (built with PyMuPDF itself), not
mocked extraction. Exercises the actual `fitz.open()` / `get_text()` code
path. See test_ingestion_service.py for the complementary mocked tests
(faster, used for ingest_manual's status-transition logic in isolation).

Note: a true zero-page PDF is not covered here - PyMuPDF itself refuses to
save a document with zero pages ("cannot save with zero pages"), so that
specific branch in extract_pages() can only be exercised with a mocked
page_count, which test_ingestion_service.py already does.
"""

from django.core.files import File
from django.test import TestCase

from manuals.models import Chunk, Manual, ManualPage
from manuals.services import IngestionError, extract_pages, ingest_manual

from . import pdf_fixtures
from .helpers import make_machine, make_manual


class RealPdfExtractionTests(TestCase):
    """Calls extract_pages() directly against real files - no mocking."""

    def setUp(self):
        self._cleanup_paths = []

    def tearDown(self):
        pdf_fixtures.cleanup(*self._cleanup_paths)

    def _track(self, path):
        self._cleanup_paths.append(path)
        return path

    def test_valid_pdf_preserves_one_based_page_numbers_and_content(self):
        path = self._track(
            pdf_fixtures.make_valid_pdf(["Page One Content Alpha", "Page Two Content Bravo", "Page Three Content Charlie"])
        )
        pages = extract_pages(path)
        self.assertEqual([p[0] for p in pages], [1, 2, 3])  # one-based, in order
        self.assertIn("Page One Content Alpha", pages[0][1])
        self.assertIn("Page Two Content Bravo", pages[1][1])
        self.assertIn("Page Three Content Charlie", pages[2][1])

    def test_corrupt_pdf_raises_ingestion_error(self):
        path = self._track(pdf_fixtures.make_corrupt_pdf())
        with self.assertRaises(IngestionError) as ctx:
            extract_pages(path)
        self.assertIn("Could not open PDF", str(ctx.exception))

    def test_encrypted_pdf_raises_ingestion_error(self):
        path = self._track(pdf_fixtures.make_encrypted_pdf())
        with self.assertRaises(IngestionError) as ctx:
            extract_pages(path)
        self.assertIn("encrypted", str(ctx.exception).lower())

    def test_image_only_pdf_reports_no_extractable_text(self):
        path = self._track(pdf_fixtures.make_image_only_pdf())
        with self.assertRaises(IngestionError) as ctx:
            extract_pages(path)
        message = str(ctx.exception).lower()
        self.assertIn("no extractable text", message)
        # Must not claim OCR support - we don't implement OCR.
        self.assertNotIn("ocr succeeded", message)
        self.assertNotIn("extracted via ocr", message)


class RealPdfIngestManualTests(TestCase):
    """
    Calls ingest_manual() end-to-end (real extraction, real chunking, real
    local embeddings) and checks the database is left in a consistent state
    either way - never 'looks successful' after a real failure, and never
    leaves orphaned rows after a real success followed by failure.
    """

    def setUp(self):
        self.machine = make_machine()
        self._cleanup_paths = []

    def tearDown(self):
        pdf_fixtures.cleanup(*self._cleanup_paths)

    def _attach_real_file(self, manual, path, name="test.pdf"):
        with open(path, "rb") as f:
            manual.file.save(name, File(f), save=True)

    def test_valid_pdf_end_to_end_creates_correct_pages_and_chunks(self):
        path = pdf_fixtures.make_valid_pdf(["Alpha content here.", "Bravo content here.", "Charlie content here."])
        self._cleanup_paths.append(path)

        manual = make_manual(self.machine, document_id="REAL-1")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, path)

        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.COMPLETED)
        self.assertEqual(manual.page_count, 3)
        self.assertEqual(manual.error_message, "")

        pages = list(ManualPage.objects.filter(manual=manual).order_by("page_number"))
        self.assertEqual([p.page_number for p in pages], [1, 2, 3])
        self.assertIn("Alpha", pages[0].text)
        self.assertIn("Bravo", pages[1].text)
        self.assertIn("Charlie", pages[2].text)

        self.assertGreater(Chunk.objects.filter(manual=manual).count(), 0)

    def test_corrupt_pdf_end_to_end_fails_cleanly_with_no_orphaned_rows(self):
        path = pdf_fixtures.make_corrupt_pdf()
        self._cleanup_paths.append(path)

        manual = make_manual(self.machine, document_id="REAL-2")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, path)

        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.FAILED)
        self.assertNotEqual(manual.error_message, "")
        self.assertEqual(manual.page_count, 0)
        # A failed ingestion must not look successful or leave partial data behind.
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 0)
        self.assertEqual(Chunk.objects.filter(manual=manual).count(), 0)

    def test_encrypted_pdf_end_to_end_fails_cleanly(self):
        path = pdf_fixtures.make_encrypted_pdf()
        self._cleanup_paths.append(path)

        manual = make_manual(self.machine, document_id="REAL-3")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, path)

        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.FAILED)
        self.assertIn("encrypted", manual.error_message.lower())
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 0)

    def test_image_only_pdf_end_to_end_fails_cleanly_and_does_not_claim_ocr(self):
        path = pdf_fixtures.make_image_only_pdf()
        self._cleanup_paths.append(path)

        manual = make_manual(self.machine, document_id="REAL-4")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, path)

        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.FAILED)
        self.assertIn("no extractable text", manual.error_message.lower())
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 0)
        self.assertEqual(Chunk.objects.filter(manual=manual).count(), 0)

    def test_failed_reingestion_clears_stale_content_from_a_prior_success(self):
        """
        Regression test for a real bug this test suite found: re-ingesting a
        manual that was PREVIOUSLY successfully indexed, with a now-bad file,
        used to leave the old pages/chunks in place while the status flipped
        to 'failed' - meaning stale content could still be retrieved under a
        manual that looked broken. A 'failed' manual must have zero pages,
        zero chunks, and page_count 0, full stop.
        """
        good_path = pdf_fixtures.make_valid_pdf(["Original valid content."])
        bad_path = pdf_fixtures.make_corrupt_pdf()
        self._cleanup_paths += [good_path, bad_path]

        manual = make_manual(self.machine, document_id="REAL-6")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, good_path)
        ingest_manual(manual)
        manual.refresh_from_db()
        self.assertEqual(manual.status, Manual.Status.COMPLETED)
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 1)
        self.assertGreater(Chunk.objects.filter(manual=manual).count(), 0)

        self._attach_real_file(manual, bad_path, name="broken.pdf")
        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.FAILED)
        self.assertEqual(manual.page_count, 0)
        self.assertIsNone(manual.processed_at)
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 0)
        self.assertEqual(Chunk.objects.filter(manual=manual).count(), 0)

    def test_reingesting_over_a_previously_failed_manual_with_a_valid_file_succeeds(self):
        """A common real workflow: admin uploads a bad file, fixes it, re-uploads."""
        bad_path = pdf_fixtures.make_corrupt_pdf()
        good_path = pdf_fixtures.make_valid_pdf(["Good content now."])
        self._cleanup_paths += [bad_path, good_path]

        manual = make_manual(self.machine, document_id="REAL-5")
        manual.status = Manual.Status.PENDING
        manual.save()
        self._attach_real_file(manual, bad_path)
        ingest_manual(manual)
        manual.refresh_from_db()
        self.assertEqual(manual.status, Manual.Status.FAILED)

        self._attach_real_file(manual, good_path, name="fixed.pdf")
        ingest_manual(manual)
        manual.refresh_from_db()

        self.assertEqual(manual.status, Manual.Status.COMPLETED)
        self.assertEqual(manual.error_message, "")
        self.assertEqual(ManualPage.objects.filter(manual=manual).count(), 1)
