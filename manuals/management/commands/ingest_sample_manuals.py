"""
Registers the two demo machines and ingests the three challenge-pack manual
PDFs directly from disk (bypassing the upload form), for fast local testing.

Usage:
  python manage.py ingest_sample_manuals "<path to 'Round 1 Student Pack'>"

The folder must contain machine_manual_index.csv and a Manuals/ subfolder,
exactly as supplied in the official challenge pack.
"""

import csv
from pathlib import Path

from django.contrib.auth.models import User
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from manuals.models import Machine, Manual
from manuals.services import ingest_manual


class Command(BaseCommand):
    help = "Register machines and ingest the sample manuals from the Round 1 Student Pack folder."

    def add_arguments(self, parser):
        parser.add_argument("student_pack_dir", type=str)

    def handle(self, *args, **options):
        pack_dir = Path(options["student_pack_dir"])
        csv_path = pack_dir / "machine_manual_index.csv"
        manuals_dir = pack_dir / "Manuals"
        if not csv_path.exists():
            raise CommandError(f"Could not find {csv_path}")
        if not manuals_dir.exists():
            raise CommandError(f"Could not find {manuals_dir}")

        uploader, _ = User.objects.get_or_create(username="admin1")

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                machine, _ = Machine.objects.get_or_create(
                    machine_id=row["machine_id"],
                    defaults={"model": row["model"], "name": row["machine_name"]},
                )
                pdf_path = manuals_dir / row["pdf_filename"]
                if not pdf_path.exists():
                    self.stderr.write(self.style.WARNING(f"Missing PDF: {pdf_path}"))
                    continue

                manual, created = Manual.objects.get_or_create(
                    machine=machine,
                    document_id=row["document_id"],
                    revision=row["revision"],
                    defaults={"uploaded_by": uploader, "original_filename": row["pdf_filename"]},
                )
                if created:
                    with open(pdf_path, "rb") as f:
                        manual.file.save(row["pdf_filename"], File(f), save=True)
                ingest_manual(manual)
                manual.refresh_from_db()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"{row['pdf_filename']} -> {manual.status} ({manual.page_count} pages)"
                    )
                )
