from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from manuals.models import Chunk, Machine, Manual, ManualPage, Profile
from manuals.rag.providers import LocalEmbeddingProvider

MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF"
)

_embedder = LocalEmbeddingProvider()


def make_user(username, role):
    user = User.objects.create_user(username=username, password="testpass123")
    Profile.objects.create(user=user, role=role)
    return user


def make_machine(machine_id="M-TEST", model="TEST-1", name="Test machine"):
    return Machine.objects.create(machine_id=machine_id, model=model, name=name)


def make_manual(machine, document_id="DOC-1", revision="1.0", uploaded_by=None):
    return Manual.objects.create(
        machine=machine,
        document_id=document_id,
        revision=revision,
        file=SimpleUploadedFile(f"{document_id}.pdf", MINIMAL_PDF_BYTES, content_type="application/pdf"),
        original_filename=f"{document_id}.pdf",
        uploaded_by=uploaded_by,
        status=Manual.Status.COMPLETED,
        page_count=1,
    )


def make_chunk(manual, text, page_number=1, chunk_index=0):
    page, _ = ManualPage.objects.get_or_create(manual=manual, page_number=page_number, defaults={"text": text})
    [vector] = _embedder.embed([text])
    return Chunk.objects.create(
        manual=manual,
        machine=manual.machine,
        page=page,
        chunk_index=chunk_index,
        text=text,
        embedding=vector,
        embedding_provider=_embedder.name,
    )
