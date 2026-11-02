from django.db import transaction
from django.db.models import Count
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Machine, Manual, Profile, Question
from .permissions import IsAdmin, IsEngineer, user_role
from .rag.pipeline import answer_question
from .serializers import (
    AskQuestionSerializer,
    ManualSerializer,
    MachineSerializer,
    QuestionSerializer,
)
from .services import check_machine_mismatch, generate_recommended_questions, ingest_manual


class ManualMachineCheckView(APIView):
    """
    Admin-only, read-only safeguard used by the upload form BEFORE the real
    upload: extracts only the first page of the given PDF and checks
    whether it seems to belong to a different machine than the one
    selected. Writes nothing to the database either way - purely advisory,
    so the Administrator can confirm or correct the machine before the
    actual POST to ManualListCreateView.
    """

    permission_classes = [IsAdmin]

    def post(self, request):
        machine_id = request.data.get("machine")
        uploaded_file = request.data.get("file")
        if not machine_id or not uploaded_file:
            return Response({"warning": None})
        try:
            machine = Machine.objects.get(pk=machine_id)
        except (Machine.DoesNotExist, ValueError, TypeError):
            return Response({"warning": None})
        warning = check_machine_mismatch(uploaded_file, machine)
        return Response({"warning": warning})


class RecommendedQuestionsView(APIView):
    """
    Deterministic, LLM-free question suggestions for one machine, built only
    from that machine's own successfully-indexed chunks
    (manuals/services.py: generate_recommended_questions). Authorization
    boundary matches MachineListCreateView's GET and QuestionHistoryView -
    any authenticated user, since no per-user machine restriction exists
    anywhere else in this app either.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, machine_id):
        machine = get_object_or_404(Machine, pk=machine_id)
        questions = generate_recommended_questions(machine)
        return Response({"questions": questions})


class MachineListCreateView(generics.ListCreateAPIView):
    """Any authenticated user can list machines (needed to populate the 'select a machine'
    dropdown for Engineers/Supervisors). Only Administrators can register a new machine."""

    # distinct=True on both annotations: without it, joining 'manuals' and
    # 'questions' in the same annotate() would fan out and double-count.
    queryset = Machine.objects.all().order_by("machine_id").annotate(
        manual_count=Count("manuals", distinct=True),
        question_count=Count("questions", distinct=True),
    )
    serializer_class = MachineSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        if self.request.method == "POST":
            return [IsAdmin()]
        return [permissions.IsAuthenticated()]


class MachineDeleteView(APIView):
    """
    Admin-only. Safe, dependency-aware deletion:

    - If the machine has NO Question history, it is permanently deleted.
      Question.machine and Manual.machine are both on_delete=CASCADE, so
      this also removes its manuals, their ManualPage/Chunk rows, and (per
      the same-machine-isolation guarantee retrieval itself relies on) it
      is not possible for any OTHER machine's Question to have ever cited
      one of this machine's manuals - so nothing outside this machine's own
      (history-free) data is touched.
    - If the machine HAS Question history, a hard delete would silently
      destroy that history (CASCADE) - which this app must never do - so
      the machine is archived instead (is_archived=True): manuals, chunks,
      and every historical question/citation are preserved untouched: only
      new question submissions against it are blocked
      (manuals/serializers.py: AskQuestionSerializer) and it is removed
      from the Engineer ask-form's machine selector.

    Either way, uploaded PDF files are removed via Django's storage API
    (never direct filesystem access) only after the database transaction
    that deletes the Manual/Machine rows has actually committed.
    """

    permission_classes = [IsAdmin]

    def delete(self, request, machine_id):
        machine = get_object_or_404(Machine, pk=machine_id)
        has_history = Question.objects.filter(machine=machine).exists()

        if has_history:
            if machine.is_archived:
                return Response(
                    {"detail": "This machine is already archived.", "archived": True}, status=200
                )
            machine.is_archived = True
            machine.save(update_fields=["is_archived"])
            return Response(
                {
                    "archived": True,
                    "detail": (
                        f"{machine.machine_id} has question history, so it was archived instead of "
                        "deleted: its manuals and all historical questions/citations are preserved. "
                        "It no longer accepts new questions."
                    ),
                },
                status=200,
            )

        file_fields = [manual.file for manual in machine.manuals.all() if manual.file]
        with transaction.atomic():
            machine.delete()
            if file_fields:
                transaction.on_commit(lambda: [f.delete(save=False) for f in file_fields])
        return Response(status=204)


class MachineUnarchiveView(APIView):
    """
    Admin-only. Reverses MachineDeleteView's archive path: clears
    is_archived so the machine reappears in the Engineer ask-form selector
    and can accept new questions again. Nothing else changes - manuals,
    chunks, and history were never touched by archiving in the first place.
    """

    permission_classes = [IsAdmin]

    def post(self, request, machine_id):
        machine = get_object_or_404(Machine, pk=machine_id)
        if not machine.is_archived:
            return Response({"detail": "This machine is not archived.", "archived": False}, status=200)
        machine.is_archived = False
        machine.save(update_fields=["is_archived"])
        return Response(
            {
                "archived": False,
                "detail": f"{machine.machine_id} was unarchived and can accept new questions again.",
            },
            status=200,
        )


class ManualDeleteView(APIView):
    """
    Admin-only. Permanently deletes one Manual: its ManualPage/Chunk rows
    cascade-delete (on_delete=CASCADE - removing it from active retrieval
    and deleting its embeddings), while Citation.manual/GroqCitation.manual
    are on_delete=SET_NULL - so historical Question answers and their
    citations survive, using the manual_document_id/manual_revision
    snapshot captured at citation-creation time (manuals/rag/pipeline.py)
    to still show which manual a citation came from. The serializer
    resolves manual_file_url to None once `manual` is null, so a historical
    citation never links to a PDF that no longer exists.

    The uploaded PDF is removed via Django's storage API only after the
    database transaction commits successfully.
    """

    permission_classes = [IsAdmin]

    def delete(self, request, manual_id):
        manual = get_object_or_404(Manual, pk=manual_id)
        file_field = manual.file if manual.file else None
        with transaction.atomic():
            manual.delete()
            if file_field:
                transaction.on_commit(lambda: file_field.delete(save=False))
        return Response(status=204)


class ManualListCreateView(generics.ListCreateAPIView):
    """
    List: Administrators see all manuals (with ingestion status).
    Create (upload): Administrators only - this is the server-side enforcement
    of 'Engineers cannot upload manuals', independent of any UI button hiding.
    """

    serializer_class = ManualSerializer

    def get_permissions(self):
        if self.request.method == "POST":
            return [IsAdmin()]
        return [permissions.IsAuthenticated()]

    def get_queryset(self):
        qs = Manual.objects.select_related("machine").order_by("-created_at")
        machine_id = self.request.query_params.get("machine")
        if machine_id:
            qs = qs.filter(machine_id=machine_id)
        return qs

    def perform_create(self, serializer):
        manual = serializer.save(uploaded_by=self.request.user)
        ingest_manual(manual)  # synchronous: corpus is tiny (3 short PDFs), ingestion is fast
        manual.refresh_from_db()


class AskQuestionView(APIView):
    """Engineer-only: ask a question about a selected machine; retrieval is restricted
    to that machine's manuals before any answer is generated (machine isolation)."""

    permission_classes = [IsEngineer]

    def post(self, request):
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        machine = serializer.validated_data["machine"]
        question_text = serializer.validated_data["question_text"]

        question = answer_question(request.user, machine, question_text)
        return Response(
            QuestionSerializer(question, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class QuestionHistoryView(generics.ListAPIView):
    """
    Engineers: only their own questions.
    Supervisors and Administrators: all questions (read-only history review).
    """

    serializer_class = QuestionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        qs = Question.objects.select_related("machine", "user").prefetch_related(
            "citations__manual", "groq_citations__manual", "groq_claims__citations__manual"
        )
        role = user_role(user)
        if role in (Profile.Role.SUPERVISOR, Profile.Role.ADMIN):
            machine_id = self.request.query_params.get("machine")
            if machine_id:
                qs = qs.filter(machine_id=machine_id)
            return qs
        # Engineer (or no profile yet): restricted to their own questions.
        return qs.filter(user=user)


class ManualFileView(APIView):
    """
    Serves a manual's PDF to any authenticated user. This is the app's
    *only* route for reading manual files - config/urls.py no longer serves
    MEDIA_ROOT publicly at all, in any environment.

    Authorization policy: IsAuthenticated only, matching the existing
    boundary the rest of the app already uses for reading manuals -
    ManualListCreateView's GET and QuestionHistoryView are both
    IsAuthenticated-only too, with no per-user manual-ownership
    restriction anywhere in the data model. There is nothing narrower to
    inherit here, so nothing narrower is invented here either.

    The manual is always resolved through the database by primary key -
    the request can never supply a filesystem path, so path traversal
    through this endpoint is not possible by construction.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, manual_id):
        manual = get_object_or_404(Manual, pk=manual_id)
        if not manual.file:
            raise Http404("No file is available for this manual.")
        try:
            file_handle = manual.file.open("rb")
        except (FileNotFoundError, OSError):
            # The DB record exists but the file is missing from disk (e.g. a
            # dev environment with a stale DB). Report it like any other
            # "not found" - never echo the underlying filesystem path/error.
            raise Http404("The manual file could not be found.")

        safe_name = "".join(ch for ch in manual.original_filename if ch not in '\r\n"').strip() or "manual.pdf"
        response = FileResponse(file_handle, content_type="application/pdf")
        response["Content-Disposition"] = f'inline; filename="{safe_name}"'
        response["X-Content-Type-Options"] = "nosniff"
        return response


class WhoAmIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        role = user_role(request.user)
        return Response({"username": request.user.username, "role": role})
