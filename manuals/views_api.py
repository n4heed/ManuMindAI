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
from .services import ingest_manual


class MachineListCreateView(generics.ListCreateAPIView):
    """Any authenticated user can list machines (needed to populate the 'select a machine'
    dropdown for Engineers/Supervisors). Only Administrators can register a new machine."""

    queryset = Machine.objects.all().order_by("machine_id")
    serializer_class = MachineSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        if self.request.method == "POST":
            return [IsAdmin()]
        return [permissions.IsAuthenticated()]


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
