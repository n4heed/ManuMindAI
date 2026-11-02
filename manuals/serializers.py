from django.urls import reverse
from rest_framework import serializers

from .models import Citation, GroqCitation, GroqClaim, Machine, Manual, Question


class MachineSerializer(serializers.ModelSerializer):
    # Annotated by MachineListCreateView's queryset (Count(..., distinct=True))
    # for GET list responses, to avoid N+1 queries. A freshly-created
    # Machine from POST has no such annotation, so these fall back to a
    # direct (always-correct, just not pre-annotated) count instead of
    # crashing on a missing attribute.
    manual_count = serializers.SerializerMethodField()
    question_count = serializers.SerializerMethodField()

    class Meta:
        model = Machine
        fields = ["id", "machine_id", "model", "name", "created_at", "is_archived", "manual_count", "question_count"]
        read_only_fields = ["is_archived", "manual_count", "question_count"]

    def get_manual_count(self, obj):
        value = getattr(obj, "manual_count", None)
        return value if value is not None else obj.manuals.count()

    def get_question_count(self, obj):
        value = getattr(obj, "question_count", None)
        return value if value is not None else obj.questions.count()


class ManualSerializer(serializers.ModelSerializer):
    machine_id_code = serializers.CharField(source="machine.machine_id", read_only=True)
    secure_file_url = serializers.SerializerMethodField()

    class Meta:
        model = Manual
        fields = [
            "id",
            "machine",
            "machine_id_code",
            "document_id",
            "revision",
            "file",
            "secure_file_url",
            "original_filename",
            "status",
            "error_message",
            "page_count",
            "created_at",
            "processed_at",
        ]
        read_only_fields = [
            "status",
            "error_message",
            "page_count",
            "created_at",
            "processed_at",
            "original_filename",
        ]
        # 'file' accepts the upload on POST but is never echoed back with its
        # raw storage URL - that URL is no longer publicly servable anyway
        # (see config/urls.py), but keeping it write-only avoids the API
        # response naming/implying a reachable public path at all. Use
        # secure_file_url (the authenticated ManualFileView route) instead.
        extra_kwargs = {"file": {"write_only": True}}

    def get_secure_file_url(self, obj):
        if not obj.file:
            return None
        request = self.context.get("request")
        url = reverse("api-manual-file", kwargs={"manual_id": obj.id})
        return request.build_absolute_uri(url) if request else url

    def validate(self, attrs):
        uploaded_file = attrs.get("file")
        if uploaded_file is not None:
            from .services import validate_pdf_upload, IngestionError

            try:
                validate_pdf_upload(uploaded_file)
            except IngestionError as exc:
                raise serializers.ValidationError({"file": str(exc)})
        return attrs

    def create(self, validated_data):
        validated_data["original_filename"] = validated_data["file"].name
        return super().create(validated_data)


class CitationSerializer(serializers.ModelSerializer):
    # document_id/revision read from the snapshot captured at citation-
    # creation time (manuals/rag/pipeline.py), not live from `manual` -
    # correct and identical either way while the Manual still exists, and
    # still correct (not blank, not a crash) after it has been deleted.
    document_id = serializers.CharField(source="manual_document_id", read_only=True)
    revision = serializers.CharField(source="manual_revision", read_only=True)
    manual_file_url = serializers.SerializerMethodField()
    manual_deleted = serializers.SerializerMethodField()

    class Meta:
        model = Citation
        fields = [
            "id", "manual", "document_id", "revision", "page_number", "excerpt", "score",
            "manual_file_url", "manual_deleted",
        ]

    def get_manual_deleted(self, obj):
        return obj.manual_id is None

    def get_manual_file_url(self, obj):
        if obj.manual_id is None or not obj.manual.file:
            return None
        request = self.context.get("request")
        url = reverse("api-manual-file", kwargs={"manual_id": obj.manual_id})
        return request.build_absolute_uri(url) if request else url


class GroqCitationSerializer(serializers.ModelSerializer):
    document_id = serializers.CharField(source="manual_document_id", read_only=True)
    revision = serializers.CharField(source="manual_revision", read_only=True)
    manual_file_url = serializers.SerializerMethodField()
    manual_deleted = serializers.SerializerMethodField()

    class Meta:
        model = GroqCitation
        fields = [
            "id",
            "manual",
            "document_id",
            "revision",
            "page_number",
            "excerpt",
            "quote",
            "score",
            "source_id",
            "manual_file_url",
            "manual_deleted",
        ]

    def get_manual_deleted(self, obj):
        return obj.manual_id is None

    def get_manual_file_url(self, obj):
        if obj.manual_id is None or not obj.manual.file:
            return None
        request = self.context.get("request")
        url = reverse("api-manual-file", kwargs={"manual_id": obj.manual_id})
        return request.build_absolute_uri(url) if request else url


class GroqClaimSerializer(serializers.ModelSerializer):
    citations = GroqCitationSerializer(many=True, read_only=True)

    class Meta:
        model = GroqClaim
        fields = ["id", "order", "text", "citations"]


class QuestionSerializer(serializers.ModelSerializer):
    citations = CitationSerializer(many=True, read_only=True)
    machine_id_code = serializers.CharField(source="machine.machine_id", read_only=True)
    asked_by = serializers.CharField(source="user.username", read_only=True)
    # Dual-Answer RAG: 'answers.rag_only' mirrors the top-level answer_text/
    # status/llm_provider/citations fields below exactly - those are kept as
    # their own top-level fields too (unchanged) for backward compatibility
    # with anything already reading this response shape. 'answers.rag_groq'
    # is the new, independent Groq-generated answer.
    answers = serializers.SerializerMethodField()

    class Meta:
        model = Question
        fields = [
            "id",
            "machine",
            "machine_id_code",
            "asked_by",
            "question_text",
            "answer_text",
            "status",
            "llm_provider",
            "created_at",
            "citations",
            "answers",
        ]
        read_only_fields = [
            "answer_text",
            "status",
            "llm_provider",
            "created_at",
            "citations",
            "asked_by",
            "answers",
        ]

    def get_answers(self, obj):
        return {
            "rag_only": {
                "status": obj.status,
                "provider": obj.llm_provider,
                "answer": obj.answer_text,
                "citations": CitationSerializer(obj.citations.all(), many=True, context=self.context).data,
            },
            "rag_groq": {
                "status": obj.groq_status,
                "provider": "groq",
                "model": obj.groq_model or None,
                "answer": obj.groq_answer_text,
                # New structured-output answers are broken into per-claim
                # citations (each claim shows only its own sources). Older
                # answers, from before structured outputs, have no claims at
                # all but still have a flat 'citations' list (claim=None) -
                # both render correctly; the frontend prefers 'claims' when
                # non-empty and falls back to the flat list otherwise.
                "claims": GroqClaimSerializer(
                    obj.groq_claims.all(), many=True, context=self.context
                ).data,
                "citations": GroqCitationSerializer(
                    obj.groq_citations.filter(claim__isnull=True), many=True, context=self.context
                ).data,
                "generation_time_ms": obj.groq_generation_time_ms,
                "error_message": obj.groq_error_message or None,
            },
        }


class AskQuestionSerializer(serializers.Serializer):
    # Archived machines are excluded here - not just hidden from the
    # Engineer ask-form's <select> - so a direct API request can never
    # submit a new question against an archived machine either.
    machine = serializers.PrimaryKeyRelatedField(queryset=Machine.objects.filter(is_archived=False))
    question_text = serializers.CharField(max_length=2000)
