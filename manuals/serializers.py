from django.urls import reverse
from rest_framework import serializers

from .models import Citation, Machine, Manual, Question


class MachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Machine
        fields = ["id", "machine_id", "model", "name", "created_at"]


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
    document_id = serializers.CharField(source="manual.document_id", read_only=True)
    revision = serializers.CharField(source="manual.revision", read_only=True)
    manual_file_url = serializers.SerializerMethodField()

    class Meta:
        model = Citation
        fields = ["id", "manual", "document_id", "revision", "page_number", "excerpt", "score", "manual_file_url"]

    def get_manual_file_url(self, obj):
        if not obj.manual.file:
            return None
        request = self.context.get("request")
        url = reverse("api-manual-file", kwargs={"manual_id": obj.manual_id})
        return request.build_absolute_uri(url) if request else url


class QuestionSerializer(serializers.ModelSerializer):
    citations = CitationSerializer(many=True, read_only=True)
    machine_id_code = serializers.CharField(source="machine.machine_id", read_only=True)
    asked_by = serializers.CharField(source="user.username", read_only=True)

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
        ]
        read_only_fields = ["answer_text", "status", "llm_provider", "created_at", "citations", "asked_by"]


class AskQuestionSerializer(serializers.Serializer):
    machine = serializers.PrimaryKeyRelatedField(queryset=Machine.objects.all())
    question_text = serializers.CharField(max_length=2000)
