from django.conf import settings
from django.db import models


class Profile(models.Model):
    class Role(models.TextChoices):
        ADMIN = 'admin', 'Administrator'
        ENGINEER = 'engineer', 'Engineer'
        SUPERVISOR = 'supervisor', 'Supervisor'

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=Role.choices)

    def __str__(self):
        return f'{self.user.username} ({self.role})'


class Machine(models.Model):
    machine_id = models.CharField(max_length=50, unique=True)
    model = models.CharField(max_length=100)
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    # Set instead of a hard delete whenever the machine has question history
    # that a hard delete would otherwise destroy (Question.machine is
    # on_delete=CASCADE) - see manuals/views_api.py: MachineDeleteView. An
    # archived machine is excluded from the Engineer ask-form's machine
    # selector but keeps all its manuals, chunks, and history intact and
    # still visible/filterable in Question History.
    is_archived = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.machine_id} - {self.name}'


class Manual(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='manuals')
    document_id = models.CharField(max_length=50)
    revision = models.CharField(max_length=20)
    file = models.FileField(upload_to='manuals/')
    original_filename = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error_message = models.TextField(blank=True, default='')
    page_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('machine', 'document_id', 'revision')

    def __str__(self):
        return f'{self.document_id} rev{self.revision} ({self.machine.machine_id})'


class ManualPage(models.Model):
    manual = models.ForeignKey(Manual, on_delete=models.CASCADE, related_name='pages')
    page_number = models.PositiveIntegerField()
    text = models.TextField(blank=True, default='')

    class Meta:
        unique_together = ('manual', 'page_number')
        ordering = ['page_number']

    def __str__(self):
        return f'{self.manual} p.{self.page_number}'


class Chunk(models.Model):
    manual = models.ForeignKey(Manual, on_delete=models.CASCADE, related_name='chunks')
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='chunks')
    page = models.ForeignKey(ManualPage, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.PositiveIntegerField()
    text = models.TextField()
    embedding = models.JSONField()
    embedding_provider = models.CharField(max_length=50)

    class Meta:
        ordering = ['manual_id', 'page__page_number', 'chunk_index']

    def __str__(self):
        return f'{self.manual.document_id} p.{self.page.page_number} #{self.chunk_index}'


class Question(models.Model):
    class Status(models.TextChoices):
        ANSWERED = 'answered', 'Answered'
        INSUFFICIENT_EVIDENCE = 'insufficient_evidence', 'Insufficient evidence'

    class GroqStatus(models.TextChoices):
        COMPLETED = 'completed', 'Completed'
        INSUFFICIENT_EVIDENCE = 'insufficient_evidence', 'Insufficient evidence'
        UNAVAILABLE = 'unavailable', 'Unavailable'
        FAILED = 'failed', 'Failed'

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='questions')
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    answer_text = models.TextField(blank=True, default='')
    status = models.CharField(max_length=30, choices=Status.choices)
    llm_provider = models.CharField(max_length=50, default='local')
    created_at = models.DateTimeField(auto_now_add=True)

    # Dual-Answer RAG: the RAG-only answer above (status/answer_text/
    # llm_provider/citations) is unchanged from before this feature. These
    # fields hold the independent Groq-generated answer, built from the
    # exact same retrieved evidence. default=UNAVAILABLE so every question
    # asked before this feature existed - which has no Groq data at all -
    # reads back as "unavailable" rather than fabricating a status.
    groq_status = models.CharField(max_length=30, choices=GroqStatus.choices, default=GroqStatus.UNAVAILABLE)
    groq_answer_text = models.TextField(blank=True, default='')
    groq_model = models.CharField(max_length=100, blank=True, default='')
    groq_error_message = models.TextField(blank=True, default='')
    groq_generation_time_ms = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} -> {self.machine.machine_id}: {self.question_text[:40]}'


class Citation(models.Model):
    """
    RAG-only answer citations. `manual` is nullable with on_delete=SET_NULL
    (not CASCADE) so permanently deleting a Manual (manuals/views_api.py:
    ManualDeleteView) never destroys a historical Question's citation list -
    `manual_document_id`/`manual_revision` are a snapshot captured at
    citation-creation time (manuals/rag/pipeline.py), so the citation still
    shows which manual it came from even after that Manual row is gone.
    `page_number`, `excerpt`, and `score` were always stored directly here,
    not derived from Manual/Chunk at read time, so they survive unchanged.
    """

    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='citations')
    manual = models.ForeignKey(Manual, on_delete=models.SET_NULL, null=True, blank=True, related_name='citations')
    manual_document_id = models.CharField(max_length=50, blank=True, default='')
    manual_revision = models.CharField(max_length=20, blank=True, default='')
    page_number = models.PositiveIntegerField()
    excerpt = models.TextField()
    score = models.FloatField()

    def __str__(self):
        return f'{self.manual_document_id} p.{self.page_number}'


class GroqClaim(models.Model):
    """
    One factual statement from a Groq structured-output answer
    (manuals/rag/groq_provider.py's {"claims": [...]} schema), kept separate
    per claim so a multi-part question's answer can show each part with its
    own sources rather than one citation list for an entire answer.

    A GroqClaim row only ever exists here if at least one of its sources
    survived backend validation (manuals/rag/pipeline.py:
    _attach_groq_answer) - an unsupported claim is dropped entirely, never
    stored and shown without evidence.
    """

    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='groq_claims')
    order = models.PositiveIntegerField()
    text = models.TextField()

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f'{self.question_id} claim #{self.order}: {self.text[:40]}'


class GroqCitation(models.Model):
    """
    Citations for the Groq-generated answer. Kept as a separate model from
    Citation (rather than a shared/generic table) so the RAG-only answer's
    existing data and behavior are completely untouched by this feature.

    Only ever created from source identifiers ([S1], [S2], ...) that the
    backend itself validated against the chunks actually retrieved and sent
    to Groq - never from an identifier Groq invented - AND, once structured
    outputs were added, only once the `quote` was also confirmed to be an
    actual (whitespace-normalized) substring of that source's text. `score`
    is the original retrieval similarity score (the same shared evidence
    used for the RAG-only answer), never a value the LLM returned.

    `claim` is null for rows created before structured outputs (a flat,
    single-answer-text response with no per-claim breakdown) and set for
    every row created after - both are rendered correctly for backward
    compatibility (see QuestionSerializer.get_answers / renderDualAnswers()
    in base.html).

    `manual` is nullable with on_delete=SET_NULL for the same reason as
    Citation.manual above - deleting a Manual must never destroy a
    historical Groq answer's citations; manual_document_id/manual_revision
    are the same kind of creation-time snapshot.
    """

    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='groq_citations')
    claim = models.ForeignKey(
        GroqClaim, on_delete=models.CASCADE, related_name='citations', null=True, blank=True
    )
    manual = models.ForeignKey(Manual, on_delete=models.SET_NULL, null=True, blank=True, related_name='groq_citations')
    manual_document_id = models.CharField(max_length=50, blank=True, default='')
    manual_revision = models.CharField(max_length=20, blank=True, default='')
    page_number = models.PositiveIntegerField()
    excerpt = models.TextField()
    quote = models.TextField(blank=True, default='')
    score = models.FloatField()
    source_id = models.CharField(max_length=10)

    def __str__(self):
        return f'{self.source_id} -> {self.manual_document_id} p.{self.page_number}'
