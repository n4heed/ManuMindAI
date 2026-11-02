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

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='questions')
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    answer_text = models.TextField(blank=True, default='')
    status = models.CharField(max_length=30, choices=Status.choices)
    llm_provider = models.CharField(max_length=50, default='local')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} -> {self.machine.machine_id}: {self.question_text[:40]}'


class Citation(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='citations')
    manual = models.ForeignKey(Manual, on_delete=models.CASCADE, related_name='citations')
    page_number = models.PositiveIntegerField()
    excerpt = models.TextField()
    score = models.FloatField()

    def __str__(self):
        return f'{self.manual.document_id} p.{self.page_number}'
