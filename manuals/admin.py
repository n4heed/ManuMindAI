from django.contrib import admin

from .models import Chunk, Citation, Machine, Manual, ManualPage, Profile, Question


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    list_filter = ('role',)


@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ('machine_id', 'model', 'name', 'created_at')
    search_fields = ('machine_id', 'model', 'name')


@admin.register(Manual)
class ManualAdmin(admin.ModelAdmin):
    list_display = ('document_id', 'revision', 'machine', 'status', 'page_count', 'created_at')
    list_filter = ('status', 'machine')
    readonly_fields = ('status', 'error_message', 'page_count', 'processed_at')


@admin.register(ManualPage)
class ManualPageAdmin(admin.ModelAdmin):
    list_display = ('manual', 'page_number')


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ('manual', 'page', 'chunk_index', 'embedding_provider')


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('user', 'machine', 'status', 'created_at')
    list_filter = ('status', 'machine')
    readonly_fields = [f.name for f in Question._meta.fields]


@admin.register(Citation)
class CitationAdmin(admin.ModelAdmin):
    list_display = ('question', 'manual', 'page_number', 'score')
