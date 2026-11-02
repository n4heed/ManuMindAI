from django.urls import path

from . import views
from .views_api import (
    AskQuestionView,
    MachineDeleteView,
    MachineUnarchiveView,
    ManualDeleteView,
    ManualFileView,
    ManualListCreateView,
    ManualMachineCheckView,
    MachineListCreateView,
    QuestionHistoryView,
    RecommendedQuestionsView,
    WhoAmIView,
)

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    # REST API
    path("api/whoami/", WhoAmIView.as_view(), name="api-whoami"),
    path("api/machines/", MachineListCreateView.as_view(), name="api-machines"),
    path("api/machines/<int:machine_id>/", MachineDeleteView.as_view(), name="api-machine-delete"),
    path(
        "api/machines/<int:machine_id>/unarchive/",
        MachineUnarchiveView.as_view(),
        name="api-machine-unarchive",
    ),
    path(
        "api/machines/<int:machine_id>/recommended-questions/",
        RecommendedQuestionsView.as_view(),
        name="api-recommended-questions",
    ),
    path("api/manuals/", ManualListCreateView.as_view(), name="api-manuals"),
    path("api/manuals/<int:manual_id>/", ManualDeleteView.as_view(), name="api-manual-delete"),
    path("api/manuals/<int:manual_id>/file/", ManualFileView.as_view(), name="api-manual-file"),
    path("api/manuals/check-machine/", ManualMachineCheckView.as_view(), name="api-manual-check-machine"),
    path("api/questions/", QuestionHistoryView.as_view(), name="api-question-history"),
    path("api/questions/ask/", AskQuestionView.as_view(), name="api-ask-question"),
]
