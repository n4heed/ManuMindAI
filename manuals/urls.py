from django.urls import path

from . import views
from .views_api import (
    AskQuestionView,
    ManualListCreateView,
    MachineListCreateView,
    QuestionHistoryView,
    WhoAmIView,
)

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    # REST API
    path("api/whoami/", WhoAmIView.as_view(), name="api-whoami"),
    path("api/machines/", MachineListCreateView.as_view(), name="api-machines"),
    path("api/manuals/", ManualListCreateView.as_view(), name="api-manuals"),
    path("api/questions/", QuestionHistoryView.as_view(), name="api-question-history"),
    path("api/questions/ask/", AskQuestionView.as_view(), name="api-ask-question"),
]
