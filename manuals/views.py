from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .permissions import user_role
from .models import Profile


@login_required
def dashboard(request):
    role = user_role(request.user)
    if role == Profile.Role.ADMIN:
        return render(request, "manuals/admin_dashboard.html")
    if role == Profile.Role.ENGINEER:
        return render(request, "manuals/engineer_dashboard.html")
    if role == Profile.Role.SUPERVISOR:
        return render(request, "manuals/supervisor_dashboard.html")
    return render(request, "manuals/no_role.html")


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")
