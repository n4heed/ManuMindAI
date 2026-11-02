from rest_framework.permissions import BasePermission

from .models import Profile


def user_role(user):
    try:
        return user.profile.role
    except Profile.DoesNotExist:
        return None


class IsAdmin(BasePermission):
    message = "Only Administrators may perform this action."

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and user_role(request.user) == Profile.Role.ADMIN)


class IsEngineer(BasePermission):
    message = "Only Engineers may perform this action."

    def has_permission(self, request, view):
        return bool(
            request.user and request.user.is_authenticated and user_role(request.user) == Profile.Role.ENGINEER
        )


class IsSupervisorOrAdmin(BasePermission):
    message = "Only Supervisors or Administrators may perform this action."

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and user_role(request.user) in (Profile.Role.SUPERVISOR, Profile.Role.ADMIN)
        )
