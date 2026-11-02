from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from manuals.models import Profile

DEMO_USERS = [
    ("admin1", Profile.Role.ADMIN),
    ("engineer1", Profile.Role.ENGINEER),
    ("supervisor1", Profile.Role.SUPERVISOR),
]
DEMO_PASSWORD = "DemoPass123!"


class Command(BaseCommand):
    help = (
        "Create local demo accounts (admin1/engineer1/supervisor1, password "
        f"'{DEMO_PASSWORD}') for development and the live demo. "
        "These are synthetic test credentials, not real secrets - do not reuse "
        "this password for anything else."
    )

    def handle(self, *args, **options):
        for username, role in DEMO_USERS:
            user, created = User.objects.get_or_create(username=username)
            user.set_password(DEMO_PASSWORD)
            user.save()
            Profile.objects.update_or_create(user=user, defaults={"role": role})
            status = "created" if created else "updated"
            self.stdout.write(self.style.SUCCESS(f"{status}: {username} -> {role}"))
        self.stdout.write(self.style.WARNING(f"Demo password for all demo accounts: {DEMO_PASSWORD}"))
