from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static

from manuals.forms import BootstrapAuthenticationForm

urlpatterns = [
    path('admin/', admin.site.urls),
    path(
        'login/',
        auth_views.LoginView.as_view(template_name='manuals/login.html', authentication_form=BootstrapAuthenticationForm),
        name='login',
    ),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('', include('manuals.urls')),
]

if settings.DEBUG and settings.STATICFILES_DIRS:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])

# No public URL ever serves MEDIA_ROOT, in any environment - manual PDFs are
# only readable through the authenticated ManualFileView (manuals/views_api.py),
# which opens the file from disk via the Manual model, never through a URL.
# See README "Secure PDF delivery" for what this means for a real deployment's
# web server / reverse-proxy configuration.
