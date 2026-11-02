"""
Django settings for config project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / '.env')


def env(key, default=None):
    return os.environ.get(key, default)


_INSECURE_DEFAULT_SECRET_KEY = 'django-insecure-dev-only-change-me'
SECRET_KEY = env('DJANGO_SECRET_KEY', _INSECURE_DEFAULT_SECRET_KEY)

DEBUG = env('DJANGO_DEBUG', 'True') == 'True'

ALLOWED_HOSTS = [h.strip() for h in env('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost').split(',') if h.strip()]

if not DEBUG:
    # Fail fast rather than silently deploying with the checked-in dev key.
    if SECRET_KEY == _INSECURE_DEFAULT_SECRET_KEY:
        raise RuntimeError(
            'DJANGO_SECRET_KEY must be set to a real secret when DJANGO_DEBUG=False.'
        )
    # These only apply when DEBUG=False so local HTTP development is unaffected.
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = env('DJANGO_SECURE_SSL_REDIRECT', 'True') == 'True'
    # Opt-in, not a default: HSTS is only safe once the deployment's TLS setup
    # is confirmed, and this project does not control that. See README for
    # the organizer/deployer action required to enable it.
    SECURE_HSTS_SECONDS = int(env('DJANGO_HSTS_SECONDS', '0'))


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'manuals',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('POSTGRES_DB', 'manual_assistant'),
        'USER': env('POSTGRES_USER', 'manual_app'),
        'PASSWORD': env('POSTGRES_PASSWORD', ''),
        'HOST': env('POSTGRES_HOST', 'localhost'),
        'PORT': env('POSTGRES_PORT', '5432'),
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'

# --- RAG provider configuration -------------------------------------------
# "local" = deterministic, dependency-free fallback (no external API calls).
# Switch to "openai" / "anthropic" once organizer-issued keys are available;
# see manuals/rag/providers.py for the provider interfaces to implement.
EMBEDDING_PROVIDER = env('EMBEDDING_PROVIDER', 'local')
LLM_PROVIDER = env('LLM_PROVIDER', 'local')
OPENAI_API_KEY = env('OPENAI_API_KEY')
ANTHROPIC_API_KEY = env('ANTHROPIC_API_KEY')

# Minimum similarity score (0-1) for a retrieved chunk to count as evidence.
# See README "Retrieval threshold" section for how this was chosen.
RAG_SIMILARITY_THRESHOLD = float(env('RAG_SIMILARITY_THRESHOLD', '0.12'))
RAG_TOP_K = int(env('RAG_TOP_K', '5'))

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}
