"""
ASGI config for mosaico project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from django.conf import settings
from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mosaico.settings')

django_application = get_asgi_application()

if settings.DEBUG:
    # Serve static assets when running the ASGI app directly in development.
    application = ASGIStaticFilesHandler(django_application)
else:
    application = django_application
