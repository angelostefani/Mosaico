#!/usr/bin/env sh
set -e

# Apply migrations
python manage.py migrate --noinput

# Collect static files (ignore errors if not configured)
python manage.py collectstatic --noinput || true

# Run Django dev server (for production consider gunicorn/uwsgi)
python manage.py runserver 0.0.0.0:${PORT:-9001}

