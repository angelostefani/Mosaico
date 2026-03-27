#!/usr/bin/env sh
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput || true

exec gunicorn mosaico.wsgi:application \
    --bind 0.0.0.0:${PORT:-9001} \
    --workers ${GUNICORN_WORKERS:-2} \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -

