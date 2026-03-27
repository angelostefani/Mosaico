-- Migration: add indexes to uploads table
-- Compatible with SQLite and PostgreSQL
-- Run once on any existing database; safe to re-run (IF NOT EXISTS).

CREATE INDEX IF NOT EXISTS idx_uploads_username
    ON uploads (username);

CREATE INDEX IF NOT EXISTS idx_uploads_username_collection
    ON uploads (username, collection);

CREATE INDEX IF NOT EXISTS idx_uploads_status
    ON uploads (status);
