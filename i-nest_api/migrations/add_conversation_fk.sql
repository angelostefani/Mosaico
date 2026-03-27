-- Migration: add foreign key constraint from conversation_messages to conversations
-- with CASCADE DELETE (deleting a conversation removes all its messages).
--
-- ============================================================
-- PostgreSQL
-- ============================================================
-- Run this on your PostgreSQL server:

ALTER TABLE conversation_messages
    ADD CONSTRAINT fk_conv_messages_conversation
    FOREIGN KEY (conversation_id)
    REFERENCES conversations (conversation_id)
    ON DELETE CASCADE;

-- ============================================================
-- SQLite
-- ============================================================
-- SQLite does NOT support ADD CONSTRAINT on existing tables.
-- The ForeignKey is enforced only on tables created after the
-- schema change (handled automatically by SQLAlchemy on fresh DBs).
--
-- For an existing SQLite database, the safest approach is to
-- recreate the table. Run the following block in a SQLite shell:
--
-- PRAGMA foreign_keys = OFF;
-- BEGIN TRANSACTION;
--
-- CREATE TABLE conversation_messages_new (
--     id INTEGER PRIMARY KEY,
--     conversation_id VARCHAR NOT NULL
--         REFERENCES conversations(conversation_id) ON DELETE CASCADE,
--     role VARCHAR NOT NULL,
--     content TEXT NOT NULL,
--     created_at DATETIME NOT NULL
-- );
--
-- INSERT INTO conversation_messages_new
--     SELECT id, conversation_id, role, content, created_at
--     FROM conversation_messages;
--
-- DROP TABLE conversation_messages;
-- ALTER TABLE conversation_messages_new RENAME TO conversation_messages;
--
-- CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_created
--     ON conversation_messages (conversation_id, created_at);
--
-- COMMIT;
-- PRAGMA foreign_keys = ON;
