import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, create_engine, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import declarative_base

# Modello ORM condiviso per creare lo schema su Postgres
Base = declarative_base()


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True)
    upload_id = Column(String, unique=True, nullable=False)
    username = Column(String)
    collection = Column(String)
    original_filename = Column(String)
    stored_filename = Column(String)
    file_path = Column(String)
    size_bytes = Column(Integer)
    num_chunks = Column(Integer)
    num_points = Column(Integer)
    status = Column(String, nullable=False)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, unique=True, nullable=False)
    user_id = Column(String, nullable=False)
    title = Column(String)
    collection = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_conversations_user_updated", "user_id", "updated_at"),
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_conv_messages_conv_created", "conversation_id", "created_at"),
    )


def parse_timestamp(val: Any) -> Optional[datetime]:
    """Parsa stringhe ISO e restituisce datetime timezone-aware UTC."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Gestione suffisso Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"Timestamp non parsabile: {val}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def build_pg_url(args: argparse.Namespace) -> str:
    port = args.pg_port or "5433"
    return (
        f"postgresql+psycopg://{args.pg_user}:{args.pg_password}"
        f"@{args.pg_host}:{port}/{args.pg_db}"
        f"{'' if not args.pg_sslmode else f'?sslmode={args.pg_sslmode}'}"
    )


def coalesce_env(value: Optional[str], env_key: str, default: Optional[str] = None) -> Optional[str]:
    return value if value is not None else os.getenv(env_key, default)


def read_sqlite_table(
    conn,
    table: str,
    columns: Iterable[str],
    order_by: str,
    batch_size: int,
) -> Iterable[List[Dict[str, Any]]]:
    stmt = text(f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_by}")
    result = conn.execute(stmt).mappings()
    while True:
        batch = result.fetchmany(batch_size)
        if not batch:
            break
        yield [dict(row) for row in batch]


def upsert_uploads(pg_conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = pg_insert(Upload).values(rows)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in Upload.__table__.columns
        if c.name not in ("id", "upload_id")
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[Upload.upload_id],
        set_=update_cols,
    )
    pg_conn.execute(stmt)


def upsert_conversations(pg_conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = pg_insert(Conversation).values(rows)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in Conversation.__table__.columns
        if c.name not in ("id", "conversation_id")
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[Conversation.conversation_id],
        set_=update_cols,
    )
    pg_conn.execute(stmt)


def insert_messages(pg_conn, rows: List[Dict[str, Any]], existing_keys: set) -> int:
    if not rows:
        return 0
    new_rows = []
    for r in rows:
        key = (r["conversation_id"], r["role"], r["content"], r["created_at"])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append(r)
    if not new_rows:
        return 0
    pg_conn.execute(ConversationMessage.__table__.insert(), new_rows)
    return len(new_rows)


def load_existing_message_keys(pg_conn) -> set:
    keys = set()
    result = pg_conn.execute(
        select(
            ConversationMessage.conversation_id,
            ConversationMessage.role,
            ConversationMessage.content,
            ConversationMessage.created_at,
        )
    )
    for row in result:
        keys.add((row.conversation_id, row.role, row.content, row.created_at))
    return keys


def migrate(sqlite_path: str, pg_url: str, batch_size: int, dry_run: bool, echo: bool) -> None:
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Database SQLite non trovato: {sqlite_path}")

    sqlite_engine = create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
    )
    pg_engine = create_engine(pg_url, echo=echo, future=True)

    inspector = inspect(sqlite_engine)
    has_uploads = inspector.has_table("uploads")
    has_conversations = inspector.has_table("conversations")
    has_messages = inspector.has_table("conversation_messages")

    print(f"SQLite: uploads={has_uploads}, conversations={has_conversations}, conversation_messages={has_messages}")
    if not any((has_uploads, has_conversations, has_messages)):
        print("Nessuna tabella sorgente trovata, nulla da migrare.")
        return

    if dry_run:
        print("Dry-run attivo: nessuna scrittura su Postgres.")

    # Prepara schema target
    Base.metadata.create_all(bind=pg_engine)

    with sqlite_engine.connect() as s_conn, pg_engine.begin() as p_conn:
        # Migrazione uploads
        if has_uploads:
            total_uploaded = 0
            for batch in read_sqlite_table(
                s_conn,
                "uploads",
                [
                    "upload_id",
                    "username",
                    "collection",
                    "original_filename",
                    "stored_filename",
                    "file_path",
                    "size_bytes",
                    "num_chunks",
                    "num_points",
                    "status",
                    "error_message",
                    "created_at",
                    "updated_at",
                ],
                order_by="id ASC",
                batch_size=batch_size,
            ):
                for row in batch:
                    row["created_at"] = parse_timestamp(row.get("created_at"))
                    row["updated_at"] = parse_timestamp(row.get("updated_at"))
                total_uploaded += len(batch)
                if dry_run:
                    continue
                upsert_uploads(p_conn, batch)
            print(f"Uploads migrati: {total_uploaded}")
        else:
            print("Tabella uploads assente su SQLite, salto.")

        # Migrazione conversations
        if has_conversations:
            total_convs = 0
            for batch in read_sqlite_table(
                s_conn,
                "conversations",
                [
                    "conversation_id",
                    "user_id",
                    "title",
                    "collection",
                    "created_at",
                    "updated_at",
                ],
                order_by="id ASC",
                batch_size=batch_size,
            ):
                for row in batch:
                    row["created_at"] = parse_timestamp(row.get("created_at"))
                    row["updated_at"] = parse_timestamp(row.get("updated_at"))
                total_convs += len(batch)
                if dry_run:
                    continue
                upsert_conversations(p_conn, batch)
            print(f"Conversazioni migrate: {total_convs}")
        else:
            print("Tabella conversations assente su SQLite, salto.")

        # Migrazione conversation_messages
        if has_messages:
            total_msgs = 0
            inserted_msgs = 0
            existing_keys = set() if dry_run else load_existing_message_keys(p_conn)
            for batch in read_sqlite_table(
                s_conn,
                "conversation_messages",
                [
                    "conversation_id",
                    "role",
                    "content",
                    "created_at",
                ],
                order_by="id ASC",
                batch_size=batch_size,
            ):
                for row in batch:
                    row["created_at"] = parse_timestamp(row.get("created_at"))
                total_msgs += len(batch)
                if dry_run:
                    continue
                inserted = insert_messages(p_conn, batch, existing_keys)
                inserted_msgs += inserted
            if dry_run:
                print(f"Messaggi trovati (dry-run): {total_msgs}")
            else:
                print(f"Messaggi totali: {total_msgs}, inseriti (deduplicati): {inserted_msgs}")
        else:
            print("Tabella conversation_messages assente su SQLite, salto.")

        if dry_run:
            print("Dry-run completato (nessuna scrittura).")
            return

        # Validazione conteggi finali
        counts_pg = {}
        for name, model in (
            ("uploads", Upload),
            ("conversations", Conversation),
            ("conversation_messages", ConversationMessage),
        ):
            result = p_conn.execute(select(func.count()).select_from(model.__table__)).scalar_one()
            counts_pg[name] = result
        print(f"Conteggi Postgres: {counts_pg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrazione da SQLite a Postgres per le tabelle uploads/conversations/conversation_messages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sqlite-path", default=None, help="Percorso file SQLite (uploads.sqlite3).")
    parser.add_argument("--pg-host", default=None, help="Host Postgres.")
    parser.add_argument("--pg-port", default=None, help="Porta Postgres.")
    parser.add_argument("--pg-db", default=None, help="Nome database Postgres.")
    parser.add_argument("--pg-user", default=None, help="Utente Postgres.")
    parser.add_argument("--pg-password", default=None, help="Password Postgres.")
    parser.add_argument("--pg-sslmode", default=None, help="sslmode (prefer, require, disable, ecc.).")
    parser.add_argument("--batch-size", type=int, default=None, help="Dimensione batch di insert.")
    parser.add_argument("--dry-run", action="store_true", help="Legge da SQLite senza scrivere su Postgres.")
    parser.add_argument("--verbose", action="store_true", help="Abilita echo SQLAlchemy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sqlite_path = coalesce_env(args.sqlite_path, "SQLITE_PATH", "./uploads/uploads.sqlite3")
    pg_host = coalesce_env(args.pg_host, "PG_HOST", "localhost")
    pg_port = coalesce_env(args.pg_port, "PG_PORT", "5433")
    pg_db = coalesce_env(args.pg_db, "PG_DB")
    pg_user = coalesce_env(args.pg_user, "PG_USER")
    pg_password = coalesce_env(args.pg_password, "PG_PASSWORD")
    pg_sslmode = coalesce_env(args.pg_sslmode, "PG_SSLMODE", "prefer")
    batch_size = int(coalesce_env(str(args.batch_size) if args.batch_size else None, "BATCH_SIZE", "500"))

    missing = [name for name, val in [
        ("PG_DB", pg_db),
        ("PG_USER", pg_user),
        ("PG_PASSWORD", pg_password),
    ] if not val]
    if missing:
        raise SystemExit(f"Parametri Postgres mancanti: {', '.join(missing)}")

    # Aggiorna args con valori definitivi per build_pg_url
    args.pg_host = pg_host
    args.pg_port = pg_port
    args.pg_db = pg_db
    args.pg_user = pg_user
    args.pg_password = pg_password
    args.pg_sslmode = pg_sslmode

    pg_url = build_pg_url(args)

    print(f"Avvio migrazione SQLite -> Postgres ({pg_host}:{pg_port}/{pg_db}), batch={batch_size}, dry_run={args.dry_run}")
    migrate(
        sqlite_path=sqlite_path,
        pg_url=pg_url,
        batch_size=batch_size,
        dry_run=args.dry_run,
        echo=args.verbose,
    )
    print("Migrazione completata.")


if __name__ == "__main__":
    main()
