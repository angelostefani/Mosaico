import os
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def _default_db_engine() -> str:
    return os.getenv("DB_ENGINE", "sqlite").lower()


def build_db_url() -> str:
    engine = _default_db_engine()
    if engine == "postgres" or engine == "postgresql":
        host = os.getenv("PG_HOST", "localhost")
        port = os.getenv("PG_PORT", "5432")
        db = os.getenv("PG_DB", "")
        user = os.getenv("PG_USER", "")
        password = os.getenv("PG_PASSWORD", "")
        sslmode = os.getenv("PG_SSLMODE", "prefer")
        if not db or not user:
            raise RuntimeError("PG_DB e PG_USER sono richiesti quando DB_ENGINE=postgres.")
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}?sslmode={sslmode}"
    # default: sqlite
    sqlite_path = os.getenv("UPLOAD_DB_PATH", os.path.join(os.getcwd(), "uploads", "uploads.sqlite3"))
    return f"sqlite:///{sqlite_path}"


def create_engine_and_session(echo: bool = False):
    url = build_db_url()
    engine = create_engine(
        url,
        echo=echo,
        connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return engine, SessionLocal


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


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)
