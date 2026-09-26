from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from dotenv import load_dotenv

load_dotenv()  # load agent/.env so DATABASE_URL is available without shell exports

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from app.db.base import Base
from app.db import models  # noqa: F401 — registers all ORM classes on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url") or "")
    if not url:
        raise RuntimeError("DATABASE_URL env var is required for migrations")
    # asyncpg driver required for async engine
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations_sync(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _db_url()

    engine = async_engine_from_config(cfg, prefix="sqlalchemy.", poolclass=NullPool)
    async with engine.connect() as conn:
        await conn.run_sync(_run_migrations_sync)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
