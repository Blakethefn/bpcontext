//! Knowledge database schema and connection management.
//!
//! The knowledge database is a single, global SQLite file at
//! `~/.local/share/bpcontext/knowledge.db`. It is separate from the per-project
//! session and content databases.

// Foundation module — consumed by later implementation prompts (P02-P06).
#![allow(dead_code)]

use anyhow::Result;
use rusqlite::Connection;
use std::path::PathBuf;

use crate::db::{data_dir, open_db};

/// Returns the path to the knowledge database.
/// `~/.local/share/bpcontext/knowledge.db`
pub fn knowledge_db_path() -> Result<PathBuf> {
    let dir = data_dir()?;
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("knowledge.db"))
}

/// Opens the knowledge database, creating it and running schema DDL if needed.
/// Uses the same connection settings as `db::open_db` (WAL, FK, busy timeout).
pub fn open_knowledge_db() -> Result<Connection> {
    let path = knowledge_db_path()?;
    let conn = open_db(&path)?;
    ensure_schema(&conn)?;
    Ok(conn)
}

/// Runs schema DDL. Called by `open_knowledge_db` on every open.
/// All statements use `IF NOT EXISTS` so repeated calls are idempotent.
///
/// Exposed as `pub(crate)` so tests can apply the schema to in-memory connections.
pub(crate) fn apply_schema(conn: &Connection) -> Result<()> {
    ensure_schema(conn)
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS knowledge_sources (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label       TEXT NOT NULL UNIQUE,
            path        TEXT NOT NULL,
            glob        TEXT,
            enrichments TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            last_sync   TEXT,
            file_count  INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS knowledge_files (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id    INTEGER NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
            rel_path     TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes   INTEGER NOT NULL,
            indexed_at   TEXT NOT NULL DEFAULT (datetime('now')),
            chunk_count  INTEGER NOT NULL DEFAULT 0,
            UNIQUE(source_id, rel_path)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING fts5(
            title,
            content,
            tokenize = 'porter unicode61'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_trigram USING fts5(
            title,
            content,
            tokenize = 'trigram'
        );

        CREATE TABLE IF NOT EXISTS knowledge_chunk_meta (
            chunk_rowid INTEGER PRIMARY KEY,
            file_id     INTEGER NOT NULL REFERENCES knowledge_files(id) ON DELETE CASCADE,
            source_id   INTEGER NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
            line_start  INTEGER,
            line_end    INTEGER,
            metadata    TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS knowledge_embeddings (
            chunk_rowid INTEGER PRIMARY KEY,
            embedding   BLOB NOT NULL,
            dim         INTEGER NOT NULL DEFAULT 384
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_meta_source ON knowledge_chunk_meta(source_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_meta_file ON knowledge_chunk_meta(file_id);
        CREATE INDEX IF NOT EXISTS idx_files_source ON knowledge_files(source_id);
        ",
    )?;
    Ok(())
}
