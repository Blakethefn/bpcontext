use anyhow::Result;
use rusqlite::Connection;

/// Initialize the content store schema (FTS5 tables + metadata)
pub fn init_content_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "-- Source metadata table
         CREATE TABLE IF NOT EXISTS sources (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             label TEXT NOT NULL,
             indexed_at TEXT NOT NULL,
             chunk_count INTEGER DEFAULT 0,
             code_chunk_count INTEGER DEFAULT 0
         );

         -- Primary FTS5 table (BM25 with Porter stemming)
         CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
             title,
             content,
             content_type,
             source_id UNINDEXED,
             tokenize = 'porter unicode61'
         );

         -- Trigram FTS5 table (fuzzy/partial matching)
         CREATE VIRTUAL TABLE IF NOT EXISTS chunks_trigram USING fts5(
             title,
             content,
             content_type,
             source_id UNINDEXED,
             tokenize = 'trigram'
         );

         -- Vector embeddings for semantic search (Phase 1)
         CREATE TABLE IF NOT EXISTS chunk_embeddings (
             chunk_rowid INTEGER PRIMARY KEY,
             embedding BLOB NOT NULL,
             dim INTEGER NOT NULL DEFAULT 384
         );"
    )?;
    Ok(())
}

/// Initialize the session events schema
pub fn init_session_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS events (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             type TEXT NOT NULL,
             category TEXT,
             data TEXT NOT NULL,
             priority INTEGER DEFAULT 2,
             data_hash TEXT,
             created_at TEXT NOT NULL,
             UNIQUE(data_hash)
         );

         -- Context ledger for tracking what was returned to the agent (Phase 3)
         CREATE TABLE IF NOT EXISTS context_ledger (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             chunk_rowid INTEGER NOT NULL,
             source_label TEXT NOT NULL,
             token_estimate INTEGER NOT NULL,
             returned_at TEXT NOT NULL DEFAULT (datetime('now')),
             access_count INTEGER NOT NULL DEFAULT 1
         );"
    )?;
    Ok(())
}
