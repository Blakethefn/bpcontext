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
         );"
    )?;
    Ok(())
}
