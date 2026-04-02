pub mod chunker;
pub mod schema;
pub mod search;

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;

use crate::db;

/// The main content store backed by SQLite FTS5
pub struct ContentStore {
    conn: Connection,
}

impl ContentStore {
    /// Open or create a content store for the given project directory
    pub fn open(project_dir: &Path) -> Result<Self> {
        let db_path = db::content_db_path(project_dir)?;
        let conn = db::open_db(&db_path)?;
        schema::init_content_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Index content with a label, splitting into chunks
    pub fn index(&self, label: &str, content: &str, content_type: Option<&str>) -> Result<IndexResult> {
        let chunks = chunker::chunk_content(content);
        let now = chrono::Utc::now().to_rfc3339();

        let source_id: i64 = self.conn.query_row(
            "INSERT INTO sources (label, indexed_at) VALUES (?1, ?2) RETURNING id",
            rusqlite::params![label, now],
            |row| row.get(0),
        )?;

        let mut chunk_count = 0u32;
        let mut code_chunk_count = 0u32;

        for chunk in &chunks {
            let ct = content_type.unwrap_or_else(|| {
                if chunk.is_code { "code" } else { "prose" }
            });

            // Insert into primary FTS5 table
            self.conn.execute(
                "INSERT INTO chunks (title, content, content_type, source_id) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk.title, chunk.content, ct, source_id],
            )?;

            // Insert into trigram table
            self.conn.execute(
                "INSERT INTO chunks_trigram (title, content, content_type, source_id) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk.title, chunk.content, ct, source_id],
            )?;

            chunk_count += 1;
            if chunk.is_code {
                code_chunk_count += 1;
            }
        }

        self.conn.execute(
            "UPDATE sources SET chunk_count = ?1, code_chunk_count = ?2 WHERE id = ?3",
            rusqlite::params![chunk_count, code_chunk_count, source_id],
        )?;

        Ok(IndexResult {
            source_id,
            chunk_count,
            code_chunk_count,
        })
    }

    /// Search indexed content using the multi-layer search stack
    pub fn search(&self, query: &str, limit: u32, source_filter: Option<&str>, type_filter: Option<&str>) -> Result<Vec<search::SearchResult>> {
        search::multi_layer_search(&self.conn, query, limit, source_filter, type_filter)
    }

    /// List all indexed sources
    pub fn list_sources(&self) -> Result<Vec<SourceInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, label, indexed_at, chunk_count, code_chunk_count FROM sources ORDER BY indexed_at DESC"
        )?;
        let sources = stmt.query_map([], |row| {
            Ok(SourceInfo {
                id: row.get(0)?,
                label: row.get(1)?,
                indexed_at: row.get(2)?,
                chunk_count: row.get(3)?,
                code_chunk_count: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()
            .context("Failed to list sources")?;
        Ok(sources)
    }

    /// Get total bytes indexed (approximate from chunk content)
    pub fn total_bytes_indexed(&self) -> Result<u64> {
        let bytes: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM chunks",
            [],
            |row| row.get(0),
        )?;
        Ok(bytes as u64)
    }

    /// Get the underlying connection (for advanced queries)
    #[allow(dead_code)]
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct IndexResult {
    pub source_id: i64,
    pub chunk_count: u32,
    pub code_chunk_count: u32,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct SourceInfo {
    pub id: i64,
    pub label: String,
    pub indexed_at: String,
    pub chunk_count: u32,
    pub code_chunk_count: u32,
}
