pub mod chunker;
pub mod schema;
pub mod search;

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;
use std::sync::Arc;

use crate::db;
use crate::embedder::{self, Embed};

/// The main content store backed by SQLite FTS5
pub struct ContentStore {
    conn: Connection,
    embedder: Option<Arc<dyn Embed>>,
    /// Set to true after a one-time warning about unavailable embedder
    #[allow(dead_code)]
    embedder_warned: std::cell::Cell<bool>,
}

impl ContentStore {
    /// Open or create a content store for the given project directory (no embedder)
    pub fn open(project_dir: &Path) -> Result<Self> {
        let db_path = db::content_db_path(project_dir)?;
        let conn = db::open_db(&db_path)?;
        schema::init_content_schema(&conn)?;
        Ok(Self {
            conn,
            embedder: None,
            embedder_warned: std::cell::Cell::new(false),
        })
    }

    /// Open with an embedder for semantic search support
    #[allow(dead_code)]
    pub fn open_with_embedder(
        project_dir: &Path,
        embedder: Arc<dyn Embed>,
    ) -> Result<Self> {
        let db_path = db::content_db_path(project_dir)?;
        let conn = db::open_db(&db_path)?;
        schema::init_content_schema(&conn)?;
        Ok(Self {
            conn,
            embedder: Some(embedder),
            embedder_warned: std::cell::Cell::new(false),
        })
    }

    /// Set the embedder after construction (for lazy initialization)
    pub fn set_embedder(&mut self, embedder: Arc<dyn Embed>) {
        self.embedder = Some(embedder);
    }

    /// Whether this store has an active embedder
    #[allow(dead_code)]
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Get the one-time notice about embedder status for the first tool response.
    /// Returns Some(message) once if the embedder is not available, then None.
    #[allow(dead_code)]
    pub fn embedder_notice(&self) -> Option<String> {
        if self.embedder.is_none() && !self.embedder_warned.get() {
            self.embedder_warned.set(true);
            Some(
                "[bpx: semantic search unavailable — model not downloaded. \
                 Run `bpcontext embed-init` to download.]"
                    .to_string(),
            )
        } else {
            None
        }
    }

    /// Index content with a label, splitting into chunks.
    /// Generates embeddings for each chunk if an embedder is available.
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
        let mut chunk_rowids: Vec<i64> = Vec::with_capacity(chunks.len());
        let mut embed_texts: Vec<String> = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let ct = content_type.unwrap_or(
                if chunk.is_code { "code" } else { "prose" }
            );

            // Insert into primary FTS5 table
            self.conn.execute(
                "INSERT INTO chunks (title, content, content_type, source_id) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk.title, chunk.content, ct, source_id],
            )?;

            let rowid = self.conn.last_insert_rowid();

            // Insert into trigram table
            self.conn.execute(
                "INSERT INTO chunks_trigram (title, content, content_type, source_id) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk.title, chunk.content, ct, source_id],
            )?;

            chunk_rowids.push(rowid);
            embed_texts.push(format!("{} {}", chunk.title, chunk.content));

            chunk_count += 1;
            if chunk.is_code {
                code_chunk_count += 1;
            }
        }

        // Generate and store embeddings if embedder is available
        if let Some(ref embedder) = self.embedder {
            let text_refs: Vec<&str> = embed_texts.iter().map(|s| s.as_str()).collect();
            match embedder.embed_batch(&text_refs) {
                Ok(embeddings) => {
                    let dim = embedder.dim() as i32;
                    for (rowid, embedding) in chunk_rowids.iter().zip(embeddings.iter()) {
                        let blob = embedder::embedding_to_bytes(embedding);
                        self.conn.execute(
                            "INSERT OR REPLACE INTO chunk_embeddings (chunk_rowid, embedding, dim) VALUES (?1, ?2, ?3)",
                            rusqlite::params![rowid, blob, dim],
                        )?;
                    }
                }
                Err(e) => {
                    eprintln!("[bpcontext] embedding failed, skipping: {e}");
                }
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

    /// Search indexed content using the multi-layer search stack.
    /// Includes vector similarity (Layer 4) when an embedder is available.
    pub fn search(
        &self,
        query: &str,
        limit: u32,
        source_filter: Option<&str>,
        type_filter: Option<&str>,
    ) -> Result<Vec<search::SearchResult>> {
        self.search_with_weights(query, limit, source_filter, type_filter, &search::SearchWeights::default())
    }

    /// Search with explicit weight configuration for keyword vs vector layers.
    pub fn search_with_weights(
        &self,
        query: &str,
        limit: u32,
        source_filter: Option<&str>,
        type_filter: Option<&str>,
        weights: &search::SearchWeights,
    ) -> Result<Vec<search::SearchResult>> {
        let embedder_ref = self.embedder.as_deref();
        search::multi_layer_search(&self.conn, query, limit, source_filter, type_filter, embedder_ref, weights)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::tests::MockEmbedder;
    use crate::embedder::{bytes_to_embedding, EMBEDDING_DIM};

    /// Create an in-memory ContentStore for testing
    fn test_store() -> ContentStore {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        schema::init_content_schema(&conn).unwrap();
        ContentStore {
            conn,
            embedder: None,
            embedder_warned: std::cell::Cell::new(false),
        }
    }

    /// Create an in-memory ContentStore with a mock embedder
    fn test_store_with_embedder() -> ContentStore {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        schema::init_content_schema(&conn).unwrap();
        ContentStore {
            conn,
            embedder: Some(Arc::new(MockEmbedder::new())),
            embedder_warned: std::cell::Cell::new(false),
        }
    }

    #[test]
    fn index_without_embedder_stores_no_embeddings() {
        let store = test_store();
        let result = store.index("test", "Hello world, this is a test.", None).unwrap();
        assert!(result.chunk_count > 0);

        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn index_with_embedder_stores_embeddings() {
        let store = test_store_with_embedder();
        let result = store
            .index("test", "Hello world, this is a test of embedding storage.", None)
            .unwrap();
        assert!(result.chunk_count > 0);

        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count as u32, result.chunk_count);
    }

    #[test]
    fn stored_embeddings_have_correct_dimensions() {
        let store = test_store_with_embedder();
        store
            .index("test", "Some content to embed for dimension check.", None)
            .unwrap();

        let blob: Vec<u8> = store
            .conn
            .query_row(
                "SELECT embedding FROM chunk_embeddings LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();

        let embedding = bytes_to_embedding(&blob);
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn stored_embeddings_are_normalized() {
        let store = test_store_with_embedder();
        store
            .index("test", "Content for normalization verification.", None)
            .unwrap();

        let blob: Vec<u8> = store
            .conn
            .query_row(
                "SELECT embedding FROM chunk_embeddings LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();

        let embedding = bytes_to_embedding(&blob);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn embedder_notice_fires_once_when_no_embedder() {
        let store = test_store();

        let first = store.embedder_notice();
        assert!(first.is_some());
        assert!(first.unwrap().contains("semantic search unavailable"));

        let second = store.embedder_notice();
        assert!(second.is_none());
    }

    #[test]
    fn embedder_notice_never_fires_when_embedder_present() {
        let store = test_store_with_embedder();
        let notice = store.embedder_notice();
        assert!(notice.is_none());
    }

    #[test]
    fn index_with_embedder_links_to_correct_chunk_rowids() {
        let store = test_store_with_embedder();

        // Index enough content to produce multiple chunks
        let content = (0..20)
            .map(|i| format!("## Section {i}\n\nThis is section {i} with enough content to form a chunk.\n"))
            .collect::<String>();

        let result = store.index("multi-chunk", &content, None).unwrap();
        assert!(result.chunk_count > 1, "expected multiple chunks");

        // Every chunk should have a corresponding embedding
        let chunk_rowids: Vec<i64> = {
            let mut stmt = store
                .conn
                .prepare("SELECT rowid FROM chunks")
                .unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };

        let emb_rowids: Vec<i64> = {
            let mut stmt = store
                .conn
                .prepare("SELECT chunk_rowid FROM chunk_embeddings ORDER BY chunk_rowid")
                .unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };

        let mut sorted_chunks = chunk_rowids.clone();
        sorted_chunks.sort();
        assert_eq!(sorted_chunks, emb_rowids);
    }
}
