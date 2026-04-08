//! Persistent knowledge store for cross-session semantic search.
//!
//! Indexes directories of files (markdown, code, documentation) into a durable
//! SQLite database with FTS5 + vector search. Parallel to the session-scoped
//! ContentStore — shares the same chunker and embedder but uses a separate
//! global database that survives across sessions.

// Foundation module — all public items are consumed by later implementation
// prompts (P02-P06). Suppressing dead_code for the entire module is correct.
#![allow(dead_code)]

pub mod db;
pub mod enrichment;
pub mod filter;
pub mod search;
pub mod sync;

use anyhow::{anyhow, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;
use std::sync::Arc;

use crate::embedder::Embed;

/// Persistent knowledge store — indexes directories of files for durable
/// cross-session search.
pub struct KnowledgeStore {
    conn: Connection,
    embedder: Option<Arc<dyn Embed>>,
}

/// A registered knowledge source.
#[derive(Debug, Clone)]
pub struct KnowledgeSourceInfo {
    pub id: i64,
    pub label: String,
    pub path: String,
    pub glob: Option<String>,
    pub enrichments: Vec<String>,
    pub created_at: String,
    pub last_sync: Option<String>,
    pub file_count: i64,
    pub chunk_count: i64,
}

/// Result of removing a knowledge source.
#[derive(Debug)]
pub struct RemoveResult {
    pub files_removed: i64,
    pub chunks_removed: i64,
}

/// Parse a JSON-encoded enrichments array. Returns an empty vec on failure.
fn parse_enrichments(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Map a SQLite row to KnowledgeSourceInfo. Columns must be in the order:
/// id, label, path, glob, enrichments, created_at, last_sync, file_count, chunk_count
fn row_to_source(row: &rusqlite::Row<'_>) -> rusqlite::Result<KnowledgeSourceInfo> {
    let enrichments_json: String = row.get(4)?;
    Ok(KnowledgeSourceInfo {
        id: row.get(0)?,
        label: row.get(1)?,
        path: row.get(2)?,
        glob: row.get(3)?,
        enrichments: parse_enrichments(&enrichments_json),
        created_at: row.get(5)?,
        last_sync: row.get(6)?,
        file_count: row.get(7)?,
        chunk_count: row.get(8)?,
    })
}

const SOURCE_COLUMNS: &str =
    "id, label, path, glob, enrichments, created_at, last_sync, file_count, chunk_count";

impl KnowledgeStore {
    /// Open the knowledge store, creating the DB if needed.
    pub fn open() -> Result<Self> {
        let conn = db::open_knowledge_db()?;
        Ok(Self {
            conn,
            embedder: None,
        })
    }

    /// Open with a shared embedder instance (reuses the session embedder).
    pub fn open_with_embedder(embedder: Arc<dyn Embed>) -> Result<Self> {
        let conn = db::open_knowledge_db()?;
        Ok(Self {
            conn,
            embedder: Some(embedder),
        })
    }

    /// Set the embedder after construction.
    pub fn set_embedder(&mut self, embedder: Arc<dyn Embed>) {
        self.embedder = Some(embedder);
    }

    /// Check if an embedder is available.
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Register a new knowledge source. Returns the source ID.
    /// Errors if the label already exists.
    pub fn add_source(
        &self,
        label: &str,
        path: &str,
        glob: Option<&str>,
        enrichments: &[String],
    ) -> Result<i64> {
        let enrichments_json = serde_json::to_string(enrichments)?;
        self.conn.execute(
            "INSERT INTO knowledge_sources (label, path, glob, enrichments)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![label, path, glob, enrichments_json],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Remove a knowledge source and all its indexed content.
    /// FTS5 virtual tables don't support ON DELETE CASCADE so chunks are
    /// removed manually before deleting the source record.
    pub fn remove_source(&self, label: &str) -> Result<RemoveResult> {
        let source = self
            .get_source(label)?
            .ok_or_else(|| anyhow!("Knowledge source '{}' not found", label))?;

        let source_id = source.id;

        // Collect chunk rowids before any deletions so we still have references.
        let chunk_rowids: Vec<i64> = {
            let mut stmt = self.conn.prepare(
                "SELECT chunk_rowid FROM knowledge_chunk_meta WHERE source_id = ?1",
            )?;
            let rows = stmt.query_map(rusqlite::params![source_id], |row| row.get(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let chunks_removed = chunk_rowids.len() as i64;

        let files_removed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_files WHERE source_id = ?1",
            rusqlite::params![source_id],
            |row| row.get(0),
        )?;

        // Delete from FTS5 virtual tables first (no FK cascade support).
        for rowid in &chunk_rowids {
            self.conn.execute(
                "DELETE FROM knowledge_chunks WHERE rowid = ?1",
                rusqlite::params![rowid],
            )?;
            self.conn.execute(
                "DELETE FROM knowledge_chunks_trigram WHERE rowid = ?1",
                rusqlite::params![rowid],
            )?;
            self.conn.execute(
                "DELETE FROM knowledge_embeddings WHERE chunk_rowid = ?1",
                rusqlite::params![rowid],
            )?;
        }

        // Delete chunk_meta before the source (FK references files and source).
        self.conn.execute(
            "DELETE FROM knowledge_chunk_meta WHERE source_id = ?1",
            rusqlite::params![source_id],
        )?;

        // Delete the source — ON DELETE CASCADE removes knowledge_files.
        self.conn.execute(
            "DELETE FROM knowledge_sources WHERE id = ?1",
            rusqlite::params![source_id],
        )?;

        Ok(RemoveResult {
            files_removed,
            chunks_removed,
        })
    }

    /// List all registered knowledge sources.
    pub fn list_sources(&self) -> Result<Vec<KnowledgeSourceInfo>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {SOURCE_COLUMNS} FROM knowledge_sources ORDER BY label"
        ))?;
        let sources: Vec<KnowledgeSourceInfo> = stmt
            .query_map([], row_to_source)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(sources)
    }

    /// Find a source by label. Returns None if not found.
    pub fn get_source(&self, label: &str) -> Result<Option<KnowledgeSourceInfo>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {SOURCE_COLUMNS} FROM knowledge_sources WHERE label = ?1"
        ))?;
        let result = stmt
            .query_row(rusqlite::params![label], row_to_source)
            .optional()?;
        Ok(result)
    }

    /// Check if a file path falls inside any registered knowledge source.
    /// Returns the first matching source (sources are typically non-overlapping).
    pub fn source_for_path(&self, file_path: &Path) -> Result<Option<KnowledgeSourceInfo>> {
        let sources = self.list_sources()?;
        for source in sources {
            let source_path = Path::new(&source.path);
            if !file_path.starts_with(source_path) {
                continue;
            }
            // If the source has a glob filter, check it against the relative path.
            if let Some(ref glob_pattern) = source.glob {
                if let Ok(pattern) = glob::Pattern::new(glob_pattern) {
                    if let Ok(rel) = file_path.strip_prefix(source_path) {
                        if !pattern.matches_path(rel) {
                            continue;
                        }
                    }
                }
            }
            return Ok(Some(source));
        }
        Ok(None)
    }

    /// Get a reference to the underlying SQLite connection.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get a reference to the embedder, if available.
    pub fn embedder(&self) -> Option<&Arc<dyn Embed>> {
        self.embedder.as_ref()
    }
}

/// Open an in-memory knowledge store for tests. Uses the same schema as the
/// on-disk store but doesn't persist to disk.
#[cfg(test)]
pub(crate) fn open_in_memory() -> anyhow::Result<KnowledgeStore> {
    use crate::db::open_db;
    use std::path::PathBuf;
    let conn = open_db(&PathBuf::from(":memory:"))?;
    crate::knowledge::db::apply_schema(&conn)?;
    Ok(KnowledgeStore {
        conn,
        embedder: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn make_store() -> KnowledgeStore {
        open_in_memory().expect("in-memory knowledge store")
    }

    // ── Schema / open ────────────────────────────────────────────────────────

    #[test]
    fn open_is_idempotent() {
        // Opening twice should not error.
        let _a = make_store();
        let _b = make_store();
    }

    #[test]
    fn schema_tables_exist() {
        let store = make_store();
        let conn = store.conn();
        for table in &[
            "knowledge_sources",
            "knowledge_files",
            "knowledge_chunks",
            "knowledge_chunks_trigram",
            "knowledge_chunk_meta",
            "knowledge_embeddings",
        ] {
            let count: i64 = conn
                .query_row(
                    &format!(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','shadow') AND name = '{table}'"
                    ),
                    [],
                    |r| r.get(0),
                )
                .expect(table);
            assert!(count > 0, "table '{table}' should exist in schema");
        }
    }

    // ── add_source ───────────────────────────────────────────────────────────

    #[test]
    fn add_source_returns_id() {
        let store = make_store();
        let id = store
            .add_source("test", "/tmp/test", None, &[])
            .expect("add_source");
        assert!(id > 0);
    }

    #[test]
    fn add_source_duplicate_label_errors() {
        let store = make_store();
        store
            .add_source("test", "/tmp/test", None, &[])
            .expect("first add");
        let result = store.add_source("test", "/tmp/other", None, &[]);
        assert!(result.is_err(), "duplicate label should error");
    }

    #[test]
    fn add_source_stores_enrichments_as_json() {
        let store = make_store();
        let enrichments = vec!["frontmatter".to_string(), "wikilinks".to_string()];
        store
            .add_source("e", "/tmp/e", None, &enrichments)
            .expect("add");
        let src = store.get_source("e").unwrap().unwrap();
        assert_eq!(src.enrichments, enrichments);
    }

    // ── list_sources / get_source ─────────────────────────────────────────────

    #[test]
    fn list_sources_empty() {
        let store = make_store();
        assert!(store.list_sources().unwrap().is_empty());
    }

    #[test]
    fn list_sources_returns_added() {
        let store = make_store();
        store
            .add_source("alpha", "/tmp/alpha", None, &[])
            .unwrap();
        store
            .add_source("beta", "/tmp/beta", Some("**/*.md"), &[])
            .unwrap();
        let sources = store.list_sources().unwrap();
        assert_eq!(sources.len(), 2);
        // Ordered by label
        assert_eq!(sources[0].label, "alpha");
        assert_eq!(sources[1].label, "beta");
        assert_eq!(sources[1].glob, Some("**/*.md".to_string()));
    }

    #[test]
    fn get_source_returns_correct_fields() {
        let store = make_store();
        let enr = vec!["frontmatter".to_string()];
        let id = store
            .add_source("vault", "/tmp/vault", Some("**/*.md"), &enr)
            .unwrap();
        let src = store.get_source("vault").unwrap().unwrap();
        assert_eq!(src.id, id);
        assert_eq!(src.label, "vault");
        assert_eq!(src.path, "/tmp/vault");
        assert_eq!(src.glob, Some("**/*.md".to_string()));
        assert_eq!(src.enrichments, enr);
        assert_eq!(src.file_count, 0);
        assert_eq!(src.chunk_count, 0);
        assert!(src.last_sync.is_none());
    }

    #[test]
    fn get_source_nonexistent_returns_none() {
        let store = make_store();
        assert!(store.get_source("nonexistent").unwrap().is_none());
    }

    // ── remove_source ─────────────────────────────────────────────────────────

    #[test]
    fn remove_source_deletes_record() {
        let store = make_store();
        store
            .add_source("gone", "/tmp/gone", None, &[])
            .unwrap();
        store.remove_source("gone").expect("remove");
        assert!(store.get_source("gone").unwrap().is_none());
    }

    #[test]
    fn remove_source_returns_zero_counts_when_no_files() {
        let store = make_store();
        store
            .add_source("empty", "/tmp/empty", None, &[])
            .unwrap();
        let result = store.remove_source("empty").unwrap();
        assert_eq!(result.files_removed, 0);
        assert_eq!(result.chunks_removed, 0);
    }

    #[test]
    fn remove_source_nonexistent_errors() {
        let store = make_store();
        assert!(store.remove_source("nosuchsource").is_err());
    }

    // ── source_for_path ───────────────────────────────────────────────────────

    #[test]
    fn source_for_path_matches_registered_source() {
        let store = make_store();
        store
            .add_source("docs", "/tmp/docs", None, &[])
            .unwrap();
        let found = store
            .source_for_path(Path::new("/tmp/docs/foo.md"))
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().label, "docs");
    }

    #[test]
    fn source_for_path_no_match_returns_none() {
        let store = make_store();
        store
            .add_source("docs", "/tmp/docs", None, &[])
            .unwrap();
        let found = store
            .source_for_path(Path::new("/other/path/file.md"))
            .unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn source_for_path_respects_glob_filter() {
        let store = make_store();
        store
            .add_source("md-only", "/tmp/md", Some("**/*.md"), &[])
            .unwrap();
        // .md file — should match
        assert!(store
            .source_for_path(Path::new("/tmp/md/notes/foo.md"))
            .unwrap()
            .is_some());
        // .rs file — should NOT match
        assert!(store
            .source_for_path(Path::new("/tmp/md/src/lib.rs"))
            .unwrap()
            .is_none());
    }

    // ── embedder ──────────────────────────────────────────────────────────────

    #[test]
    fn has_embedder_false_when_not_set() {
        let store = make_store();
        assert!(!store.has_embedder());
    }
}
