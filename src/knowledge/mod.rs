//! Persistent knowledge store for cross-session semantic search.
//!
//! Indexes directories of files (markdown, code, documentation) into a durable
//! SQLite database with FTS5 + vector search. Parallel to the session-scoped
//! ContentStore — shares the same chunker and embedder but uses a separate
//! global database that survives across sessions.

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

/// Cross-module integration tests exercising the full knowledge store pipeline:
/// register → sync (with enrichment) → search (with filters) → reindex → remove.
///
/// These verify the spec's "Verification Contracts" and "Test Anchors" that span
/// multiple modules (P01 through P06).
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::knowledge::{enrichment, search, sync};
    use crate::store::search::SearchWeights;
    use rusqlite::params;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_store() -> KnowledgeStore {
        open_in_memory().expect("in-memory knowledge store")
    }

    fn make_store_with_embedder() -> KnowledgeStore {
        let mut store = make_store();
        store.set_embedder(Arc::new(crate::embedder::tests::MockEmbedder::new()));
        store
    }

    // ── Test Anchor 1: Incremental sync correctness ────────────────────────
    // "Create temp dir with 10 files. Sync. Modify 2, delete 1, add 1.
    //  Sync. Verify counts and content."

    #[test]
    fn full_sync_crud_cycle() {
        let store = make_store();
        let dir = tempdir().unwrap();

        // Create 10 files
        for i in 0..10 {
            std::fs::write(
                dir.path().join(format!("note{i}.md")),
                format!("---\ntype: note\nindex: {i}\n---\n# Note {i}\n\nContent for note {i}."),
            )
            .unwrap();
        }

        // Register and initial sync
        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);

        let r1 = sync::sync_source(&store, &source, enrich_ref).unwrap();
        assert_eq!(r1.files_added, 10);
        assert_eq!(r1.files_unchanged, 0);
        assert!(r1.chunks_total > 0);

        // Verify source stats updated
        let source = store.get_source("vault").unwrap().unwrap();
        assert_eq!(source.file_count, 10);
        assert!(source.chunk_count > 0);
        assert!(source.last_sync.is_some());

        // Modify 2, delete 1, add 1
        std::fs::write(
            dir.path().join("note0.md"),
            "---\ntype: note\nindex: 0\n---\n# Note 0\n\nModified content for note zero.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("note1.md"),
            "---\ntype: task\nindex: 1\n---\n# Task 1\n\nConverted to task type.",
        )
        .unwrap();
        std::fs::remove_file(dir.path().join("note9.md")).unwrap();
        std::fs::write(
            dir.path().join("note10.md"),
            "---\ntype: note\nindex: 10\n---\n# Note 10\n\nBrand new note.",
        )
        .unwrap();

        let source = store.get_source("vault").unwrap().unwrap();
        let r2 = sync::sync_source(&store, &source, enrich_ref).unwrap();
        assert_eq!(r2.files_added, 1);
        assert_eq!(r2.files_updated, 2);
        assert_eq!(r2.files_removed, 1);
        assert_eq!(r2.files_unchanged, 7);

        // Verify file count: 10 - 1 + 1 = 10
        let source = store.get_source("vault").unwrap().unwrap();
        assert_eq!(source.file_count, 10);

        // Verify modified content is present in chunks
        let chunks: Vec<String> = {
            let mut stmt = store
                .conn()
                .prepare(
                    "SELECT kc.content FROM knowledge_chunks kc
                     JOIN knowledge_chunk_meta kcm ON kcm.chunk_rowid = kc.rowid
                     WHERE kcm.source_id = ?1",
                )
                .unwrap();
            stmt.query_map(params![source.id], |row| row.get(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };
        let all_content = chunks.join("\n");
        assert!(
            all_content.contains("Modified content for note zero"),
            "updated content should be present"
        );
        assert!(
            all_content.contains("Brand new note"),
            "new file content should be present"
        );
    }

    // ── Test Anchor 2: Frontmatter extraction + filtered search ────────────
    // "Index file with YAML frontmatter. Search with filter type:task → match.
    //  Search type:project → no match. Search tag:foo → match."

    #[test]
    fn enrichment_to_filtered_search_pipeline() {
        let store = make_store();
        let dir = tempdir().unwrap();

        // Create files with different frontmatter
        std::fs::write(
            dir.path().join("task.md"),
            "---\ntype: task\nstatus: active\ntags:\n  - implementation\n  - system3\n---\n# Active Task\n\nThis task involves building the prediction engine.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("note.md"),
            "---\ntype: note\nstatus: draft\n---\n# Draft Note\n\nSome draft notes about the prediction engine.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("plain.md"),
            "# Plain File\n\nNo frontmatter here, just content about the prediction engine.",
        )
        .unwrap();

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        let weights = SearchWeights::default();

        // Filter type:task → should find only the task
        let results = search::knowledge_search(
            &store,
            "prediction engine",
            Some("type:task"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1, "type:task should match exactly one file");
        assert_eq!(results[0].metadata["frontmatter"]["type"], "task");

        // Filter type:project → should find nothing
        let results = search::knowledge_search(
            &store,
            "prediction engine",
            Some("type:project"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(results.is_empty(), "type:project should match nothing");

        // Filter tag:implementation → should find only the task
        let results = search::knowledge_search(
            &store,
            "prediction engine",
            Some("tag:implementation"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(
            results.len(),
            1,
            "tag:implementation should match exactly one file"
        );
        assert_eq!(results[0].metadata["frontmatter"]["type"], "task");

        // Filter status:active → should find only the task
        let results = search::knowledge_search(
            &store,
            "prediction engine",
            Some("status:active"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata["frontmatter"]["status"], "active");

        // No filter → should find all three
        let results = search::knowledge_search(
            &store,
            "prediction engine",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(
            results.len(),
            3,
            "unfiltered search should find all three files"
        );
    }

    // ── Test Anchor 3: Write-through re-index + immediate search ──────────
    // "Register source. Sync. Write new file. reindex_file. Search for it."

    #[test]
    fn write_through_reindex_then_search() {
        let store = make_store();
        let dir = tempdir().unwrap();

        // Initial file
        std::fs::write(
            dir.path().join("existing.md"),
            "# Existing\n\nOriginal content about authentication.",
        )
        .unwrap();

        store
            .add_source("docs", dir.path().to_str().unwrap(), None, &[])
            .unwrap();
        let source = store.get_source("docs").unwrap().unwrap();
        sync::sync_source(&store, &source, None).unwrap();

        // Write a new file (simulating Write tool)
        let new_path = dir.path().join("new-feature.md");
        std::fs::write(
            &new_path,
            "# New Feature\n\nImplementing the xylophone integration module.",
        )
        .unwrap();

        // Single-file reindex (what the hook triggers)
        let source = store.get_source("docs").unwrap().unwrap();
        sync::reindex_file(&store, &source, &new_path, None).unwrap();

        // Search should find the new content immediately
        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "xylophone integration",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(
            !results.is_empty(),
            "newly reindexed file should be searchable immediately"
        );
        assert!(results[0].snippet.contains("xylophone"));
    }

    // ── Test Anchor 7: Write-through edit updates search results ───────────
    // "Edit existing indexed file. Reindex. New content appears, old gone."

    #[test]
    fn write_through_edit_updates_content() {
        let store = make_store();
        let dir = tempdir().unwrap();

        let file_path = dir.path().join("evolving.md");
        std::fs::write(
            &file_path,
            "# Alpha Version\n\nOriginal implementation of the quasar protocol.",
        )
        .unwrap();

        store
            .add_source("docs", dir.path().to_str().unwrap(), None, &[])
            .unwrap();
        let source = store.get_source("docs").unwrap().unwrap();
        sync::sync_source(&store, &source, None).unwrap();

        // Verify original content is searchable
        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "quasar protocol",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(!results.is_empty(), "original content should be searchable");

        // Edit the file (simulating Edit tool)
        std::fs::write(
            &file_path,
            "# Beta Version\n\nReplacement nebula framework with improved throughput.",
        )
        .unwrap();

        let source = store.get_source("docs").unwrap().unwrap();
        sync::reindex_file(&store, &source, &file_path, None).unwrap();

        // New content is searchable
        let results = search::knowledge_search(
            &store,
            "nebula framework",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(
            !results.is_empty(),
            "updated content should be searchable after reindex"
        );

        // Old content is no longer present
        let results = search::knowledge_search(
            &store,
            "quasar protocol",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(
            results.is_empty(),
            "old content should be gone after reindex"
        );
    }

    // ── Invariant 2: Cascading deletion ────────────────────────────────────
    // "Removing a source deletes all files, chunks, chunk_meta, embeddings."

    #[test]
    fn cascading_deletion_leaves_zero_orphans() {
        let store = make_store_with_embedder();
        let dir = tempdir().unwrap();

        for i in 0..5 {
            std::fs::write(
                dir.path().join(format!("f{i}.md")),
                format!(
                    "---\ntype: task\n---\n# File {i}\n\nContent for file {i} with unique terms.",
                ),
            )
            .unwrap();
        }

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("doomed", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("doomed").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        // Verify data exists before removal
        let source = store.get_source("doomed").unwrap().unwrap();
        assert!(source.file_count > 0);
        assert!(source.chunk_count > 0);

        let embed_count_before: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_embeddings",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(embed_count_before > 0, "embeddings should exist before removal");

        // Remove the source
        let result = store.remove_source("doomed").unwrap();
        assert!(result.files_removed > 0);
        assert!(result.chunks_removed > 0);

        // Verify zero orphans in every table
        let orphaned_files: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_files",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_files, 0, "no files should remain");

        let orphaned_chunks: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunks",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_chunks, 0, "no FTS5 chunks should remain");

        let orphaned_trigram: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunks_trigram",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_trigram, 0, "no trigram chunks should remain");

        let orphaned_meta: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_meta, 0, "no chunk_meta should remain");

        let orphaned_embeds: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_embeddings",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_embeds, 0, "no embeddings should remain");

        let orphaned_sources: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_sources",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned_sources, 0, "no sources should remain");
    }

    // ── Invariant 4: Idempotent sync ───────────────────────────────────────
    // "Running sync twice with no changes produces identical state."

    #[test]
    fn idempotent_sync_no_changes() {
        let store = make_store();
        let dir = tempdir().unwrap();

        for i in 0..5 {
            std::fs::write(
                dir.path().join(format!("s{i}.md")),
                format!("# Stable {i}\n\nUnchanging content."),
            )
            .unwrap();
        }

        store
            .add_source("stable", dir.path().to_str().unwrap(), None, &[])
            .unwrap();
        let source = store.get_source("stable").unwrap().unwrap();
        sync::sync_source(&store, &source, None).unwrap();

        let source = store.get_source("stable").unwrap().unwrap();
        let chunks_after_first: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta WHERE source_id = ?1",
                params![source.id],
                |row| row.get(0),
            )
            .unwrap();

        // Second sync — no file changes
        let r2 = sync::sync_source(&store, &source, None).unwrap();
        assert_eq!(r2.files_unchanged, 5);
        assert_eq!(r2.files_added, 0);
        assert_eq!(r2.files_updated, 0);
        assert_eq!(r2.files_removed, 0);

        let chunks_after_second: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta WHERE source_id = ?1",
                params![source.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            chunks_after_first, chunks_after_second,
            "chunk count should be identical after idempotent sync"
        );
    }

    // ── Test Anchor 4: Empty knowledge graceful handling ───────────────────
    // "With no sources registered, search returns empty, not error."

    #[test]
    fn empty_knowledge_search_returns_empty() {
        let store = make_store();
        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "anything at all",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn empty_knowledge_filtered_search_returns_empty() {
        let store = make_store();
        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "anything",
            Some("type:task status:active"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(results.is_empty());
    }

    // ── All three enrichments + search ─────────────────────────────────────

    #[test]
    fn all_enrichments_indexed_and_searchable() {
        let store = make_store();
        let dir = tempdir().unwrap();

        // Create nested structure for folder_tags
        std::fs::create_dir_all(dir.path().join("01-projects/myapp")).unwrap();
        std::fs::write(
            dir.path().join("01-projects/myapp/spec.md"),
            "---\ntype: spec\nstatus: active\ntags:\n  - architecture\n---\n# App Spec\n\nSee [[other-note]] for details about the waveform analyzer.",
        )
        .unwrap();

        let enrichments = vec![
            "frontmatter".to_string(),
            "wikilinks".to_string(),
            "folder_tags".to_string(),
        ];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        let weights = SearchWeights::default();

        // Verify frontmatter filter works
        let results = search::knowledge_search(
            &store,
            "waveform analyzer",
            Some("type:spec"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);

        // Verify folder filter works
        let results = search::knowledge_search(
            &store,
            "waveform analyzer",
            Some("folder:01-projects"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);

        // Verify tag filter works
        let results = search::knowledge_search(
            &store,
            "waveform analyzer",
            Some("tag:architecture"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);

        // Verify metadata contains all three enrichment types
        let metadata = &results[0].metadata;
        assert!(
            metadata.get("frontmatter").is_some(),
            "metadata should contain frontmatter"
        );
        assert!(
            metadata.get("wikilinks").is_some(),
            "metadata should contain wikilinks"
        );
        assert!(
            metadata.get("folder_tags").is_some(),
            "metadata should contain folder_tags"
        );

        // Verify wikilinks extracted correctly
        let wikilinks = metadata["wikilinks"].as_array().unwrap();
        assert!(
            wikilinks
                .iter()
                .any(|l| l.as_str() == Some("other-note")),
            "wikilinks should contain 'other-note'"
        );

        // Verify folder_tags extracted correctly
        let folder_tags = metadata["folder_tags"].as_array().unwrap();
        assert!(
            folder_tags
                .iter()
                .any(|t| t.as_str() == Some("01-projects")),
            "folder_tags should contain '01-projects'"
        );
        assert!(
            folder_tags.iter().any(|t| t.as_str() == Some("myapp")),
            "folder_tags should contain 'myapp'"
        );
    }

    // ── Boundary probe: file with no frontmatter still searchable ──────────

    #[test]
    fn no_frontmatter_still_searchable() {
        let store = make_store();
        let dir = tempdir().unwrap();

        std::fs::write(
            dir.path().join("plain.md"),
            "# Plain Markdown\n\nNo frontmatter here, just the crystallography methodology.",
        )
        .unwrap();

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "crystallography",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(
            !results.is_empty(),
            "file without frontmatter should still be searchable by content"
        );
    }

    // ── Boundary probe: glob filter restricts sync ─────────────────────────

    #[test]
    fn glob_filter_restricts_sync_and_search() {
        let store = make_store();
        let dir = tempdir().unwrap();

        std::fs::write(
            dir.path().join("included.md"),
            "# Included\n\nThe photosynthesis discussion is here.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("excluded.txt"),
            "This txt file about photosynthesis should not be indexed.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("excluded.rs"),
            "// photosynthesis in rust, should not be indexed",
        )
        .unwrap();

        store
            .add_source(
                "md-only",
                dir.path().to_str().unwrap(),
                Some("*.md"),
                &[],
            )
            .unwrap();
        let source = store.get_source("md-only").unwrap().unwrap();
        let r = sync::sync_source(&store, &source, None).unwrap();
        assert_eq!(r.files_added, 1, "only .md files should be synced");

        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "photosynthesis",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(
            results.len(),
            1,
            "search should only find the .md file content"
        );
    }

    // ── Multiple sources with source filter ────────────────────────────────

    #[test]
    fn multi_source_with_source_filter() {
        let store = make_store();
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();

        std::fs::write(
            dir1.path().join("v.md"),
            "# Vault Note\n\nContent about the holographic rendering pipeline.",
        )
        .unwrap();
        std::fs::write(
            dir2.path().join("d.md"),
            "# Docs Note\n\nContent about the holographic rendering pipeline.",
        )
        .unwrap();

        store
            .add_source("vault", dir1.path().to_str().unwrap(), None, &[])
            .unwrap();
        store
            .add_source("docs", dir2.path().to_str().unwrap(), None, &[])
            .unwrap();

        let s1 = store.get_source("vault").unwrap().unwrap();
        let s2 = store.get_source("docs").unwrap().unwrap();
        sync::sync_source(&store, &s1, None).unwrap();
        sync::sync_source(&store, &s2, None).unwrap();

        let weights = SearchWeights::default();

        // Unfiltered → both sources
        let results = search::knowledge_search(
            &store,
            "holographic rendering",
            None,
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 2, "unfiltered should return from both sources");

        // Source filter → only vault
        let results = search::knowledge_search(
            &store,
            "holographic rendering",
            Some("source:vault"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_label, "vault");

        // Source label param → only docs
        let results = search::knowledge_search(
            &store,
            "holographic rendering",
            None,
            Some("docs"),
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_label, "docs");
    }

    // ── Invariant 3: No session contamination ──────────────────────────────
    // "Knowledge DB operations never affect session tables and vice versa."

    #[test]
    fn knowledge_and_session_tables_are_isolated() {
        let store = make_store();
        let conn = store.conn();

        // Knowledge tables should exist
        let k_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'knowledge_sources'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(k_count, 1, "knowledge_sources table should exist");

        // Session tables should NOT exist in the knowledge DB
        let s_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'events'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            s_count, 0,
            "session 'events' table should not exist in knowledge DB"
        );
    }

    // ── Invariant 5: Filter correctness ────────────────────────────────────
    // "A metadata filter never returns chunks whose metadata doesn't match."

    #[test]
    fn filter_never_returns_non_matching_chunks() {
        let store = make_store();
        let dir = tempdir().unwrap();

        // Create 5 files with different types
        for (i, ftype) in ["task", "note", "spec", "output", "log"].iter().enumerate() {
            std::fs::write(
                dir.path().join(format!("{ftype}.md")),
                format!(
                    "---\ntype: {ftype}\n---\n# {ftype} {i}\n\nContent about the electron microscope.",
                ),
            )
            .unwrap();
        }

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        let weights = SearchWeights::default();

        // Filter for type:task — every result must have type=task
        let results = search::knowledge_search(
            &store,
            "electron microscope",
            Some("type:task"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        for result in &results {
            assert_eq!(
                result.metadata["frontmatter"]["type"], "task",
                "every filtered result must match the filter predicate"
            );
        }
        assert_eq!(results.len(), 1, "exactly one task file should match");
    }

    // ── Invariant 6: Write-through freshness ───────────────────────────────
    // "After reindex_file, next search returns updated content."

    #[test]
    fn reindex_with_enrichments_updates_metadata() {
        let store = make_store();
        let dir = tempdir().unwrap();

        let file_path = dir.path().join("evolving.md");
        std::fs::write(
            &file_path,
            "---\ntype: task\nstatus: active\n---\n# Task\n\nWorking on the quantum simulator.",
        )
        .unwrap();

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        // Verify initial state: searchable with type:task
        let weights = SearchWeights::default();
        let results = search::knowledge_search(
            &store,
            "quantum simulator",
            Some("type:task"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);

        // Change the type from task → spec
        std::fs::write(
            &file_path,
            "---\ntype: spec\nstatus: done\n---\n# Spec\n\nCompleted quantum simulator specification.",
        )
        .unwrap();

        let source = store.get_source("vault").unwrap().unwrap();
        sync::reindex_file(&store, &source, &file_path, enrich_ref).unwrap();

        // Old filter should no longer match
        let results = search::knowledge_search(
            &store,
            "quantum simulator",
            Some("type:task"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert!(results.is_empty(), "type:task should no longer match after status change");

        // New filter should match
        let results = search::knowledge_search(
            &store,
            "quantum simulator",
            Some("type:spec"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1, "type:spec should match after reindex");
        assert_eq!(results[0].metadata["frontmatter"]["status"], "done");
    }

    // ── Boundary: source path no longer exists ─────────────────────────────

    #[test]
    fn sync_nonexistent_source_path_errors_gracefully() {
        let store = make_store();
        let dir = tempdir().unwrap();
        let path_str = dir.path().to_str().unwrap().to_string();

        store.add_source("temp", &path_str, None, &[]).unwrap();
        let source = store.get_source("temp").unwrap().unwrap();

        // Remove the directory
        drop(dir);

        // Sync should error but not panic
        let result = sync::sync_source(&store, &source, None);
        assert!(result.is_err(), "sync should error when source path is gone");
    }

    // ── Sync with embedder generates searchable vectors ────────────────────

    #[test]
    fn sync_with_embedder_produces_embeddings() {
        let store = make_store_with_embedder();
        let dir = tempdir().unwrap();

        std::fs::write(
            dir.path().join("embed.md"),
            "# Embedding Test\n\nVector embeddings for semantic search.",
        )
        .unwrap();

        store
            .add_source("emb", dir.path().to_str().unwrap(), None, &[])
            .unwrap();
        let source = store.get_source("emb").unwrap().unwrap();
        sync::sync_source(&store, &source, None).unwrap();

        let embed_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_embeddings",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(embed_count > 0, "embeddings should be generated during sync");

        // Every chunk should have a corresponding embedding
        let chunk_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            chunk_count, embed_count,
            "every chunk should have an embedding"
        );
    }

    // ── Filter parser → SQL → search roundtrip ─────────────────────────────

    #[test]
    fn compound_filter_roundtrip() {
        let store = make_store();
        let dir = tempdir().unwrap();

        std::fs::create_dir_all(dir.path().join("tasks")).unwrap();
        std::fs::write(
            dir.path().join("tasks/match.md"),
            "---\ntype: task\nstatus: active\ntags:\n  - urgent\n---\n# Matching Task\n\nContent about the spectral analysis.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tasks/mismatch-type.md"),
            "---\ntype: note\nstatus: active\ntags:\n  - urgent\n---\n# Mismatch Type\n\nContent about the spectral analysis.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tasks/mismatch-tag.md"),
            "---\ntype: task\nstatus: active\ntags:\n  - low\n---\n# Mismatch Tag\n\nContent about the spectral analysis.",
        )
        .unwrap();

        let enrichments = vec!["frontmatter".to_string(), "folder_tags".to_string()];
        store
            .add_source("vault", dir.path().to_str().unwrap(), None, &enrichments)
            .unwrap();
        let source = store.get_source("vault").unwrap().unwrap();
        let enrich = enrichment::build_enrichment_fn(&enrichments, dir.path());
        let enrich_ref = enrich
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> serde_json::Value);
        sync::sync_source(&store, &source, enrich_ref).unwrap();

        let weights = SearchWeights::default();

        // Compound filter: type:task AND tag:urgent AND folder:tasks
        let results = search::knowledge_search(
            &store,
            "spectral analysis",
            Some("type:task tag:urgent folder:tasks"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(
            results.len(),
            1,
            "compound filter should match exactly one file"
        );
        assert_eq!(results[0].title, "Matching Task");
    }
}
