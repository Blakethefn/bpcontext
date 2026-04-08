//! Incremental sync engine for the knowledge store.
//!
//! Walks registered knowledge source directories, compares file content hashes
//! against the database, and re-indexes only changed files. Provides
//! `reindex_file()` for single-file updates used by the write-through hook (P05).

use anyhow::{Context, Result};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, io};

use crate::embedder::{embedding_to_bytes, Embed};
use crate::store::chunker::chunk_content;

use super::{KnowledgeSourceInfo, KnowledgeStore};

/// Result of a knowledge sync operation.
#[derive(Debug, serde::Serialize)]
pub struct SyncResult {
    pub source_label: String,
    pub files_scanned: usize,
    pub files_added: usize,
    pub files_updated: usize,
    pub files_removed: usize,
    pub files_unchanged: usize,
    pub chunks_total: usize,
    pub duration_ms: u64,
}

/// Sync a single knowledge source. Walks the directory, hashes files,
/// and re-indexes only changed/new files. Removes chunks for deleted files.
pub fn sync_source(
    store: &KnowledgeStore,
    source: &KnowledgeSourceInfo,
    enrichment_fn: Option<&dyn Fn(&str, &Path) -> serde_json::Value>,
) -> Result<SyncResult> {
    let start = Instant::now();
    let conn = store.conn();
    let embedder = store.embedder();
    let base_path = Path::new(&source.path);

    // 1. Walk directory (respects .gitignore, skips binary/empty files)
    let disk_files = walk_source_dir(base_path, source.glob.as_deref())?;

    // 2. Load existing file records from DB
    let db_files = load_db_files(conn, source.id)?;

    // 3. Classify files: read once, hash, decode UTF-8, then classify
    let mut to_add: Vec<(String, PathBuf, String, String)> = Vec::new(); // (rel, abs, content, hash)
    let mut to_update: Vec<(i64, PathBuf, String, String)> = Vec::new(); // (file_id, abs, content, hash)
    let mut to_remove: Vec<i64> = Vec::new();
    let mut unchanged: usize = 0;

    for (rel_path, abs_path) in &disk_files {
        let bytes = fs::read(abs_path)
            .with_context(|| format!("failed to read {}", abs_path.display()))?;

        // Skip non-UTF-8 files silently
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let hash = sha256_hex(content.as_bytes());

        if let Some((file_id, old_hash)) = db_files.get(rel_path) {
            if *old_hash != hash {
                to_update.push((*file_id, abs_path.clone(), content, hash));
            } else {
                unchanged += 1;
            }
        } else {
            to_add.push((rel_path.clone(), abs_path.clone(), content, hash));
        }
    }

    for (rel_path, (file_id, _)) in &db_files {
        if !disk_files.contains_key(rel_path) {
            to_remove.push(*file_id);
        }
    }

    let files_scanned = disk_files.len();
    let files_added = to_add.len();
    let files_updated = to_update.len();
    let files_removed = to_remove.len();

    // 4. Execute in a single transaction
    conn.execute_batch("BEGIN IMMEDIATE")?;

    let inner = (|| -> Result<()> {
        // 4a. Remove deleted files
        for file_id in &to_remove {
            delete_file_chunks(conn, *file_id)?;
            conn.execute("DELETE FROM knowledge_files WHERE id = ?1", params![file_id])?;
        }

        // 4b. Update changed files (delete old chunks, re-index)
        for (file_id, abs_path, content, hash) in &to_update {
            delete_file_chunks(conn, *file_id)?;
            index_file_inner(
                conn,
                embedder,
                source.id,
                *file_id,
                content,
                abs_path,
                hash,
                content.len(),
                enrichment_fn,
            )?;
        }

        // 4c. Add new files
        for (rel_path, abs_path, content, hash) in &to_add {
            conn.execute(
                "INSERT INTO knowledge_files (source_id, rel_path, content_hash, size_bytes)
                 VALUES (?1, ?2, ?3, ?4)",
                params![source.id, rel_path, hash, content.len() as i64],
            )?;
            let file_id = conn.last_insert_rowid();
            index_file_inner(
                conn,
                embedder,
                source.id,
                file_id,
                content,
                abs_path,
                hash,
                content.len(),
                enrichment_fn,
            )?;
        }

        // 4d. Update source stats
        conn.execute(
            "UPDATE knowledge_sources SET
                last_sync = datetime('now'),
                file_count = (SELECT COUNT(*) FROM knowledge_files WHERE source_id = ?1),
                chunk_count = (SELECT COUNT(*) FROM knowledge_chunk_meta WHERE source_id = ?1)
             WHERE id = ?1",
            params![source.id],
        )?;

        Ok(())
    })();

    match inner {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    let chunks_total: i64 = conn.query_row(
        "SELECT chunk_count FROM knowledge_sources WHERE id = ?1",
        params![source.id],
        |row| row.get(0),
    )?;

    Ok(SyncResult {
        source_label: source.label.clone(),
        files_scanned,
        files_added,
        files_updated,
        files_removed,
        files_unchanged: unchanged,
        chunks_total: chunks_total as usize,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Sync all registered knowledge sources.
pub fn sync_all(
    store: &KnowledgeStore,
    enrichment_fn: Option<&dyn Fn(&str, &Path) -> serde_json::Value>,
) -> Result<Vec<SyncResult>> {
    let sources = store.list_sources()?;
    let mut results = Vec::with_capacity(sources.len());
    for source in &sources {
        results.push(sync_source(store, source, enrichment_fn)?);
    }
    Ok(results)
}

/// Re-index a single file within a knowledge source.
/// Used by the write-through hook for immediate updates.
pub fn reindex_file(
    store: &KnowledgeStore,
    source: &KnowledgeSourceInfo,
    abs_path: &Path,
    enrichment_fn: Option<&dyn Fn(&str, &Path) -> serde_json::Value>,
) -> Result<()> {
    let conn = store.conn();
    let embedder = store.embedder();
    let base_path = Path::new(&source.path);

    let rel_path = abs_path
        .strip_prefix(base_path)
        .context("file path is not inside the source directory")?
        .to_string_lossy()
        .to_string();

    let bytes =
        fs::read(abs_path).with_context(|| format!("failed to read {}", abs_path.display()))?;
    let content =
        String::from_utf8(bytes).context("file content is not valid UTF-8 for reindex")?;
    let hash = sha256_hex(content.as_bytes());

    // Check if file exists and hash matches — skip early if unchanged
    let existing: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, content_hash FROM knowledge_files
             WHERE source_id = ?1 AND rel_path = ?2",
            params![source.id, rel_path],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;

    if let Some((_, ref old_hash)) = existing {
        if *old_hash == hash {
            return Ok(()); // No change
        }
    }

    conn.execute_batch("BEGIN IMMEDIATE")?;

    let inner = (|| -> Result<()> {
        if let Some((file_id, _)) = existing {
            delete_file_chunks(conn, file_id)?;
            index_file_inner(
                conn,
                embedder,
                source.id,
                file_id,
                &content,
                abs_path,
                &hash,
                content.len(),
                enrichment_fn,
            )?;
        } else {
            conn.execute(
                "INSERT INTO knowledge_files (source_id, rel_path, content_hash, size_bytes)
                 VALUES (?1, ?2, ?3, ?4)",
                params![source.id, rel_path, hash, content.len() as i64],
            )?;
            let file_id = conn.last_insert_rowid();
            index_file_inner(
                conn,
                embedder,
                source.id,
                file_id,
                &content,
                abs_path,
                &hash,
                content.len(),
                enrichment_fn,
            )?;
        }

        conn.execute(
            "UPDATE knowledge_sources SET
                file_count = (SELECT COUNT(*) FROM knowledge_files WHERE source_id = ?1),
                chunk_count = (SELECT COUNT(*) FROM knowledge_chunk_meta WHERE source_id = ?1)
             WHERE id = ?1",
            params![source.id],
        )?;

        Ok(())
    })();

    match inner {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Walk a knowledge source directory, returning files keyed by relative path.
/// Respects .gitignore, skips binary files, empty files, and applies glob filter.
fn walk_source_dir(
    base_path: &Path,
    glob_pattern: Option<&str>,
) -> Result<HashMap<String, PathBuf>> {
    if !base_path.is_dir() {
        return Err(anyhow::anyhow!(
            "knowledge source path is not a directory: {}",
            base_path.display()
        ));
    }

    let mut builder = WalkBuilder::new(base_path);
    builder
        .hidden(false) // Don't skip hidden files (dotfiles may be relevant)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    if let Some(pattern) = glob_pattern {
        let mut overrides = OverrideBuilder::new(base_path);
        overrides.add(pattern).context("invalid glob pattern")?;
        builder.overrides(overrides.build()?);
    }

    let mut files = HashMap::new();

    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let abs_path = entry.path().to_path_buf();

        // Skip binary files (null byte in first 1KB)
        match is_binary(&abs_path) {
            Ok(true) | Err(_) => continue,
            Ok(false) => {}
        }

        // Skip empty files
        match fs::metadata(&abs_path) {
            Ok(m) if m.len() == 0 => continue,
            Err(_) => continue,
            _ => {}
        }

        let rel_path = abs_path
            .strip_prefix(base_path)
            .unwrap_or(&abs_path)
            .to_string_lossy()
            .to_string();

        files.insert(rel_path, abs_path);
    }

    Ok(files)
}

/// Compute SHA-256 hex digest of bytes.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Load existing file records from the knowledge DB for a source.
fn load_db_files(
    conn: &rusqlite::Connection,
    source_id: i64,
) -> Result<HashMap<String, (i64, String)>> {
    let mut stmt = conn.prepare(
        "SELECT id, rel_path, content_hash FROM knowledge_files WHERE source_id = ?1",
    )?;
    let mut files = HashMap::new();
    let rows = stmt.query_map(params![source_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    for row in rows {
        let (id, rel_path, hash) = row?;
        files.insert(rel_path, (id, hash));
    }
    Ok(files)
}

/// Index a single file: chunk content, insert FTS5 + trigram + meta + embeddings,
/// and update the file record with hash and chunk count.
fn index_file_inner(
    conn: &rusqlite::Connection,
    embedder: Option<&Arc<dyn Embed>>,
    source_id: i64,
    file_id: i64,
    content: &str,
    abs_path: &Path,
    hash: &str,
    size_bytes: usize,
    enrichment_fn: Option<&dyn Fn(&str, &Path) -> serde_json::Value>,
) -> Result<usize> {
    let chunks = chunk_content(content);

    let metadata = match enrichment_fn {
        Some(f) => f(content, abs_path),
        None => serde_json::json!({}),
    };
    let metadata_str = metadata.to_string();

    let mut texts_for_embedding: Vec<String> = Vec::new();
    let mut chunk_rowids: Vec<i64> = Vec::new();
    let mut chunk_count: usize = 0;

    for chunk in &chunks {
        // FTS5 (BM25)
        conn.execute(
            "INSERT INTO knowledge_chunks(title, content) VALUES (?1, ?2)",
            params![chunk.title, chunk.content],
        )?;
        let rowid = conn.last_insert_rowid();

        // Trigram FTS5 with explicit rowid
        conn.execute(
            "INSERT INTO knowledge_chunks_trigram(rowid, title, content) VALUES (?1, ?2, ?3)",
            params![rowid, chunk.title, chunk.content],
        )?;

        // Chunk metadata
        conn.execute(
            "INSERT INTO knowledge_chunk_meta
                (chunk_rowid, file_id, source_id, line_start, line_end, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                rowid,
                file_id,
                source_id,
                chunk.line_start,
                chunk.line_end,
                metadata_str
            ],
        )?;

        texts_for_embedding.push(format!("{} {}", chunk.title, chunk.content));
        chunk_rowids.push(rowid);
        chunk_count += 1;
    }

    // Batch embed if embedder available
    if let Some(emb) = embedder {
        let refs: Vec<&str> = texts_for_embedding.iter().map(|s| s.as_str()).collect();
        let embeddings = emb.embed_batch(&refs)?;
        for (rowid, embedding) in chunk_rowids.iter().zip(embeddings.iter()) {
            let blob = embedding_to_bytes(embedding);
            conn.execute(
                "INSERT INTO knowledge_embeddings (chunk_rowid, embedding, dim)
                 VALUES (?1, ?2, ?3)",
                params![rowid, blob, embedding.len() as i64],
            )?;
        }
    }

    // Update file record
    conn.execute(
        "UPDATE knowledge_files SET content_hash = ?1, size_bytes = ?2,
            chunk_count = ?3, indexed_at = datetime('now')
         WHERE id = ?4",
        params![hash, size_bytes as i64, chunk_count as i64, file_id],
    )?;

    Ok(chunk_count)
}

/// Delete all chunks (FTS5, trigram, embeddings, metadata) for a given file.
fn delete_file_chunks(conn: &rusqlite::Connection, file_id: i64) -> Result<()> {
    let rowids: Vec<i64> = {
        let mut stmt =
            conn.prepare("SELECT chunk_rowid FROM knowledge_chunk_meta WHERE file_id = ?1")?;
        let rows = stmt.query_map(params![file_id], |row| row.get(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };

    // FTS5 tables: delete one at a time (safest for all SQLite versions)
    for rowid in &rowids {
        conn.execute(
            "DELETE FROM knowledge_chunks WHERE rowid = ?1",
            params![rowid],
        )?;
        conn.execute(
            "DELETE FROM knowledge_chunks_trigram WHERE rowid = ?1",
            params![rowid],
        )?;
    }

    // Embeddings and metadata (regular tables, bulk delete is fine)
    conn.execute(
        "DELETE FROM knowledge_embeddings WHERE chunk_rowid IN
         (SELECT chunk_rowid FROM knowledge_chunk_meta WHERE file_id = ?1)",
        params![file_id],
    )?;
    conn.execute(
        "DELETE FROM knowledge_chunk_meta WHERE file_id = ?1",
        params![file_id],
    )?;

    Ok(())
}

/// Check if a file is binary by looking for null bytes in the first 1KB.
fn is_binary(path: &Path) -> io::Result<bool> {
    let mut file = fs::File::open(path)?;
    let mut buf = [0u8; 1024];
    let bytes_read = file.read(&mut buf)?;
    Ok(buf[..bytes_read].contains(&0u8))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::embedder::tests::MockEmbedder;

    fn make_store() -> KnowledgeStore {
        crate::knowledge::open_in_memory().expect("in-memory knowledge store")
    }

    fn make_store_with_embedder() -> KnowledgeStore {
        let mut store = make_store();
        store.set_embedder(Arc::new(MockEmbedder::new()));
        store
    }

    fn register_source(store: &KnowledgeStore, dir: &Path) -> KnowledgeSourceInfo {
        store
            .add_source("test", dir.to_str().unwrap(), None, &[])
            .unwrap();
        store.get_source("test").unwrap().unwrap()
    }

    fn register_source_glob(
        store: &KnowledgeStore,
        dir: &Path,
        glob: &str,
    ) -> KnowledgeSourceInfo {
        store
            .add_source("test", dir.to_str().unwrap(), Some(glob), &[])
            .unwrap();
        store.get_source("test").unwrap().unwrap()
    }

    fn chunk_count_for_source(store: &KnowledgeStore, source_id: i64) -> i64 {
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta WHERE source_id = ?1",
                params![source_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn file_count_for_source(store: &KnowledgeStore, source_id: i64) -> i64 {
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_files WHERE source_id = ?1",
                params![source_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    // ── sync_source ─────────────────────────────────────────────────────────

    #[test]
    fn sync_new_source_indexes_all_files() {
        let store = make_store();
        let dir = tempdir().unwrap();
        for i in 0..5 {
            fs::write(
                dir.path().join(format!("file{i}.md")),
                format!("# File {i}\nContent for file {i}"),
            )
            .unwrap();
        }
        let source = register_source(&store, dir.path());
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_added, 5);
        assert_eq!(result.files_unchanged, 0);
        assert_eq!(result.files_updated, 0);
        assert_eq!(result.files_removed, 0);
        assert!(result.chunks_total > 0, "should produce chunks");
        assert_eq!(file_count_for_source(&store, source.id), 5);
    }

    #[test]
    fn sync_unchanged_reports_zero_changes() {
        let store = make_store();
        let dir = tempdir().unwrap();
        for i in 0..5 {
            fs::write(
                dir.path().join(format!("f{i}.md")),
                format!("# F{i}\nBody"),
            )
            .unwrap();
        }
        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        let source = store.get_source("test").unwrap().unwrap();
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_unchanged, 5);
        assert_eq!(result.files_added, 0);
        assert_eq!(result.files_updated, 0);
        assert_eq!(result.files_removed, 0);
    }

    #[test]
    fn sync_detects_add_update_remove() {
        let store = make_store();
        let dir = tempdir().unwrap();
        for i in 0..5 {
            fs::write(
                dir.path().join(format!("f{i}.md")),
                format!("# F{i}\nOriginal"),
            )
            .unwrap();
        }
        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        // Modify 2, delete 1, add 1
        fs::write(dir.path().join("f0.md"), "# F0\nModified content").unwrap();
        fs::write(dir.path().join("f1.md"), "# F1\nAlso modified").unwrap();
        fs::remove_file(dir.path().join("f4.md")).unwrap();
        fs::write(dir.path().join("f5.md"), "# F5\nBrand new").unwrap();

        let source = store.get_source("test").unwrap().unwrap();
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_added, 1);
        assert_eq!(result.files_updated, 2);
        assert_eq!(result.files_removed, 1);
        assert_eq!(result.files_unchanged, 2);
        assert_eq!(file_count_for_source(&store, source.id), 5); // 5-1+1 = 5
    }

    #[test]
    fn sync_empty_directory() {
        let store = make_store();
        let dir = tempdir().unwrap();
        let source = register_source(&store, dir.path());
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_scanned, 0);
        assert_eq!(result.files_added, 0);
        assert_eq!(result.chunks_total, 0);
    }

    #[test]
    fn sync_skips_binary_files() {
        let store = make_store();
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("text.md"), "# Hello\nText").unwrap();
        fs::write(dir.path().join("binary.dat"), b"data\x00\x00\x01").unwrap();

        let source = register_source(&store, dir.path());
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_added, 1, "only text file should be indexed");
    }

    #[test]
    fn sync_respects_glob_filter() {
        let store = make_store();
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("notes.md"), "# Notes\nContent").unwrap();
        fs::write(dir.path().join("readme.txt"), "Some text").unwrap();
        fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();

        let source = register_source_glob(&store, dir.path(), "*.md");
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_added, 1, "only .md files should match");
    }

    // ── reindex_file ────────────────────────────────────────────────────────

    #[test]
    fn reindex_file_skips_unchanged() {
        let store = make_store();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.md");
        fs::write(&file_path, "# Test\nContent here").unwrap();

        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        let before = chunk_count_for_source(&store, source.id);
        let source = store.get_source("test").unwrap().unwrap();
        reindex_file(&store, &source, &file_path, None).unwrap();
        let after = chunk_count_for_source(&store, source.id);

        assert_eq!(before, after, "unchanged file should not alter chunks");
    }

    #[test]
    fn reindex_file_updates_changed_content() {
        let store = make_store();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.md");
        fs::write(&file_path, "# Original\nOld content only").unwrap();

        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        // Modify the file
        fs::write(
            &file_path,
            "# Replacement\nCompletely different text\n\n## Extra\nMore sections",
        )
        .unwrap();

        let source = store.get_source("test").unwrap().unwrap();
        reindex_file(&store, &source, &file_path, None).unwrap();

        // Verify new content is present and old content is gone
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
        let all = chunks.join("\n");
        assert!(all.contains("Replacement"), "new content should be present");
        assert!(
            !all.contains("Original"),
            "old content should have been replaced"
        );
    }

    #[test]
    fn reindex_file_adds_new_file() {
        let store = make_store();
        let dir = tempdir().unwrap();
        let source = register_source(&store, dir.path());

        let file_path = dir.path().join("new.md");
        fs::write(&file_path, "# Brand New\nFresh content").unwrap();
        reindex_file(&store, &source, &file_path, None).unwrap();

        assert_eq!(file_count_for_source(&store, source.id), 1);
        assert!(chunk_count_for_source(&store, source.id) > 0);
    }

    // ── Embeddings ──────────────────────────────────────────────────────────

    #[test]
    fn sync_generates_embeddings_when_embedder_present() {
        let store = make_store_with_embedder();
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("test.md"),
            "# Embedding Test\nContent for embedding generation",
        )
        .unwrap();

        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        let count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_embeddings",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count > 0, "embeddings should be generated");
    }

    #[test]
    fn sync_works_without_embedder() {
        let store = make_store(); // no embedder
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("test.md"), "# No Embed\nStill works").unwrap();

        let source = register_source(&store, dir.path());
        let result = sync_source(&store, &source, None).unwrap();

        assert_eq!(result.files_added, 1);
        let embed_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_embeddings",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(embed_count, 0, "no embeddings without embedder");
    }

    // ── FTS5 cleanup ────────────────────────────────────────────────────────

    #[test]
    fn no_orphaned_chunks_after_file_removal() {
        let store = make_store();
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("keep.md"), "# Keep\nPersistent").unwrap();
        fs::write(dir.path().join("gone.md"), "# Gone\nWill be deleted").unwrap();

        let source = register_source(&store, dir.path());
        sync_source(&store, &source, None).unwrap();

        fs::remove_file(dir.path().join("gone.md")).unwrap();
        let source = store.get_source("test").unwrap().unwrap();
        sync_source(&store, &source, None).unwrap();

        // No orphaned chunk_meta (meta pointing to deleted files)
        let orphaned: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta kcm
                 LEFT JOIN knowledge_files kf ON kcm.file_id = kf.id
                 WHERE kf.id IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(orphaned, 0, "no orphaned chunk_meta");

        // Chunk count in knowledge_chunks matches chunk_meta count
        let fts_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunks",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let meta_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_chunk_meta",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fts_count, meta_count, "FTS5 and meta counts should match");
    }

    // ── sync_all ────────────────────────────────────────────────────────────

    #[test]
    fn sync_all_handles_multiple_sources() {
        let store = make_store();
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();
        fs::write(dir1.path().join("a.md"), "# A\nFirst source").unwrap();
        fs::write(dir2.path().join("b.md"), "# B\nSecond source").unwrap();

        store
            .add_source("src1", dir1.path().to_str().unwrap(), None, &[])
            .unwrap();
        store
            .add_source("src2", dir2.path().to_str().unwrap(), None, &[])
            .unwrap();

        let results = sync_all(&store, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].files_added, 1);
        assert_eq!(results[1].files_added, 1);
    }

    // ── enrichment_fn ───────────────────────────────────────────────────────

    #[test]
    fn enrichment_fn_metadata_stored() {
        let store = make_store();
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("test.md"), "# Enriched\nContent").unwrap();
        let source = register_source(&store, dir.path());

        let enrich = |_content: &str, _path: &Path| -> serde_json::Value {
            serde_json::json!({"frontmatter": {"type": "task", "status": "active"}})
        };

        sync_source(&store, &source, Some(&enrich)).unwrap();

        let metadata: String = store
            .conn()
            .query_row(
                "SELECT metadata FROM knowledge_chunk_meta WHERE source_id = ?1 LIMIT 1",
                params![source.id],
                |row| row.get(0),
            )
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&metadata).unwrap();
        assert_eq!(parsed["frontmatter"]["type"], "task");
        assert_eq!(parsed["frontmatter"]["status"], "active");
    }

    // ── walk_source_dir ─────────────────────────────────────────────────────

    #[test]
    fn walk_skips_empty_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("empty.md"), "").unwrap();
        fs::write(dir.path().join("content.md"), "has content").unwrap();

        let files = walk_source_dir(dir.path(), None).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("content.md"));
    }

    #[test]
    fn walk_errors_on_nonexistent_path() {
        let result = walk_source_dir(Path::new("/tmp/surely_nonexistent_dir_abc123"), None);
        assert!(result.is_err());
    }

    // ── sha256_hex ──────────────────────────────────────────────────────────

    #[test]
    fn sha256_produces_hex_string() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(hash.len(), 64, "SHA-256 hex is 64 chars");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sha256_is_deterministic() {
        assert_eq!(sha256_hex(b"same input"), sha256_hex(b"same input"));
    }

    #[test]
    fn sha256_differs_for_different_input() {
        assert_ne!(sha256_hex(b"input a"), sha256_hex(b"input b"));
    }
}
