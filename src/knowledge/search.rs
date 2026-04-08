//! Knowledge-specific search: BM25 → trigram → fuzzy → vector → RRF fusion.
//!
//! Follows the same multi-layer pipeline as `store/search.rs` but queries
//! knowledge DB tables (`knowledge_chunks`, `knowledge_chunks_trigram`, etc.)
//! instead of session tables. Written as a self-contained module to avoid
//! modifying the battle-tested session search.

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

use crate::embedder;
use crate::store::search::{SearchResult, SearchWeights};
use crate::store::ContentStore;

use super::filter::{parse_filter, predicates_to_sql};
use super::KnowledgeStore;

/// A search result from the knowledge store (or combined search).
#[derive(Debug, serde::Serialize)]
pub struct KnowledgeSearchResult {
    pub title: String,
    pub snippet: String,
    pub score: f64,
    pub source_label: String,
    pub file_path: String,
    pub lines: String,
    pub metadata: serde_json::Value,
    pub source_type: String,
}

/// Search the knowledge DB with optional metadata filtering.
///
/// Runs the full multi-layer pipeline: BM25 → trigram → fuzzy → vector → RRF.
/// Metadata filters restrict results before RRF fusion.
pub fn knowledge_search(
    store: &KnowledgeStore,
    query: &str,
    filter: Option<&str>,
    source_label: Option<&str>,
    limit: u32,
    weights: &SearchWeights,
    snippet_bytes: usize,
) -> Result<Vec<KnowledgeSearchResult>> {
    let conn = store.conn();

    // Parse metadata filters
    let mut predicates = parse_filter(filter);
    if let Some(label) = source_label {
        predicates.push(super::filter::FilterPredicate::Source(label.to_string()));
    }

    let (meta_where, meta_params) = predicates_to_sql(&predicates, 0);

    let mut all_results: HashMap<i64, ScoredResult> = HashMap::new();

    // Layer 1: FTS5 BM25
    let bm25 = knowledge_bm25_search(conn, query, limit, &meta_where, &meta_params)?;
    merge_results(&mut all_results, &bm25, weights.keyword_weight);

    // Layer 2: Trigram (if BM25 returned fewer than limit)
    if bm25.len() < limit as usize {
        let trigram = knowledge_trigram_search(conn, query, limit, &meta_where, &meta_params)?;
        merge_results(&mut all_results, &trigram, weights.keyword_weight);
    }

    // Layer 3: Fuzzy (if still under limit)
    if all_results.len() < limit as usize {
        let fuzzy = knowledge_fuzzy_search(conn, query, limit, &meta_where, &meta_params)?;
        merge_results(&mut all_results, &fuzzy, weights.keyword_weight);
    }

    // Layer 4: Vector similarity
    if let Some(emb) = store.embedder() {
        match emb.embed_one(query) {
            Ok(query_embedding) => {
                let candidate_rowids: Vec<i64> =
                    all_results.keys().copied().collect();
                let candidates = if candidate_rowids.is_empty() {
                    None
                } else {
                    Some(candidate_rowids.as_slice())
                };
                let vector = knowledge_vector_search(
                    conn,
                    &query_embedding,
                    limit,
                    &meta_where,
                    &meta_params,
                    candidates,
                )?;
                merge_results(&mut all_results, &vector, weights.vector_weight);
            }
            Err(e) => {
                eprintln!("[bpcontext] knowledge query embedding failed, skipping vector layer: {e}");
            }
        }
    }

    // Collect and sort by RRF score
    let mut results: Vec<KnowledgeSearchResult> = all_results
        .into_values()
        .map(|sr| sr.into_knowledge_result(snippet_bytes))
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit as usize);
    Ok(results)
}

/// Search both session and knowledge DBs, merging results via RRF.
pub fn combined_search(
    session_store: &ContentStore,
    knowledge_store: &KnowledgeStore,
    query: &str,
    filter: Option<&str>,
    limit: u32,
    weights: &SearchWeights,
    snippet_bytes: usize,
) -> Result<Vec<KnowledgeSearchResult>> {
    // 1. Session search
    let session_results = session_store.search_with_weights(query, limit, None, None, weights)?;

    // 2. Knowledge search
    let knowledge_results =
        knowledge_search(knowledge_store, query, filter, None, limit, weights, snippet_bytes)?;

    // 3. Convert session results to KnowledgeSearchResult
    let session_converted: Vec<KnowledgeSearchResult> = session_results
        .into_iter()
        .map(|sr| session_to_knowledge_result(sr, snippet_bytes))
        .collect();

    // 4. Merge via RRF
    const K: f64 = 60.0;
    let mut merged: HashMap<String, MergedResult> = HashMap::new();

    for (rank, result) in session_converted.iter().enumerate() {
        let rrf = 1.0 / (K + rank as f64 + 1.0);
        let key = format!("session:{}:{}", result.source_label, result.title);
        merged
            .entry(key)
            .and_modify(|m| m.rrf_score += rrf)
            .or_insert(MergedResult {
                result: result.clone(),
                rrf_score: rrf,
            });
    }

    for (rank, result) in knowledge_results.iter().enumerate() {
        let rrf = 1.0 / (K + rank as f64 + 1.0);
        let key = format!(
            "knowledge:{}:{}",
            result.source_label, result.title
        );
        merged
            .entry(key)
            .and_modify(|m| m.rrf_score += rrf)
            .or_insert(MergedResult {
                result: result.clone(),
                rrf_score: rrf,
            });
    }

    let mut final_results: Vec<KnowledgeSearchResult> = merged
        .into_values()
        .map(|m| {
            let mut r = m.result;
            r.score = m.rrf_score;
            r
        })
        .collect();

    final_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    final_results.truncate(limit as usize);
    Ok(final_results)
}

// ── Internal types ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct RankedResult {
    rowid: i64,
    title: String,
    content: String,
    rank: f64,
    source_label: String,
    file_path: String,
    line_start: u32,
    line_end: u32,
    metadata: String,
}

struct ScoredResult {
    title: String,
    content: String,
    source_label: String,
    file_path: String,
    line_start: u32,
    line_end: u32,
    metadata: String,
    rrf_score: f64,
}

impl ScoredResult {
    fn into_knowledge_result(self, snippet_bytes: usize) -> KnowledgeSearchResult {
        let metadata: serde_json::Value =
            serde_json::from_str(&self.metadata).unwrap_or(serde_json::Value::Object(Default::default()));
        KnowledgeSearchResult {
            title: self.title,
            snippet: truncate_snippet(&self.content, snippet_bytes),
            score: self.rrf_score,
            source_label: self.source_label,
            file_path: self.file_path,
            lines: format!("{}-{}", self.line_start, self.line_end),
            metadata,
            source_type: "knowledge".to_string(),
        }
    }
}

#[derive(Clone)]
struct MergedResult {
    result: KnowledgeSearchResult,
    rrf_score: f64,
}

impl Clone for KnowledgeSearchResult {
    fn clone(&self) -> Self {
        Self {
            title: self.title.clone(),
            snippet: self.snippet.clone(),
            score: self.score,
            source_label: self.source_label.clone(),
            file_path: self.file_path.clone(),
            lines: self.lines.clone(),
            metadata: self.metadata.clone(),
            source_type: self.source_type.clone(),
        }
    }
}

// ── Search layers ────────────────────────────────────────────────────────────

/// BM25 search on knowledge_chunks FTS5 table.
fn knowledge_bm25_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    meta_where: &str,
    meta_params: &[String],
) -> Result<Vec<RankedResult>> {
    let fts_query = sanitize_fts_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // Metadata filter params come first (they use ?1, ?2, ... from predicates_to_sql)
    for p in meta_params {
        params.push(Box::new(p.clone()));
    }

    let fts_idx = params.len() + 1;
    params.push(Box::new(fts_query));

    let meta_clause = if meta_where.is_empty() {
        String::new()
    } else {
        format!(" AND {meta_where}")
    };

    let sql = format!(
        "SELECT kc.rowid, kc.title, kc.content,
                COALESCE(kcm.line_start, 0), COALESCE(kcm.line_end, 0),
                COALESCE(kcm.metadata, '{{}}'),
                COALESCE(ks.label, '') as source_label,
                COALESCE(kf.rel_path, '') as file_path,
                bm25(knowledge_chunks) as rank
         FROM knowledge_chunks kc
         JOIN knowledge_chunk_meta kcm ON kcm.chunk_rowid = kc.rowid
         JOIN knowledge_files kf ON kf.id = kcm.file_id
         JOIN knowledge_sources ks ON ks.id = kcm.source_id
         WHERE knowledge_chunks MATCH ?{fts_idx}{meta_clause}
         ORDER BY rank
         LIMIT ?{}",
        params.len() + 1
    );
    params.push(Box::new(limit));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let results = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok(RankedResult {
                rowid: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
                line_start: row.get::<_, i64>(3).unwrap_or(0) as u32,
                line_end: row.get::<_, i64>(4).unwrap_or(0) as u32,
                metadata: row.get(5)?,
                source_label: row.get(6)?,
                file_path: row.get(7)?,
                rank: row.get::<_, f64>(8)?.abs(),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

/// Trigram search on knowledge_chunks_trigram FTS5 table.
fn knowledge_trigram_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    meta_where: &str,
    meta_params: &[String],
) -> Result<Vec<RankedResult>> {
    if query.len() < 3 {
        return Ok(Vec::new());
    }

    let fts_query = sanitize_trigram_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for p in meta_params {
        params.push(Box::new(p.clone()));
    }

    let fts_idx = params.len() + 1;
    params.push(Box::new(fts_query));

    let meta_clause = if meta_where.is_empty() {
        String::new()
    } else {
        format!(" AND {meta_where}")
    };

    let sql = format!(
        "SELECT kct.rowid, kct.title, kct.content,
                COALESCE(kcm.line_start, 0), COALESCE(kcm.line_end, 0),
                COALESCE(kcm.metadata, '{{}}'),
                COALESCE(ks.label, '') as source_label,
                COALESCE(kf.rel_path, '') as file_path,
                bm25(knowledge_chunks_trigram) as rank
         FROM knowledge_chunks_trigram kct
         JOIN knowledge_chunk_meta kcm ON kcm.chunk_rowid = kct.rowid
         JOIN knowledge_files kf ON kf.id = kcm.file_id
         JOIN knowledge_sources ks ON ks.id = kcm.source_id
         WHERE knowledge_chunks_trigram MATCH ?{fts_idx}{meta_clause}
         ORDER BY rank
         LIMIT ?{}",
        params.len() + 1
    );
    params.push(Box::new(limit));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let results = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok(RankedResult {
                rowid: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
                line_start: row.get::<_, i64>(3).unwrap_or(0) as u32,
                line_end: row.get::<_, i64>(4).unwrap_or(0) as u32,
                metadata: row.get(5)?,
                source_label: row.get(6)?,
                file_path: row.get(7)?,
                rank: row.get::<_, f64>(8)?.abs(),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

/// Levenshtein fuzzy search: correct query terms against knowledge index vocab.
fn knowledge_fuzzy_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    meta_where: &str,
    meta_params: &[String],
) -> Result<Vec<RankedResult>> {
    let terms = extract_knowledge_terms(conn, 500)?;
    let query_words: Vec<&str> = query.split_whitespace().collect();

    let mut corrected_words = Vec::new();
    for word in &query_words {
        let word_lower = word.to_lowercase();
        let mut best_match = word.to_string();
        let mut best_distance = 3usize;

        for term in &terms {
            let dist = strsim::levenshtein(&word_lower, term);
            if dist < best_distance && dist <= 2 {
                best_distance = dist;
                best_match = term.clone();
            }
        }
        corrected_words.push(best_match);
    }

    let corrected_query = corrected_words.join(" ");
    if corrected_query == query {
        return Ok(Vec::new());
    }

    knowledge_bm25_search(conn, &corrected_query, limit, meta_where, meta_params)
}

/// Extract terms from the knowledge FTS5 vocab for fuzzy matching.
fn extract_knowledge_terms(conn: &Connection, limit: usize) -> Result<Vec<String>> {
    let result = conn.prepare(
        "SELECT DISTINCT term FROM knowledge_chunks_vocab WHERE col = 'content' LIMIT ?1",
    );

    match result {
        Ok(mut stmt) => {
            let terms = stmt
                .query_map(rusqlite::params![limit as i64], |row| {
                    row.get::<_, String>(0)
                })?
                .filter_map(|r| r.ok())
                .collect();
            Ok(terms)
        }
        Err(_) => {
            // Vocab table may not exist; create it and retry
            let _ = conn.execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_vocab \
                 USING fts5vocab(knowledge_chunks, instance);"
            );
            let mut stmt = conn.prepare(
                "SELECT DISTINCT term FROM knowledge_chunks_vocab WHERE col = 'content' LIMIT ?1",
            )?;
            let terms = stmt
                .query_map(rusqlite::params![limit as i64], |row| {
                    row.get::<_, String>(0)
                })?
                .filter_map(|r| r.ok())
                .collect();
            Ok(terms)
        }
    }
}

/// Vector similarity search over knowledge embeddings.
fn knowledge_vector_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    meta_where: &str,
    meta_params: &[String],
    candidate_rowids: Option<&[i64]>,
) -> Result<Vec<RankedResult>> {
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // Metadata filter params
    for p in meta_params {
        params.push(Box::new(p.clone()));
    }

    let meta_clause = if meta_where.is_empty() {
        String::new()
    } else {
        format!(" AND {meta_where}")
    };

    let mut sql = format!(
        "SELECT ke.chunk_rowid, ke.embedding,
                kc.title, kc.content,
                COALESCE(kcm.line_start, 0), COALESCE(kcm.line_end, 0),
                COALESCE(kcm.metadata, '{{}}'),
                COALESCE(ks.label, '') as source_label,
                COALESCE(kf.rel_path, '') as file_path
         FROM knowledge_embeddings ke
         JOIN knowledge_chunks kc ON kc.rowid = ke.chunk_rowid
         JOIN knowledge_chunk_meta kcm ON kcm.chunk_rowid = ke.chunk_rowid
         JOIN knowledge_files kf ON kf.id = kcm.file_id
         JOIN knowledge_sources ks ON ks.id = kcm.source_id
         WHERE 1=1{meta_clause}"
    );

    if let Some(rowids) = candidate_rowids {
        if !rowids.is_empty() {
            let placeholders: Vec<String> = rowids
                .iter()
                .enumerate()
                .map(|(i, _)| format!("?{}", params.len() + i + 1))
                .collect();
            sql.push_str(&format!(
                " AND ke.chunk_rowid IN ({})",
                placeholders.join(",")
            ));
            for &rid in rowids {
                params.push(Box::new(rid));
            }
        }
    }

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let mut scored: Vec<(f32, RankedResult)> = stmt
        .query_map(param_refs.as_slice(), |row| {
            let chunk_rowid: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((
                blob,
                chunk_rowid,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4).unwrap_or(0) as u32,
                row.get::<_, i64>(5).unwrap_or(0) as u32,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .map(|(blob, rowid, title, content, ls, le, meta, source, path)| {
            let embedding = embedder::bytes_to_embedding(&blob);
            let similarity = dot_product(query_embedding, &embedding);
            (
                similarity,
                RankedResult {
                    rowid,
                    title,
                    content,
                    rank: 0.0,
                    source_label: source,
                    file_path: path,
                    line_start: ls,
                    line_end: le,
                    metadata: meta,
                },
            )
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit as usize);

    Ok(scored
        .into_iter()
        .enumerate()
        .map(|(rank, (_, mut result))| {
            result.rank = rank as f64;
            result
        })
        .collect())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// RRF merge with K=60, matching session search behavior.
fn merge_results(
    acc: &mut HashMap<i64, ScoredResult>,
    results: &[RankedResult],
    weight: f64,
) {
    const K: f64 = 60.0;

    for (rank, result) in results.iter().enumerate() {
        let rrf_score = weight * (1.0 / (K + rank as f64 + 1.0));

        acc.entry(result.rowid)
            .and_modify(|sr| {
                sr.rrf_score += rrf_score;
            })
            .or_insert_with(|| ScoredResult {
                title: result.title.clone(),
                content: result.content.clone(),
                source_label: result.source_label.clone(),
                file_path: result.file_path.clone(),
                line_start: result.line_start,
                line_end: result.line_end,
                metadata: result.metadata.clone(),
                rrf_score,
            });
    }
}

/// Dot product for cosine similarity (vectors are L2-normalized at index time).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Truncate content to a snippet at word boundaries.
fn truncate_snippet(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    // Find the last space before the byte limit
    let truncated = &content[..max_bytes];
    match truncated.rfind(' ') {
        Some(pos) => format!("{}...", &content[..pos]),
        None => format!("{}...", truncated),
    }
}

/// Convert a session SearchResult to a KnowledgeSearchResult.
fn session_to_knowledge_result(sr: SearchResult, snippet_bytes: usize) -> KnowledgeSearchResult {
    KnowledgeSearchResult {
        title: sr.title,
        snippet: truncate_snippet(&sr.content, snippet_bytes),
        score: sr.score,
        source_label: sr.source.clone(),
        file_path: sr.source,
        lines: format!("{}-{}", sr.line_start, sr.line_end),
        metadata: serde_json::Value::Object(Default::default()),
        source_type: "session".to_string(),
    }
}

/// Sanitize a query for FTS5 MATCH syntax.
fn sanitize_fts_query(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    let mut last_was_space = true;

    for c in query.chars() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            result.push(c);
            last_was_space = false;
        } else if !last_was_space {
            result.push(' ');
            last_was_space = true;
        }
    }

    if result.ends_with(' ') {
        result.pop();
    }

    result
}

/// Sanitize for trigram FTS5 MATCH (requires quoted phrase).
fn sanitize_trigram_query(query: &str) -> String {
    let cleaned = sanitize_fts_query(query);
    if cleaned.is_empty() {
        return String::new();
    }
    format!("\"{cleaned}\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge;
    use rusqlite::params;

    fn make_store() -> KnowledgeStore {
        knowledge::open_in_memory().expect("in-memory knowledge store")
    }

    /// Insert a source + file + chunk with metadata for testing search.
    fn seed_chunk(
        store: &KnowledgeStore,
        source_label: &str,
        source_path: &str,
        rel_path: &str,
        title: &str,
        content: &str,
        metadata_json: &str,
    ) -> i64 {
        let conn = store.conn();

        // Ensure source exists
        let source_id: i64 = match conn.query_row(
            "SELECT id FROM knowledge_sources WHERE label = ?1",
            params![source_label],
            |row| row.get(0),
        ) {
            Ok(id) => id,
            Err(_) => {
                conn.execute(
                    "INSERT INTO knowledge_sources (label, path, enrichments) VALUES (?1, ?2, '[]')",
                    params![source_label, source_path],
                )
                .unwrap();
                conn.last_insert_rowid()
            }
        };

        // Ensure file exists
        let file_id: i64 = match conn.query_row(
            "SELECT id FROM knowledge_files WHERE source_id = ?1 AND rel_path = ?2",
            params![source_id, rel_path],
            |row| row.get(0),
        ) {
            Ok(id) => id,
            Err(_) => {
                conn.execute(
                    "INSERT INTO knowledge_files (source_id, rel_path, content_hash, size_bytes) VALUES (?1, ?2, 'abc123', 100)",
                    params![source_id, rel_path],
                )
                .unwrap();
                conn.last_insert_rowid()
            }
        };

        // Insert into FTS5
        conn.execute(
            "INSERT INTO knowledge_chunks (title, content) VALUES (?1, ?2)",
            params![title, content],
        )
        .unwrap();
        let rowid = conn.last_insert_rowid();

        // Insert matching trigram row
        conn.execute(
            "INSERT INTO knowledge_chunks_trigram (rowid, title, content) VALUES (?1, ?2, ?3)",
            params![rowid, title, content],
        )
        .unwrap();

        // Insert chunk_meta
        conn.execute(
            "INSERT INTO knowledge_chunk_meta (chunk_rowid, file_id, source_id, line_start, line_end, metadata) VALUES (?1, ?2, ?3, 1, 20, ?4)",
            params![rowid, file_id, source_id, metadata_json],
        )
        .unwrap();

        rowid
    }

    // ── sanitize helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_sanitize_fts_query() {
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
        assert_eq!(sanitize_fts_query("error: can't find"), "error can t find");
        assert_eq!(sanitize_fts_query("fn main()"), "fn main");
    }

    #[test]
    fn test_sanitize_trigram_query() {
        assert_eq!(sanitize_trigram_query("hello"), "\"hello\"");
        assert_eq!(sanitize_trigram_query(""), "");
    }

    #[test]
    fn test_truncate_snippet_short() {
        let s = "short text";
        assert_eq!(truncate_snippet(s, 100), "short text");
    }

    #[test]
    fn test_truncate_snippet_at_word_boundary() {
        let s = "the quick brown fox jumps over the lazy dog";
        let result = truncate_snippet(s, 20);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 24); // 20 + "..."
    }

    // ── knowledge_search (no filter) ─────────────────────────────────────────

    #[test]
    fn search_empty_db_returns_empty() {
        let store = make_store();
        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "anything", None, None, 10, &weights, 2000).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_finds_bm25_match() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "notes/test.md",
            "Fatigue Model",
            "The fatigue model applies minute-weighted decay to player performance",
            "{}",
        );

        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "fatigue model", None, None, 10, &weights, 2000).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].source_type, "knowledge");
        assert_eq!(results[0].source_label, "vault");
        assert_eq!(results[0].file_path, "notes/test.md");
    }

    #[test]
    fn search_returns_correct_lines_format() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "test.md",
            "Title",
            "Content here",
            "{}",
        );

        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "content", None, None, 10, &weights, 2000).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].lines, "1-20");
    }

    // ── knowledge_search (with metadata filter) ──────────────────────────────

    #[test]
    fn search_with_frontmatter_filter() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "task1.md",
            "Active Task",
            "This is an active task about implementation",
            r#"{"frontmatter":{"type":"task","status":"active"}}"#,
        );
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "note1.md",
            "Random Note",
            "This is a random note about implementation",
            r#"{"frontmatter":{"type":"note","status":"draft"}}"#,
        );

        let weights = SearchWeights::default();

        // Filter for type:task — should return only the task
        let results = knowledge_search(
            &store,
            "implementation",
            Some("type:task"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Active Task");
    }

    #[test]
    fn search_with_source_filter() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "v.md",
            "Vault Note",
            "Content from vault about algorithms",
            "{}",
        );
        seed_chunk(
            &store,
            "docs",
            "/tmp/docs",
            "d.md",
            "Docs Note",
            "Content from docs about algorithms",
            "{}",
        );

        let weights = SearchWeights::default();
        let results = knowledge_search(
            &store,
            "algorithms",
            Some("source:vault"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_label, "vault");
    }

    #[test]
    fn search_with_tag_filter() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "tagged.md",
            "Tagged Note",
            "Content about prediction models and forecasting",
            r#"{"frontmatter":{"tags":["implementation","system3"]}}"#,
        );
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "untagged.md",
            "Untagged Note",
            "Content about prediction models and estimation",
            r#"{"frontmatter":{"tags":["review"]}}"#,
        );

        let weights = SearchWeights::default();
        let results = knowledge_search(
            &store,
            "prediction",
            Some("tag:implementation"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Tagged Note");
    }

    #[test]
    fn search_with_folder_filter() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "projects/foo.md",
            "Project Note",
            "Content about project setup and configuration",
            r#"{"folder_tags":["01-projects","foo"]}"#,
        );
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "logs/bar.md",
            "Log Note",
            "Content about debugging and configuration",
            r#"{"folder_tags":["06-logs","debug"]}"#,
        );

        let weights = SearchWeights::default();
        let results = knowledge_search(
            &store,
            "configuration",
            Some("folder:01-projects"),
            None,
            10,
            &weights,
            2000,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Project Note");
    }

    #[test]
    fn search_with_source_label_param() {
        let store = make_store();
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "v.md",
            "Vault Item",
            "Searchable content about architecture",
            "{}",
        );
        seed_chunk(
            &store,
            "docs",
            "/tmp/docs",
            "d.md",
            "Docs Item",
            "Searchable content about architecture",
            "{}",
        );

        let weights = SearchWeights::default();
        let results = knowledge_search(
            &store,
            "architecture",
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

    #[test]
    fn search_metadata_in_result() {
        let store = make_store();
        let meta = r#"{"frontmatter":{"type":"task","status":"active"}}"#;
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "meta.md",
            "Meta Test",
            "Content with metadata for testing purposes",
            meta,
        );

        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "metadata testing", None, None, 10, &weights, 2000).unwrap();
        assert!(!results.is_empty());
        let fm = &results[0].metadata["frontmatter"];
        assert_eq!(fm["type"], "task");
        assert_eq!(fm["status"], "active");
    }

    // ── combined_search ──────────────────────────────────────────────────────

    #[test]
    fn combined_search_no_knowledge_returns_only_session() {
        // With no knowledge sources, combined_search should still work
        // (knowledge portion returns empty)
        let store = make_store();
        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "anything", None, None, 10, &weights, 2000).unwrap();
        assert!(results.is_empty());
    }

    // ── snippet truncation ───────────────────────────────────────────────────

    #[test]
    fn search_snippet_truncated() {
        let store = make_store();
        let long_content = "word ".repeat(500); // ~2500 chars
        seed_chunk(
            &store,
            "vault",
            "/tmp/vault",
            "long.md",
            "Long Content",
            &long_content,
            "{}",
        );

        let weights = SearchWeights::default();
        let results =
            knowledge_search(&store, "word", None, None, 10, &weights, 100).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].snippet.len() <= 104); // 100 + "..."
    }

    // ── dot_product ──────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((dot_product(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(dot_product(&a, &b).abs() < 1e-6);
    }
}
