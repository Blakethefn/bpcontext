use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

use crate::embedder::{self, Embed};

/// A search result with relevance metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SearchResult {
    pub title: String,
    pub content: String,
    pub content_type: String,
    pub source_id: i64,
    pub score: f64,
    pub source: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// Weight configuration for RRF layer blending
pub struct SearchWeights {
    pub keyword_weight: f64,
    pub vector_weight: f64,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            keyword_weight: 1.0,
            vector_weight: 1.0,
        }
    }
}

/// Multi-layer search: BM25 → trigram → Levenshtein → vector → RRF fusion
pub fn multi_layer_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    source_id_filter: Option<i64>,
    type_filter: Option<&str>,
    embedder: Option<&dyn Embed>,
    weights: &SearchWeights,
) -> Result<Vec<SearchResult>> {
    let mut all_results: HashMap<String, ScoredResult> = HashMap::new();

    // Layer 1: FTS5 BM25 with Porter stemming
    let bm25_results = fts5_bm25_search(
        conn,
        query,
        limit,
        source_filter,
        source_id_filter,
        type_filter,
    )?;
    merge_results(&mut all_results, &bm25_results, weights.keyword_weight);

    // Layer 2: Trigram search (if BM25 returned fewer than limit)
    if bm25_results.len() < limit as usize {
        let trigram_results = trigram_search(
            conn,
            query,
            limit,
            source_filter,
            source_id_filter,
            type_filter,
        )?;
        merge_results(&mut all_results, &trigram_results, weights.keyword_weight);
    }

    // Layer 3: Levenshtein fuzzy correction (if still too few results)
    if all_results.len() < limit as usize {
        let fuzzy_results = fuzzy_search(
            conn,
            query,
            limit,
            source_filter,
            source_id_filter,
            type_filter,
        )?;
        merge_results(&mut all_results, &fuzzy_results, weights.keyword_weight);
    }

    // Layer 4: Vector similarity (always runs if embedder available)
    if let Some(emb) = embedder {
        match emb.embed_one(query) {
            Ok(query_embedding) => {
                let vector_results = vector_search(
                    conn,
                    &query_embedding,
                    limit,
                    source_filter,
                    source_id_filter,
                    type_filter,
                )?;
                merge_results(&mut all_results, &vector_results, weights.vector_weight);
            }
            Err(e) => {
                eprintln!("[bpcontext] query embedding failed, skipping vector layer: {e}");
            }
        }
    }

    // RRF fusion scoring + proximity boost
    let mut results: Vec<SearchResult> = all_results
        .into_values()
        .map(|sr| sr.into_search_result())
        .collect();

    // Apply proximity boost for multi-word queries
    if query.split_whitespace().count() > 1 {
        apply_proximity_boost(&mut results, query);
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit as usize);
    Ok(results)
}

/// FTS5 BM25 search on the primary chunks table
fn fts5_bm25_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    source_id_filter: Option<i64>,
    type_filter: Option<&str>,
) -> Result<Vec<RankedResult>> {
    let fts_query = sanitize_fts_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut sql = String::from(
        "SELECT chunks.title, chunks.content, chunks.content_type, chunks.source_id,
                bm25(chunks) as rank, COALESCE(s.label, '') as source_label,
                chunks.line_start, chunks.line_end
         FROM chunks
         LEFT JOIN sources s ON chunks.source_id = s.id
         WHERE chunks MATCH ?1",
    );

    if let Some(tf) = type_filter {
        sql.push_str(&format!(" AND chunks.content_type = '{}'", escape_sql(tf)));
    }
    if let Some(sf) = source_filter {
        sql.push_str(&format!(" AND s.label LIKE '%{}%'", escape_sql(sf)));
    }
    if let Some(sid) = source_id_filter {
        sql.push_str(&format!(" AND chunks.source_id = {sid}"));
    }

    sql.push_str(&format!(" ORDER BY rank LIMIT {limit}"));

    let mut stmt = conn.prepare(&sql)?;
    let results = stmt
        .query_map(rusqlite::params![fts_query], |row| {
            Ok(RankedResult {
                title: row.get(0)?,
                content: row.get(1)?,
                content_type: row.get(2)?,
                source_id: row.get(3)?,
                rank: row.get::<_, f64>(4)?.abs(),
                source_label: row.get(5)?,
                line_start: row.get::<_, i64>(6).unwrap_or(0) as u32,
                line_end: row.get::<_, i64>(7).unwrap_or(0) as u32,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

/// Trigram search on the chunks_trigram table
fn trigram_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    source_id_filter: Option<i64>,
    type_filter: Option<&str>,
) -> Result<Vec<RankedResult>> {
    // Trigram requires at least 3 characters
    if query.len() < 3 {
        return Ok(Vec::new());
    }

    let fts_query = sanitize_trigram_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut sql = String::from(
        "SELECT chunks_trigram.title, chunks_trigram.content, chunks_trigram.content_type,
                chunks_trigram.source_id, bm25(chunks_trigram) as rank,
                COALESCE(s.label, '') as source_label,
                chunks_trigram.line_start, chunks_trigram.line_end
         FROM chunks_trigram
         LEFT JOIN sources s ON chunks_trigram.source_id = s.id
         WHERE chunks_trigram MATCH ?1",
    );

    if let Some(tf) = type_filter {
        sql.push_str(&format!(
            " AND chunks_trigram.content_type = '{}'",
            escape_sql(tf)
        ));
    }
    if let Some(sf) = source_filter {
        sql.push_str(&format!(" AND s.label LIKE '%{}%'", escape_sql(sf)));
    }
    if let Some(sid) = source_id_filter {
        sql.push_str(&format!(" AND chunks_trigram.source_id = {sid}"));
    }

    sql.push_str(&format!(" ORDER BY rank LIMIT {limit}"));

    let mut stmt = conn.prepare(&sql)?;
    let results = stmt
        .query_map(rusqlite::params![fts_query], |row| {
            Ok(RankedResult {
                title: row.get(0)?,
                content: row.get(1)?,
                content_type: row.get(2)?,
                source_id: row.get(3)?,
                rank: row.get::<_, f64>(4)?.abs(),
                source_label: row.get(5)?,
                line_start: row.get::<_, i64>(6).unwrap_or(0) as u32,
                line_end: row.get::<_, i64>(7).unwrap_or(0) as u32,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

/// Levenshtein fuzzy search: correct query terms and re-search
fn fuzzy_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    source_id_filter: Option<i64>,
    type_filter: Option<&str>,
) -> Result<Vec<RankedResult>> {
    // Extract unique terms from the index to find close matches
    let terms: Vec<String> = extract_index_terms(conn, 500)?;
    let query_words: Vec<&str> = query.split_whitespace().collect();

    let mut corrected_words = Vec::new();
    for word in &query_words {
        let word_lower = word.to_lowercase();
        let mut best_match = word.to_string();
        let mut best_distance = 3usize; // max edit distance

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
        return Ok(Vec::new()); // No corrections found
    }

    fts5_bm25_search(
        conn,
        &corrected_query,
        limit,
        source_filter,
        source_id_filter,
        type_filter,
    )
}

/// Extract frequent terms from the FTS5 index for fuzzy matching
fn extract_index_terms(conn: &Connection, limit: usize) -> Result<Vec<String>> {
    // Use FTS5 vocab table to get terms
    let result =
        conn.prepare("SELECT DISTINCT term FROM chunks_vocab WHERE col = 'content' LIMIT ?1");

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
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks, instance);"
            );
            let mut stmt = conn
                .prepare("SELECT DISTINCT term FROM chunks_vocab WHERE col = 'content' LIMIT ?1")?;
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

/// Brute-force vector similarity search over stored embeddings.
///
/// Computes dot-product similarity (embeddings are pre-normalized at index time,
/// so dot product equals cosine similarity). Returns results ranked by similarity.
fn vector_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    source_filter: Option<&str>,
    source_id_filter: Option<i64>,
    type_filter: Option<&str>,
) -> Result<Vec<RankedResult>> {
    // Build query to load embeddings with chunk metadata
    let mut sql = String::from(
        "SELECT ce.chunk_rowid, ce.embedding,
                c.title, c.content, c.content_type, c.source_id,
                COALESCE(s.label, '') as source_label,
                c.line_start, c.line_end
         FROM chunk_embeddings ce
         JOIN chunks c ON c.rowid = ce.chunk_rowid
         LEFT JOIN sources s ON c.source_id = s.id
         WHERE 1=1",
    );

    if let Some(tf) = type_filter {
        sql.push_str(&format!(" AND c.content_type = '{}'", escape_sql(tf)));
    }
    if let Some(sf) = source_filter {
        sql.push_str(&format!(" AND s.label LIKE '%{}%'", escape_sql(sf)));
    }
    if let Some(sid) = source_id_filter {
        sql.push_str(&format!(" AND c.source_id = {sid}"));
    }

    let mut stmt = conn.prepare(&sql)?;
    let mut scored: Vec<(f32, RankedResult)> = stmt
        .query_map([], |row| {
            let blob: Vec<u8> = row.get(1)?;
            Ok((
                blob,
                RankedResult {
                    title: row.get(2)?,
                    content: row.get(3)?,
                    content_type: row.get(4)?,
                    source_id: row.get(5)?,
                    rank: 0.0, // will be set after sorting
                    source_label: row.get(6)?,
                    line_start: row.get::<_, i64>(7).unwrap_or(0) as u32,
                    line_end: row.get::<_, i64>(8).unwrap_or(0) as u32,
                },
            ))
        })?
        .filter_map(|r| r.ok())
        .map(|(blob, result)| {
            let embedding = embedder::bytes_to_embedding(&blob);
            let similarity = dot_product(query_embedding, &embedding);
            (similarity, result)
        })
        .collect();

    // Sort by similarity descending
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit as usize);

    // Convert to RankedResult with rank set for RRF
    Ok(scored
        .into_iter()
        .enumerate()
        .map(|(rank, (_, mut result))| {
            result.rank = rank as f64;
            result
        })
        .collect())
}

/// Dot product of two vectors. With L2-normalized vectors, this equals cosine similarity.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Merge ranked results into the accumulator using Reciprocal Rank Fusion.
/// The `weight` multiplier scales the RRF contribution of this layer.
fn merge_results(acc: &mut HashMap<String, ScoredResult>, results: &[RankedResult], weight: f64) {
    const K: f64 = 60.0;

    for (rank, result) in results.iter().enumerate() {
        let rrf_score = weight * (1.0 / (K + rank as f64 + 1.0));
        let key = format!("{}:{}", result.source_id, result.title);

        acc.entry(key)
            .and_modify(|sr| sr.rrf_score += rrf_score)
            .or_insert_with(|| ScoredResult {
                title: result.title.clone(),
                content: result.content.clone(),
                content_type: result.content_type.clone(),
                source_id: result.source_id,
                source_label: result.source_label.clone(),
                rrf_score,
                line_start: result.line_start,
                line_end: result.line_end,
            });
    }
}

/// Apply proximity boost for multi-word queries
fn apply_proximity_boost(results: &mut [SearchResult], query: &str) {
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.len() < 2 {
        return;
    }

    for result in results.iter_mut() {
        let content_lower = result.content.to_lowercase();
        // Check if all query words appear within a 200-char window
        let mut all_found = true;
        let mut positions: Vec<usize> = Vec::new();

        for word in &words {
            if let Some(pos) = content_lower.find(&word.to_lowercase()) {
                positions.push(pos);
            } else {
                all_found = false;
                break;
            }
        }

        if all_found && positions.len() >= 2 {
            let span = positions.iter().max().unwrap() - positions.iter().min().unwrap();
            if span < 200 {
                result.score *= 1.5; // 50% boost for proximity
            } else if span < 500 {
                result.score *= 1.2; // 20% boost for moderate proximity
            }
        }
    }
}

/// Sanitize a query for FTS5 MATCH syntax
fn sanitize_fts_query(query: &str) -> String {
    // Remove FTS5 special characters, keep words
    query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '_' || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Sanitize a query for trigram FTS5 MATCH
fn sanitize_trigram_query(query: &str) -> String {
    // Trigram tokenizer expects quoted phrases for substring matching
    let cleaned = sanitize_fts_query(query);
    if cleaned.is_empty() {
        return String::new();
    }
    format!("\"{cleaned}\"")
}

/// Escape single quotes for SQL string interpolation
fn escape_sql(s: &str) -> String {
    s.replace('\'', "''")
}

#[derive(Debug)]
#[allow(dead_code)]
struct RankedResult {
    title: String,
    content: String,
    content_type: String,
    source_id: i64,
    rank: f64,
    source_label: String,
    line_start: u32,
    line_end: u32,
}

struct ScoredResult {
    title: String,
    content: String,
    content_type: String,
    source_id: i64,
    source_label: String,
    rrf_score: f64,
    line_start: u32,
    line_end: u32,
}

impl ScoredResult {
    fn into_search_result(self) -> SearchResult {
        SearchResult {
            title: self.title,
            content: self.content,
            content_type: self.content_type,
            source_id: self.source_id,
            score: self.rrf_score,
            source: self.source_label,
            line_start: self.line_start,
            line_end: self.line_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_fts_query() {
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
        assert_eq!(sanitize_fts_query("error: can't find"), "error can t find");
        assert_eq!(sanitize_fts_query("fn main()"), "fn main");
    }

    #[test]
    fn test_sanitize_trigram_query() {
        assert_eq!(sanitize_trigram_query("hello"), "\"hello\"");
    }

    #[test]
    fn test_dot_product_identical_unit_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((dot_product(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((dot_product(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_normalized() {
        // Two similar-ish vectors, both L2-normalized
        let mut a = vec![3.0f32, 4.0];
        let norm_a = (a[0] * a[0] + a[1] * a[1]).sqrt();
        a.iter_mut().for_each(|v| *v /= norm_a);

        let mut b = vec![4.0f32, 3.0];
        let norm_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
        b.iter_mut().for_each(|v| *v /= norm_b);

        let sim = dot_product(&a, &b);
        // cos(angle between [3,4] and [4,3]) = 24/25 = 0.96
        assert!((sim - 0.96).abs() < 1e-4);
    }

    #[test]
    fn test_merge_results_with_weight() {
        let mut acc: HashMap<String, ScoredResult> = HashMap::new();
        let results = vec![RankedResult {
            title: "test".to_string(),
            content: "content".to_string(),
            content_type: "prose".to_string(),
            source_id: 1,
            rank: 0.0,
            source_label: "src".to_string(),
            line_start: 0,
            line_end: 0,
        }];

        // Weight 1.0: rank 0 → 1/(60+0+1) = 0.01639...
        merge_results(&mut acc, &results, 1.0);
        let score_w1 = acc.values().next().unwrap().rrf_score;

        let mut acc2: HashMap<String, ScoredResult> = HashMap::new();
        merge_results(&mut acc2, &results, 2.0);
        let score_w2 = acc2.values().next().unwrap().rrf_score;

        assert!((score_w2 - 2.0 * score_w1).abs() < 1e-10);
    }

    /// Helper: set up an in-memory DB with schema, insert chunks + embeddings
    fn setup_vector_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        super::super::schema::init_content_schema(&conn).unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO sources (label, indexed_at, chunk_count) VALUES (?1, ?2, 3)",
            rusqlite::params!["test-source", now],
        )
        .unwrap();

        // Insert 3 chunks with different content
        let chunks = [
            (
                "authentication",
                "login flow with JWT tokens and session management",
            ),
            (
                "database schema",
                "SQL tables with indexes and foreign keys",
            ),
            (
                "error handling",
                "retry logic with exponential backoff and circuit breaker",
            ),
        ];

        for (title, content) in &chunks {
            conn.execute(
                "INSERT INTO chunks (title, content, content_type, source_id, line_start, line_end) VALUES (?1, ?2, 'prose', 1, 0, 0)",
                rusqlite::params![title, content],
            ).unwrap();

            let rowid = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO chunks_trigram (title, content, content_type, source_id, line_start, line_end) VALUES (?1, ?2, 'prose', 1, 0, 0)",
                rusqlite::params![title, content],
            ).unwrap();

            // Generate a simple deterministic embedding from the content
            let text = format!("{title} {content}");
            let embedding = test_embedding(&text);
            let blob = embedder::embedding_to_bytes(&embedding);

            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_rowid, embedding, dim) VALUES (?1, ?2, ?3)",
                rusqlite::params![rowid, blob, embedding.len() as i32],
            )
            .unwrap();
        }

        conn
    }

    /// Generate a simple test embedding (small dim for speed)
    fn test_embedding(text: &str) -> Vec<f32> {
        let dim = 16;
        let mut vec = vec![0.0f32; dim];
        for (i, byte) in text.bytes().enumerate() {
            vec[i % dim] += byte as f32;
        }
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for val in &mut vec {
            *val /= norm;
        }
        vec
    }

    #[test]
    fn test_vector_search_returns_results() {
        let conn = setup_vector_test_db();
        let query_emb = test_embedding("authentication login JWT");

        let results = vector_search(&conn, &query_emb, 10, None, None, None).unwrap();
        assert!(!results.is_empty(), "vector search should return results");
        assert!(results.len() <= 3, "should not exceed chunk count");
    }

    #[test]
    fn test_vector_search_ranks_by_similarity() {
        let conn = setup_vector_test_db();
        // Query very close to "authentication" chunk embedding
        let query_emb =
            test_embedding("authentication login flow with JWT tokens and session management");

        let results = vector_search(&conn, &query_emb, 10, None, None, None).unwrap();
        assert!(!results.is_empty());
        // First result should be the "authentication" chunk (most similar)
        assert_eq!(results[0].title, "authentication");
    }

    #[test]
    fn test_vector_search_respects_limit() {
        let conn = setup_vector_test_db();
        let query_emb = test_embedding("anything");

        let results = vector_search(&conn, &query_emb, 1, None, None, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_search_source_filter() {
        let conn = setup_vector_test_db();
        let query_emb = test_embedding("anything");

        // Filter by existing source
        let results =
            vector_search(&conn, &query_emb, 10, Some("test-source"), None, None).unwrap();
        assert!(!results.is_empty());

        // Filter by non-existing source
        let results =
            vector_search(&conn, &query_emb, 10, Some("nonexistent"), None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_search_type_filter() {
        let conn = setup_vector_test_db();
        let query_emb = test_embedding("anything");

        // Filter by existing type
        let results = vector_search(&conn, &query_emb, 10, None, None, Some("prose")).unwrap();
        assert!(!results.is_empty());

        // Filter by non-existing type
        let results = vector_search(&conn, &query_emb, 10, None, None, Some("code")).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_search_empty_db() {
        let conn = Connection::open_in_memory().unwrap();
        super::super::schema::init_content_schema(&conn).unwrap();

        let query_emb = vec![0.1f32; 16];
        let results = vector_search(&conn, &query_emb, 10, None, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_layer_search_without_embedder() {
        let conn = setup_vector_test_db();
        let weights = SearchWeights::default();

        // Should work fine without embedder — keyword layers only
        let results = multi_layer_search(
            &conn,
            "authentication",
            10,
            None,
            None,
            None,
            None,
            &weights,
        )
        .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_multi_layer_search_with_mock_embedder() {
        use crate::embedder::tests::MockEmbedder;

        let conn = setup_vector_test_db();
        let weights = SearchWeights::default();
        let embedder = MockEmbedder::new();

        let results = multi_layer_search(
            &conn,
            "authentication",
            10,
            None,
            None,
            None,
            Some(&embedder),
            &weights,
        )
        .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_results_include_line_numbers() {
        let conn = Connection::open_in_memory().unwrap();
        super::super::schema::init_content_schema(&conn).unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO sources (label, indexed_at, chunk_count) VALUES (?1, ?2, 1)",
            rusqlite::params!["line-test-source", now],
        )
        .unwrap();

        // Insert a chunk with explicit line numbers
        conn.execute(
            "INSERT INTO chunks (title, content, content_type, source_id, line_start, line_end) \
             VALUES (?1, ?2, 'prose', 1, 10, 25)",
            rusqlite::params![
                "line number test",
                "unique phrase for line number verification"
            ],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO chunks_trigram (title, content, content_type, source_id, line_start, line_end) \
             VALUES (?1, ?2, 'prose', 1, 10, 25)",
            rusqlite::params!["line number test", "unique phrase for line number verification"],
        )
        .unwrap();

        let weights = SearchWeights::default();
        let results = multi_layer_search(
            &conn,
            "unique phrase line number verification",
            10,
            None,
            None,
            None,
            None,
            &weights,
        )
        .unwrap();

        assert!(!results.is_empty(), "search should return results");
        let result = &results[0];
        assert_eq!(result.line_start, 10, "line_start should be 10");
        assert_eq!(result.line_end, 25, "line_end should be 25");
    }
}
