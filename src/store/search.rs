use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

/// A search result with relevance metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub content: String,
    pub content_type: String,
    pub source_id: i64,
    pub score: f64,
    pub source: String,
}

/// Multi-layer search: BM25 → trigram → Levenshtein correction → RRF fusion
pub fn multi_layer_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    type_filter: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let mut all_results: HashMap<String, ScoredResult> = HashMap::new();

    // Layer 1: FTS5 BM25 with Porter stemming
    let bm25_results = fts5_bm25_search(conn, query, limit, source_filter, type_filter)?;
    merge_results(&mut all_results, &bm25_results, 1);

    // Layer 2: Trigram search (if BM25 returned fewer than limit)
    if bm25_results.len() < limit as usize {
        let trigram_results = trigram_search(conn, query, limit, source_filter, type_filter)?;
        merge_results(&mut all_results, &trigram_results, 2);
    }

    // Layer 3: Levenshtein fuzzy correction (if still too few results)
    if all_results.len() < limit as usize {
        let fuzzy_results = fuzzy_search(conn, query, limit, source_filter, type_filter)?;
        merge_results(&mut all_results, &fuzzy_results, 3);
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

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit as usize);
    Ok(results)
}

/// FTS5 BM25 search on the primary chunks table
fn fts5_bm25_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
    type_filter: Option<&str>,
) -> Result<Vec<RankedResult>> {
    let fts_query = sanitize_fts_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut sql = String::from(
        "SELECT chunks.title, chunks.content, chunks.content_type, chunks.source_id,
                bm25(chunks) as rank, COALESCE(s.label, '') as source_label
         FROM chunks
         LEFT JOIN sources s ON chunks.source_id = s.id
         WHERE chunks MATCH ?1"
    );

    if let Some(tf) = type_filter {
        sql.push_str(&format!(" AND chunks.content_type = '{}'", escape_sql(tf)));
    }
    if let Some(sf) = source_filter {
        sql.push_str(&format!(" AND s.label LIKE '%{}%'", escape_sql(sf)));
    }

    sql.push_str(&format!(" ORDER BY rank LIMIT {limit}"));

    let mut stmt = conn.prepare(&sql)?;
    let results = stmt.query_map(rusqlite::params![fts_query], |row| {
        Ok(RankedResult {
            title: row.get(0)?,
            content: row.get(1)?,
            content_type: row.get(2)?,
            source_id: row.get(3)?,
            rank: row.get::<_, f64>(4)?.abs(),
            source_label: row.get(5)?,
        })
    })?.filter_map(|r| r.ok())
    .collect();

    Ok(results)
}

/// Trigram search on the chunks_trigram table
fn trigram_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
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
                COALESCE(s.label, '') as source_label
         FROM chunks_trigram
         LEFT JOIN sources s ON chunks_trigram.source_id = s.id
         WHERE chunks_trigram MATCH ?1"
    );

    if let Some(tf) = type_filter {
        sql.push_str(&format!(" AND chunks_trigram.content_type = '{}'", escape_sql(tf)));
    }
    if let Some(sf) = source_filter {
        sql.push_str(&format!(" AND s.label LIKE '%{}%'", escape_sql(sf)));
    }

    sql.push_str(&format!(" ORDER BY rank LIMIT {limit}"));

    let mut stmt = conn.prepare(&sql)?;
    let results = stmt.query_map(rusqlite::params![fts_query], |row| {
        Ok(RankedResult {
            title: row.get(0)?,
            content: row.get(1)?,
            content_type: row.get(2)?,
            source_id: row.get(3)?,
            rank: row.get::<_, f64>(4)?.abs(),
            source_label: row.get(5)?,
        })
    })?.filter_map(|r| r.ok())
    .collect();

    Ok(results)
}

/// Levenshtein fuzzy search: correct query terms and re-search
fn fuzzy_search(
    conn: &Connection,
    query: &str,
    limit: u32,
    source_filter: Option<&str>,
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

    fts5_bm25_search(conn, &corrected_query, limit, source_filter, type_filter)
}

/// Extract frequent terms from the FTS5 index for fuzzy matching
fn extract_index_terms(conn: &Connection, limit: usize) -> Result<Vec<String>> {
    // Use FTS5 vocab table to get terms
    let result = conn.prepare(
        "SELECT DISTINCT term FROM chunks_vocab WHERE col = 'content' LIMIT ?1"
    );

    match result {
        Ok(mut stmt) => {
            let terms = stmt.query_map(rusqlite::params![limit as i64], |row| {
                row.get::<_, String>(0)
            })?.filter_map(|r| r.ok())
            .collect();
            Ok(terms)
        }
        Err(_) => {
            // Vocab table may not exist; create it and retry
            let _ = conn.execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks, instance);"
            );
            let mut stmt = conn.prepare(
                "SELECT DISTINCT term FROM chunks_vocab WHERE col = 'content' LIMIT ?1"
            )?;
            let terms = stmt.query_map(rusqlite::params![limit as i64], |row| {
                row.get::<_, String>(0)
            })?.filter_map(|r| r.ok())
            .collect();
            Ok(terms)
        }
    }
}

/// Merge ranked results into the accumulator using Reciprocal Rank Fusion
fn merge_results(
    acc: &mut HashMap<String, ScoredResult>,
    results: &[RankedResult],
    _layer: u32,
) {
    const K: f64 = 60.0;

    for (rank, result) in results.iter().enumerate() {
        let rrf_score = 1.0 / (K + rank as f64 + 1.0);
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
struct RankedResult {
    title: String,
    content: String,
    content_type: String,
    source_id: i64,
    rank: f64,
    source_label: String,
}

struct ScoredResult {
    title: String,
    content: String,
    content_type: String,
    source_id: i64,
    source_label: String,
    rrf_score: f64,
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
}
