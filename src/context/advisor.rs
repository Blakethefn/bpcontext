use anyhow::Result;
use rusqlite::Connection;

use super::ledger;

/// Relevance scoring weights (configurable via `[context]` config).
pub struct RelevanceWeights {
    pub recency: f64,
    pub frequency: f64,
    pub staleness: f64,
}

impl Default for RelevanceWeights {
    fn default() -> Self {
        Self {
            recency: 0.4,
            frequency: 0.3,
            staleness: 0.3,
        }
    }
}

/// A scored source with relevance metadata for keep/drop recommendations.
#[derive(Debug, Clone)]
pub struct ScoredSource {
    pub label: String,
    pub tokens: u64,
    pub access_count: u32,
    pub relevance: f64,
}

/// Score all tracked sources by relevance and return sorted (highest first).
pub fn score_sources(
    conn: &Connection,
    weights: &RelevanceWeights,
    stale_threshold_minutes: u64,
) -> Result<Vec<ScoredSource>> {
    let sources = ledger::source_breakdown(conn)?;
    if sources.is_empty() {
        return Ok(Vec::new());
    }

    let now = chrono::Utc::now();
    let max_accesses = sources
        .iter()
        .map(|s| s.access_count)
        .max()
        .unwrap_or(1)
        .max(1) as f64;

    let mut scored: Vec<ScoredSource> = sources
        .into_iter()
        .map(|s| {
            let minutes_ago = parse_minutes_ago(&s.last_access, &now);

            // Recency: 1.0 for just accessed, decays toward 0
            let recency_score = 1.0 / (1.0 + minutes_ago / 10.0);

            // Frequency: normalized by max access count
            let frequency_score = s.access_count as f64 / max_accesses;

            // Staleness: 1.0 if stale (beyond threshold), 0.0 if fresh
            let staleness_score = if minutes_ago > stale_threshold_minutes as f64 {
                (minutes_ago - stale_threshold_minutes as f64).min(60.0) / 60.0
            } else {
                0.0
            };

            let relevance = (weights.recency * recency_score)
                + (weights.frequency * frequency_score)
                - (weights.staleness * staleness_score);

            ScoredSource {
                label: s.label,
                tokens: s.tokens,
                access_count: s.access_count,
                relevance,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(scored)
}

/// Generate keep/drop recommendation lists.
///
/// "Keep" = top sources by relevance. "Drop" = bottom sources.
/// Returns (keep, drop).
pub fn keep_drop_lists(
    scored: &[ScoredSource],
    keep_count: usize,
    drop_count: usize,
) -> (Vec<&ScoredSource>, Vec<&ScoredSource>) {
    let keep: Vec<&ScoredSource> = scored.iter().take(keep_count).collect();
    // Only drop sources that aren't already in the keep list
    let keep_labels: std::collections::HashSet<&str> =
        keep.iter().map(|s| s.label.as_str()).collect();
    let drop: Vec<&ScoredSource> = scored
        .iter()
        .rev()
        .filter(|s| !keep_labels.contains(s.label.as_str()))
        .take(drop_count)
        .collect();
    (keep, drop)
}

/// Estimate token savings if the drop list sources were removed.
pub fn estimate_savings(drop_list: &[&ScoredSource]) -> u64 {
    drop_list.iter().map(|s| s.tokens).sum()
}

/// Parse an RFC3339 timestamp and return minutes elapsed since then.
fn parse_minutes_ago(timestamp: &str, now: &chrono::DateTime<chrono::Utc>) -> f64 {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|t| {
            let diff = *now - t.with_timezone(&chrono::Utc);
            diff.num_seconds().max(0) as f64 / 60.0
        })
        .unwrap_or(999.0) // If parse fails, treat as very stale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::schema;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        schema::init_session_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn score_empty_returns_empty() {
        let conn = test_conn();
        let weights = RelevanceWeights::default();
        let scored = score_sources(&conn, &weights, 30).unwrap();
        assert!(scored.is_empty());
    }

    #[test]
    fn recently_accessed_scores_higher() {
        let conn = test_conn();
        let now = chrono::Utc::now().to_rfc3339();

        // "recent" — accessed now
        conn.execute(
            "INSERT INTO context_ledger (chunk_rowid, source_label, token_estimate, returned_at, access_count) VALUES (1, 'recent', 100, ?1, 1)",
            rusqlite::params![now],
        ).unwrap();

        // "old" — accessed 60 minutes ago
        let old_time = (chrono::Utc::now() - chrono::Duration::minutes(60)).to_rfc3339();
        conn.execute(
            "INSERT INTO context_ledger (chunk_rowid, source_label, token_estimate, returned_at, access_count) VALUES (2, 'old', 100, ?1, 1)",
            rusqlite::params![old_time],
        ).unwrap();

        let weights = RelevanceWeights::default();
        let scored = score_sources(&conn, &weights, 30).unwrap();

        assert_eq!(scored[0].label, "recent");
        assert!(scored[0].relevance > scored[1].relevance);
    }

    #[test]
    fn frequently_accessed_scores_higher() {
        let conn = test_conn();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO context_ledger (chunk_rowid, source_label, token_estimate, returned_at, access_count) VALUES (1, 'frequent', 100, ?1, 10)",
            rusqlite::params![now],
        ).unwrap();

        conn.execute(
            "INSERT INTO context_ledger (chunk_rowid, source_label, token_estimate, returned_at, access_count) VALUES (2, 'infrequent', 100, ?1, 1)",
            rusqlite::params![now],
        ).unwrap();

        let weights = RelevanceWeights::default();
        let scored = score_sources(&conn, &weights, 30).unwrap();

        assert_eq!(scored[0].label, "frequent");
        assert!(scored[0].relevance > scored[1].relevance);
    }

    #[test]
    fn keep_drop_lists_correct() {
        let scored = vec![
            ScoredSource { label: "a".into(), tokens: 100, access_count: 5, relevance: 0.9 },
            ScoredSource { label: "b".into(), tokens: 200, access_count: 3, relevance: 0.7 },
            ScoredSource { label: "c".into(), tokens: 300, access_count: 1, relevance: 0.3 },
            ScoredSource { label: "d".into(), tokens: 150, access_count: 1, relevance: 0.1 },
        ];

        let (keep, drop) = keep_drop_lists(&scored, 2, 2);
        assert_eq!(keep.len(), 2);
        assert_eq!(keep[0].label, "a");
        assert_eq!(keep[1].label, "b");
        assert_eq!(drop.len(), 2);
        assert_eq!(drop[0].label, "d");
        assert_eq!(drop[1].label, "c");
    }

    #[test]
    fn estimate_savings_sums_tokens() {
        let a = ScoredSource { label: "x".into(), tokens: 100, access_count: 1, relevance: 0.1 };
        let b = ScoredSource { label: "y".into(), tokens: 200, access_count: 1, relevance: 0.2 };
        assert_eq!(estimate_savings(&[&a, &b]), 300);
    }

    #[test]
    fn keep_drop_no_overlap_when_few_sources() {
        let scored = vec![
            ScoredSource { label: "a".into(), tokens: 100, access_count: 3, relevance: 0.9 },
            ScoredSource { label: "b".into(), tokens: 200, access_count: 1, relevance: 0.3 },
        ];

        let (keep, drop) = keep_drop_lists(&scored, 3, 3);
        // keep gets both (only 2 sources), drop should be empty (both already in keep)
        assert_eq!(keep.len(), 2);
        assert!(drop.is_empty());
    }
}
