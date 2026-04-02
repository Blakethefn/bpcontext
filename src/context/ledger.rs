use anyhow::{Context, Result};
use rusqlite::Connection;

/// Record that content was returned to the agent's context.
///
/// If the same `source_label` was already returned, increments `access_count`
/// and updates the timestamp rather than inserting a duplicate row.
pub fn record_return(
    conn: &Connection,
    chunk_rowid: i64,
    source_label: &str,
    token_estimate: u64,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();

    // Try to update an existing entry for this source_label
    let updated = conn.execute(
        "UPDATE context_ledger
         SET access_count = access_count + 1,
             token_estimate = token_estimate + ?1,
             returned_at = ?2
         WHERE source_label = ?3",
        rusqlite::params![token_estimate as i64, now, source_label],
    )?;

    if updated == 0 {
        conn.execute(
            "INSERT INTO context_ledger (chunk_rowid, source_label, token_estimate, returned_at, access_count)
             VALUES (?1, ?2, ?3, ?4, 1)",
            rusqlite::params![chunk_rowid, source_label, token_estimate as i64, now],
        )?;
    }

    Ok(())
}

/// Record that a tool response of `text_len` chars was returned under `source_label`.
/// Uses chars/4 as a rough token estimate.
pub fn record_tool_return(conn: &Connection, source_label: &str, text_len: usize) -> Result<()> {
    let token_estimate = text_len as u64 / 4;
    // Use 0 for chunk_rowid since this tracks the whole tool response, not a specific chunk
    record_return(conn, 0, source_label, token_estimate)
}

/// Total estimated tokens currently tracked in the ledger.
pub fn total_tokens(conn: &Connection) -> Result<u64> {
    let total: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM context_ledger",
            [],
            |row| row.get(0),
        )
        .context("failed to sum context ledger tokens")?;
    Ok(total as u64)
}

/// Per-source token breakdown, ordered by tokens descending.
pub fn source_breakdown(conn: &Connection) -> Result<Vec<SourceUsage>> {
    let mut stmt = conn.prepare(
        "SELECT source_label,
                SUM(token_estimate) as total_tokens,
                SUM(access_count) as accesses,
                MAX(returned_at) as last_access
         FROM context_ledger
         GROUP BY source_label
         ORDER BY total_tokens DESC",
    )?;

    let results = stmt
        .query_map([], |row| {
            Ok(SourceUsage {
                label: row.get(0)?,
                tokens: row.get::<_, i64>(1)? as u64,
                access_count: row.get::<_, i64>(2)? as u32,
                last_access: row.get(3)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

/// Number of distinct sources in the ledger.
pub fn source_count(conn: &Connection) -> Result<u32> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT source_label) FROM context_ledger",
            [],
            |row| row.get(0),
        )?;
    Ok(count as u32)
}

/// Clear the entire ledger (for session reset).
pub fn clear(conn: &Connection) -> Result<()> {
    conn.execute("DELETE FROM context_ledger", [])?;
    Ok(())
}

/// A source's usage summary from the ledger.
#[derive(Debug, Clone)]
pub struct SourceUsage {
    pub label: String,
    pub tokens: u64,
    pub access_count: u32,
    pub last_access: String,
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
    fn record_and_sum_tokens() {
        let conn = test_conn();
        record_return(&conn, 1, "src-a", 100).unwrap();
        record_return(&conn, 2, "src-b", 200).unwrap();
        assert_eq!(total_tokens(&conn).unwrap(), 300);
    }

    #[test]
    fn same_source_aggregates() {
        let conn = test_conn();
        record_return(&conn, 1, "src-a", 100).unwrap();
        record_return(&conn, 2, "src-a", 50).unwrap();

        let sources = source_breakdown(&conn).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].tokens, 150);
        assert_eq!(sources[0].access_count, 2);
    }

    #[test]
    fn source_count_works() {
        let conn = test_conn();
        record_return(&conn, 1, "a", 10).unwrap();
        record_return(&conn, 2, "b", 20).unwrap();
        record_return(&conn, 3, "a", 30).unwrap();
        assert_eq!(source_count(&conn).unwrap(), 2);
    }

    #[test]
    fn clear_removes_all() {
        let conn = test_conn();
        record_return(&conn, 1, "src", 100).unwrap();
        clear(&conn).unwrap();
        assert_eq!(total_tokens(&conn).unwrap(), 0);
    }

    #[test]
    fn record_tool_return_estimates_tokens() {
        let conn = test_conn();
        record_tool_return(&conn, "cmd-output", 400).unwrap();
        assert_eq!(total_tokens(&conn).unwrap(), 100); // 400 chars / 4
    }

    #[test]
    fn breakdown_ordered_by_tokens_desc() {
        let conn = test_conn();
        record_return(&conn, 1, "small", 10).unwrap();
        record_return(&conn, 2, "big", 500).unwrap();
        record_return(&conn, 3, "medium", 100).unwrap();

        let sources = source_breakdown(&conn).unwrap();
        assert_eq!(sources[0].label, "big");
        assert_eq!(sources[1].label, "medium");
        assert_eq!(sources[2].label, "small");
    }
}
