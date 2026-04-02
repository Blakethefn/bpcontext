use anyhow::Result;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

/// Event priority levels
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Priority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Insert a session event with dedup
pub fn insert_event(
    conn: &Connection,
    event_type: &str,
    category: Option<&str>,
    data: &str,
    priority: Priority,
) -> Result<bool> {
    let hash = compute_hash(data);
    let now = chrono::Utc::now().to_rfc3339();

    let result = conn.execute(
        "INSERT OR IGNORE INTO events (type, category, data, priority, data_hash, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![event_type, category, data, priority as u8, hash, now],
    )?;

    Ok(result > 0) // true if inserted (not a duplicate)
}

/// Get recent events ordered by priority and recency
pub fn recent_events(conn: &Connection, limit: u32) -> Result<Vec<EventRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, type, category, data, priority, created_at
         FROM events
         ORDER BY priority DESC, created_at DESC
         LIMIT ?1"
    )?;

    let events = stmt.query_map(rusqlite::params![limit], |row| {
        Ok(EventRecord {
            id: row.get(0)?,
            event_type: row.get(1)?,
            category: row.get(2)?,
            data: row.get(3)?,
            priority: row.get(4)?,
            created_at: row.get(5)?,
        })
    })?.filter_map(|r| r.ok())
    .collect();

    Ok(events)
}

/// Evict oldest low-priority events when count exceeds threshold
pub fn evict_if_needed(conn: &Connection, max_events: u32) -> Result<u32> {
    let count: u32 = conn.query_row(
        "SELECT COUNT(*) FROM events",
        [],
        |row| row.get(0),
    )?;

    if count <= max_events {
        return Ok(0);
    }

    let to_delete = count - max_events;
    let deleted = conn.execute(
        "DELETE FROM events WHERE id IN (
            SELECT id FROM events ORDER BY priority ASC, created_at ASC LIMIT ?1
        )",
        rusqlite::params![to_delete],
    )?;

    Ok(deleted as u32)
}

/// Count total events
#[allow(dead_code)]
pub fn event_count(conn: &Connection) -> Result<u32> {
    let count: u32 = conn.query_row(
        "SELECT COUNT(*) FROM events",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

fn compute_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct EventRecord {
    pub id: i64,
    pub event_type: String,
    pub category: Option<String>,
    pub data: String,
    pub priority: u8,
    pub created_at: String,
}
