use anyhow::Result;
use rusqlite::Connection;

use crate::search_model::{normalize_query, SearchProfile};

#[derive(Debug, Clone)]
pub struct SearchMemoryEntry {
    pub normalized_query: String,
    pub profile: String,
    pub learned_query: String,
    pub learned_source_hint: Option<String>,
    pub learned_filter_hint: Option<String>,
    pub winning_source_label: Option<String>,
    pub winning_file_path: Option<String>,
    pub success_count: i64,
}

#[derive(Debug, Clone)]
pub struct SearchRunHit {
    pub source_type: String,
    pub source_label: String,
    pub file_path: Option<String>,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct SearchRunRecord {
    pub normalized_query: String,
    pub profile: String,
    pub effective_query: String,
    pub applied_source_hint: Option<String>,
    pub applied_filter_hint: Option<String>,
}

pub fn init_session_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS search_runs (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             normalized_query TEXT NOT NULL,
             profile TEXT NOT NULL,
             effective_query TEXT NOT NULL,
             applied_source_hint TEXT,
             applied_filter_hint TEXT,
             memory_applied INTEGER NOT NULL DEFAULT 0,
             created_at TEXT NOT NULL DEFAULT (datetime('now'))
         );

         CREATE TABLE IF NOT EXISTS search_run_hits (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             search_run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
             source_type TEXT NOT NULL,
             source_label TEXT NOT NULL,
             file_path TEXT,
             title TEXT NOT NULL
         );

         CREATE INDEX IF NOT EXISTS idx_search_run_hits_label
             ON search_run_hits(source_label);
         CREATE INDEX IF NOT EXISTS idx_search_run_hits_file
             ON search_run_hits(file_path);",
    )?;
    Ok(())
}

pub fn init_knowledge_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS search_memory (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             normalized_query TEXT NOT NULL,
             profile TEXT NOT NULL,
             learned_query TEXT NOT NULL,
             learned_source_hint TEXT,
             learned_filter_hint TEXT,
             winning_source_label TEXT,
             winning_file_path TEXT,
             success_count INTEGER NOT NULL DEFAULT 1,
             last_success TEXT NOT NULL DEFAULT (datetime('now')),
             UNIQUE(normalized_query, profile)
         );",
    )?;
    Ok(())
}

pub fn lookup_memory(
    conn: &Connection,
    query: &str,
    profile: SearchProfile,
) -> Result<Option<SearchMemoryEntry>> {
    let normalized = normalize_query(query);
    let mut stmt = conn.prepare(
        "SELECT normalized_query, profile, learned_query, learned_source_hint,
                learned_filter_hint, winning_source_label, winning_file_path,
                success_count
         FROM search_memory
         WHERE normalized_query = ?1 AND profile = ?2",
    )?;

    let entry = stmt
        .query_row(rusqlite::params![normalized, profile.as_str()], |row| {
            Ok(SearchMemoryEntry {
                normalized_query: row.get(0)?,
                profile: row.get(1)?,
                learned_query: row.get(2)?,
                learned_source_hint: row.get(3)?,
                learned_filter_hint: row.get(4)?,
                winning_source_label: row.get(5)?,
                winning_file_path: row.get(6)?,
                success_count: row.get(7)?,
            })
        })
        .optional()?;

    Ok(entry)
}

pub fn record_search_run(
    conn: &Connection,
    query: &str,
    profile: SearchProfile,
    effective_query: &str,
    applied_source_hint: Option<&str>,
    applied_filter_hint: Option<&str>,
    memory_applied: bool,
    hits: &[SearchRunHit],
) -> Result<i64> {
    let normalized = normalize_query(query);
    conn.execute(
        "INSERT INTO search_runs
            (normalized_query, profile, effective_query, applied_source_hint, applied_filter_hint, memory_applied)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            normalized,
            profile.as_str(),
            effective_query,
            applied_source_hint,
            applied_filter_hint,
            if memory_applied { 1 } else { 0 }
        ],
    )?;
    let run_id = conn.last_insert_rowid();

    let mut stmt = conn.prepare(
        "INSERT INTO search_run_hits
            (search_run_id, source_type, source_label, file_path, title)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    for hit in hits {
        stmt.execute(rusqlite::params![
            run_id,
            hit.source_type,
            hit.source_label,
            hit.file_path,
            hit.title
        ])?;
    }
    Ok(run_id)
}

pub fn reinforce_from_read_chunks(
    session_conn: &Connection,
    knowledge_conn: &Connection,
    label: &str,
) -> Result<bool> {
    let run = latest_run_for_label(session_conn, label)?;
    if let Some(run) = run {
        upsert_memory(
            knowledge_conn,
            &run.normalized_query,
            &run.profile,
            &run.effective_query,
            Some(label),
            run.applied_filter_hint.as_deref(),
            Some(label),
            None,
        )?;
        return Ok(true);
    }
    Ok(false)
}

pub fn reinforce_from_knowledge_links(
    session_conn: &Connection,
    knowledge_conn: &Connection,
    query: Option<&str>,
    file_path: Option<&str>,
    source_label: Option<&str>,
) -> Result<bool> {
    let run = if let Some(file_path) = file_path {
        latest_run_for_file(session_conn, file_path)?
    } else if let Some(query) = query {
        latest_run_for_query(session_conn, query)?
    } else {
        None
    };

    if let Some(run) = run {
        upsert_memory(
            knowledge_conn,
            &run.normalized_query,
            &run.profile,
            &run.effective_query,
            run.applied_source_hint.as_deref(),
            run.applied_filter_hint.as_deref(),
            source_label.or(run.applied_source_hint.as_deref()),
            file_path,
        )?;
        return Ok(true);
    }
    Ok(false)
}

fn latest_run_for_label(conn: &Connection, label: &str) -> Result<Option<SearchRunRecord>> {
    let mut stmt = conn.prepare(
        "SELECT sr.normalized_query, sr.profile, sr.effective_query,
                sr.applied_source_hint, sr.applied_filter_hint
         FROM search_runs sr
         JOIN search_run_hits sh ON sh.search_run_id = sr.id
         WHERE sh.source_type = 'session' AND sh.source_label = ?1
         ORDER BY sr.id DESC
         LIMIT 1",
    )?;
    Ok(stmt
        .query_row(rusqlite::params![label], |row| {
            Ok(SearchRunRecord {
                normalized_query: row.get(0)?,
                profile: row.get(1)?,
                effective_query: row.get(2)?,
                applied_source_hint: row.get(3)?,
                applied_filter_hint: row.get(4)?,
            })
        })
        .optional()?)
}

fn latest_run_for_file(conn: &Connection, file_path: &str) -> Result<Option<SearchRunRecord>> {
    let mut stmt = conn.prepare(
        "SELECT sr.normalized_query, sr.profile, sr.effective_query,
                sr.applied_source_hint, sr.applied_filter_hint
         FROM search_runs sr
         JOIN search_run_hits sh ON sh.search_run_id = sr.id
         WHERE sh.source_type = 'knowledge' AND sh.file_path = ?1
         ORDER BY sr.id DESC
         LIMIT 1",
    )?;
    Ok(stmt
        .query_row(rusqlite::params![file_path], |row| {
            Ok(SearchRunRecord {
                normalized_query: row.get(0)?,
                profile: row.get(1)?,
                effective_query: row.get(2)?,
                applied_source_hint: row.get(3)?,
                applied_filter_hint: row.get(4)?,
            })
        })
        .optional()?)
}

fn latest_run_for_query(conn: &Connection, query: &str) -> Result<Option<SearchRunRecord>> {
    let normalized = normalize_query(query);
    let mut stmt = conn.prepare(
        "SELECT normalized_query, profile, effective_query, applied_source_hint, applied_filter_hint
         FROM search_runs
         WHERE normalized_query = ?1
         ORDER BY id DESC
         LIMIT 1",
    )?;
    Ok(stmt
        .query_row(rusqlite::params![normalized], |row| {
            Ok(SearchRunRecord {
                normalized_query: row.get(0)?,
                profile: row.get(1)?,
                effective_query: row.get(2)?,
                applied_source_hint: row.get(3)?,
                applied_filter_hint: row.get(4)?,
            })
        })
        .optional()?)
}

fn upsert_memory(
    conn: &Connection,
    normalized_query: &str,
    profile: &str,
    learned_query: &str,
    learned_source_hint: Option<&str>,
    learned_filter_hint: Option<&str>,
    winning_source_label: Option<&str>,
    winning_file_path: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO search_memory
            (normalized_query, profile, learned_query, learned_source_hint, learned_filter_hint,
             winning_source_label, winning_file_path, success_count, last_success)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, datetime('now'))
         ON CONFLICT(normalized_query, profile) DO UPDATE SET
             learned_query = excluded.learned_query,
             learned_source_hint = COALESCE(excluded.learned_source_hint, search_memory.learned_source_hint),
             learned_filter_hint = COALESCE(excluded.learned_filter_hint, search_memory.learned_filter_hint),
             winning_source_label = COALESCE(excluded.winning_source_label, search_memory.winning_source_label),
             winning_file_path = COALESCE(excluded.winning_file_path, search_memory.winning_file_path),
             success_count = search_memory.success_count + 1,
             last_success = datetime('now')",
        rusqlite::params![
            normalized_query,
            profile,
            learned_query,
            learned_source_hint,
            learned_filter_hint,
            winning_source_label,
            winning_file_path
        ],
    )?;
    Ok(())
}

use rusqlite::OptionalExtension;
