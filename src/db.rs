use anyhow::{Context, Result};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Base directory for all bpcontext data
pub(crate) fn data_dir() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .context("Could not find local data directory")?
        .join("bpcontext");
    Ok(dir)
}

/// Hash a project directory path to a short hex string for DB isolation
fn project_hash(project_dir: &Path) -> String {
    let normalized = project_dir
        .canonicalize()
        .unwrap_or_else(|_| project_dir.to_path_buf());
    let mut hasher = Sha256::new();
    hasher.update(normalized.to_string_lossy().as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..8])
}

/// Get the path to a per-project content database
pub fn content_db_path(project_dir: &Path) -> Result<PathBuf> {
    let dir = data_dir()?.join("content");
    fs::create_dir_all(&dir)?;
    let hash = project_hash(project_dir);
    Ok(dir.join(format!("{hash}.db")))
}

/// Get the path to a per-project session database
pub fn session_db_path(project_dir: &Path) -> Result<PathBuf> {
    let dir = data_dir()?.join("sessions");
    fs::create_dir_all(&dir)?;
    let hash = project_hash(project_dir);
    Ok(dir.join(format!("{hash}.db")))
}

/// Open a SQLite connection with WAL mode and recommended pragmas
pub fn open_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)
        .with_context(|| format!("Failed to open database at {}", path.display()))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;",
    )?;
    Ok(conn)
}

/// List all content DB files with their modification times
#[allow(dead_code)]
pub fn list_content_dbs() -> Result<Vec<(PathBuf, std::time::SystemTime)>> {
    let dir = data_dir()?.join("content");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut dbs = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "db") {
            let modified = entry.metadata()?.modified()?;
            dbs.push((path, modified));
        }
    }
    Ok(dbs)
}

/// Delete content DBs older than the given number of days
#[allow(dead_code)]
pub fn cleanup_stale_dbs(stale_days: u64) -> Result<u32> {
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(stale_days * 86400);
    let dbs = list_content_dbs()?;
    let mut deleted = 0u32;
    for (path, modified) in dbs {
        if modified < cutoff {
            // Also remove WAL and SHM files
            let _ = fs::remove_file(path.with_extension("db-wal"));
            let _ = fs::remove_file(path.with_extension("db-shm"));
            fs::remove_file(&path)?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

/// Hex encoding helper (avoids adding the hex crate)
mod hex {
    use std::fmt::Write;

    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}
