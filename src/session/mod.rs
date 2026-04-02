pub mod events;
pub mod snapshot;

use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;

use crate::db;
use crate::store::schema;

/// Session database for tracking events across compactions
pub struct SessionStore {
    conn: Connection,
}

impl SessionStore {
    /// Open or create a session store for the given project directory
    pub fn open(project_dir: &Path) -> Result<Self> {
        let db_path = db::session_db_path(project_dir)?;
        let conn = db::open_db(&db_path)?;
        schema::init_session_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Get the underlying connection
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}
