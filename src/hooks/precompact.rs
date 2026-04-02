use anyhow::Result;
use serde_json::json;

use super::write_hook_output;
use crate::session::SessionStore;
use crate::session::snapshot;

/// Handle PreCompact hook
///
/// Builds a resume snapshot from session events and returns it
/// so the LLM can restore context after compaction.
pub fn handle() -> Result<()> {
    let project_dir = std::env::current_dir()?;
    let session_store = SessionStore::open(&project_dir)?;

    let resume = snapshot::build_resume_snapshot(session_store.conn())?;

    write_hook_output(&json!({
        "decision": "approve",
        "resume_snapshot": resume
    }))
}
