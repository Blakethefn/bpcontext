use anyhow::Result;
use serde_json::json;

use super::{read_hook_input, write_hook_output};
use crate::session::events::{self, Priority};
use crate::session::SessionStore;

/// Handle PostToolUse hook
///
/// Captures tool call responses and indexes them as session events
/// for continuity across context compactions.
pub fn handle() -> Result<()> {
    let input = read_hook_input()?;

    let tool_name = input["tool_name"].as_str().unwrap_or("unknown");
    let tool_response = input.get("tool_response");

    // Determine priority based on tool type
    let priority = match tool_name {
        "Bash" | "Read" | "Write" | "Edit" => Priority::Normal,
        "Agent" => Priority::High,
        _ => Priority::Low,
    };

    // Try to get project dir from environment
    let project_dir = std::env::current_dir()?;
    let session_store = SessionStore::open(&project_dir)?;

    // Build event data
    let event_data = json!({
        "tool": tool_name,
        "response_preview": truncate_response(tool_response),
    });

    events::insert_event(
        session_store.conn(),
        "tool_call",
        Some(tool_name),
        &event_data.to_string(),
        priority,
    )?;

    // Evict if over threshold
    events::evict_if_needed(session_store.conn(), 500)?;

    write_hook_output(&json!({ "decision": "approve" }))
}

/// Truncate a tool response to a reasonable size for event storage
fn truncate_response(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(v) => {
            let s = v.to_string();
            if s.len() > 500 {
                format!("{}...", &s[..497])
            } else {
                s
            }
        }
        None => String::from("(no response)"),
    }
}
