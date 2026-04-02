use anyhow::Result;
use rusqlite::Connection;

use super::events;

/// Maximum snapshot size in bytes
const MAX_SNAPSHOT_BYTES: usize = 2048;

/// Build a priority-weighted resume snapshot from session events
///
/// Returns a compressed text summary suitable for injection into
/// the LLM context after a compaction event.
pub fn build_resume_snapshot(conn: &Connection) -> Result<String> {
    let events = events::recent_events(conn, 100)?;
    if events.is_empty() {
        return Ok(String::from("No session events recorded."));
    }

    let mut snapshot = String::from("## Session Resume\n\n");
    let mut current_bytes = snapshot.len();

    // Group by category and summarize
    let mut categories: std::collections::HashMap<String, Vec<&events::EventRecord>> =
        std::collections::HashMap::new();

    for event in &events {
        let cat = event.category.as_deref().unwrap_or("general").to_string();
        categories.entry(cat).or_default().push(event);
    }

    // High-priority events first
    let mut sorted_cats: Vec<_> = categories.iter().collect();
    sorted_cats.sort_by(|a, b| {
        let max_a = a.1.iter().map(|e| e.priority).max().unwrap_or(0);
        let max_b = b.1.iter().map(|e| e.priority).max().unwrap_or(0);
        max_b.cmp(&max_a)
    });

    for (category, events) in sorted_cats {
        let section = format!("### {category}\n- {} events (latest: {})\n\n",
            events.len(),
            events.first().map(|e| e.created_at.as_str()).unwrap_or("unknown")
        );

        if current_bytes + section.len() > MAX_SNAPSHOT_BYTES {
            snapshot.push_str(&format!("... [{} more categories]\n",
                categories.len() - snapshot.matches("###").count()));
            break;
        }

        snapshot.push_str(&section);
        current_bytes += section.len();
    }

    Ok(snapshot)
}
