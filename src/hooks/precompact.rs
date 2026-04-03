use anyhow::Result;
use serde_json::json;

use super::write_hook_output;
use crate::config::Config;
use crate::context::advisor::RelevanceWeights;
use crate::context::ContextManager;
use crate::session::snapshot;
use crate::session::SessionStore;

/// Handle PreCompact hook
///
/// Builds a resume snapshot from session events and returns it
/// so the LLM can restore context after compaction. Also includes
/// context manager keep/drop recommendations.
pub fn handle() -> Result<()> {
    let project_dir = std::env::current_dir()?;
    let config = Config::load()?;
    let session_store = SessionStore::open(&project_dir)?;

    let resume = snapshot::build_resume_snapshot(session_store.conn())?;

    // Generate context recommendations
    let weights = RelevanceWeights {
        recency: config.context.recency_weight,
        frequency: config.context.frequency_weight,
        staleness: config.context.staleness_weight,
    };
    let ctx_mgr = ContextManager::new(
        config.context.budget_tokens,
        config.context.stale_threshold_minutes,
        weights,
    );
    let context_advice = ctx_mgr
        .precompact_recommendation(session_store.conn())
        .unwrap_or_default();

    let combined_resume = if context_advice.is_empty() {
        resume
    } else {
        format!("{resume}\n\n{context_advice}")
    };

    write_hook_output(&json!({
        "decision": "approve",
        "resume_snapshot": combined_resume
    }))
}
