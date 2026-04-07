pub mod advisor;
pub mod ledger;

use anyhow::Result;
use rusqlite::Connection;

use crate::stats;
use advisor::{RelevanceWeights, ScoredSource};

/// Utilization thresholds that trigger inline alerts.
/// Each fires once per session (tracked by `fired` flags).
const THRESHOLDS: [u32; 5] = [40, 60, 70, 80, 90];

/// The context manager tracks what bpx returns to the agent
/// and generates inline alerts when context utilization crosses thresholds.
pub struct ContextManager {
    /// Token budget for the session (configurable, default 200k)
    budget: u64,
    /// Stale threshold in minutes (configurable, default 30)
    stale_minutes: u64,
    /// Relevance weights for scoring
    weights: RelevanceWeights,
    /// Which thresholds have already fired this session
    fired: [bool; 5],
    /// Whether the low-visibility warning has already fired this session
    visibility_alert_fired: bool,
}

impl ContextManager {
    pub fn new(budget: u64, stale_minutes: u64, weights: RelevanceWeights) -> Self {
        Self {
            budget,
            stale_minutes,
            weights,
            fired: [false; 5],
            visibility_alert_fired: false,
        }
    }

    /// Record that a tool response was returned to the agent.
    pub fn record(&self, conn: &Connection, source_label: &str, text_len: usize) -> Result<()> {
        ledger::record_tool_return(conn, source_label, text_len)
    }

    /// Check if any threshold was crossed and return an alert string if so.
    ///
    /// Returns `None` if no new threshold was crossed.
    /// Each threshold fires at most once per session.
    pub fn check_alert(&mut self, conn: &Connection) -> Result<Option<String>> {
        let used = ledger::total_tokens(conn)?;
        let pct = if self.budget > 0 {
            (used as f64 / self.budget as f64 * 100.0) as u32
        } else {
            0
        };

        let session_stats = stats::get_stats();

        // Find the highest threshold that was crossed but not yet fired
        let mut alert_idx = None;
        for (i, &threshold) in THRESHOLDS.iter().enumerate() {
            if pct >= threshold && !self.fired[i] {
                alert_idx = Some(i);
            }
        }

        let idx = match alert_idx {
            Some(i) => i,
            None => {
                if !self.visibility_alert_fired
                    && session_stats.bytes_indexed > 10_000
                    && session_stats.visibility_ratio() < 0.05
                {
                    self.visibility_alert_fired = true;
                    return Ok(Some(format!(
                        "[bpx visibility: only {:.1}% of indexed content seen. Use bpx_search or bpx_read_chunks.]",
                        session_stats.visibility_ratio() * 100.0
                    )));
                }
                return Ok(None);
            }
        };

        // Mark this and all lower thresholds as fired
        for i in 0..=idx {
            self.fired[i] = true;
        }

        let threshold = THRESHOLDS[idx];
        let alert = self.format_alert(conn, threshold, used)?;
        Ok(Some(alert))
    }

    /// Format an alert for the given threshold tier.
    fn format_alert(&self, conn: &Connection, threshold: u32, used: u64) -> Result<String> {
        let n_sources = ledger::source_count(conn)?;

        match threshold {
            40 => Ok(format!(
                "[bpx context: 40% — {used}/{} tokens across {n_sources} sources]",
                self.budget
            )),

            60 => {
                let sources = ledger::source_breakdown(conn)?;
                let top: Vec<String> = sources
                    .iter()
                    .take(3)
                    .map(|s| format!("{} ({}t)", s.label, s.tokens))
                    .collect();
                Ok(format!(
                    "[bpx context: 60% — top consumers: {}. Consider searching instead of re-reading.]",
                    top.join(", ")
                ))
            }

            70 => {
                let scored = advisor::score_sources(conn, &self.weights, self.stale_minutes)?;
                let (keep, drop) = advisor::keep_drop_lists(&scored, 3, 3);

                let keep_str = format_scored_list(&keep);
                let drop_str: Vec<String> = drop.iter().map(|s| s.label.clone()).collect();

                Ok(format!(
                    "[bpx context: 70% — recommend dropping: {}. Relevance scores: {}.]",
                    drop_str.join(", "),
                    keep_str
                ))
            }

            80 => {
                let scored = advisor::score_sources(conn, &self.weights, self.stale_minutes)?;
                let (keep, drop) = advisor::keep_drop_lists(&scored, 3, 3);
                let savings = advisor::estimate_savings(&drop);

                let keep_str: Vec<String> = keep.iter().map(|s| s.label.clone()).collect();
                let drop_str: Vec<String> = drop.iter().map(|s| s.label.clone()).collect();

                Ok(format!(
                    "[bpx context: 80% — COMPACT RECOMMENDED. Keep: {}. Drop: {}. Estimated savings: {} tokens.]",
                    keep_str.join(", "),
                    drop_str.join(", "),
                    savings
                ))
            }

            90 => {
                let remaining = self.budget.saturating_sub(used);
                let scored = advisor::score_sources(conn, &self.weights, self.stale_minutes)?;
                let top3: Vec<String> = scored.iter().take(3).map(|s| s.label.clone()).collect();

                Ok(format!(
                    "[bpx context: 90% — CRITICAL. Immediate compaction needed. Only {} tokens remaining. Highest-value sources: {}.]",
                    remaining,
                    top3.join(", ")
                ))
            }

            _ => Ok(String::new()),
        }
    }

    /// Generate a keep/drop recommendation for the precompact hook.
    pub fn precompact_recommendation(&self, conn: &Connection) -> Result<String> {
        let used = ledger::total_tokens(conn)?;
        let scored = advisor::score_sources(conn, &self.weights, self.stale_minutes)?;

        if scored.is_empty() {
            return Ok("No tracked sources — no context recommendations.".to_string());
        }

        let (keep, drop) = advisor::keep_drop_lists(&scored, 3, 3);
        let savings = advisor::estimate_savings(&drop);

        let mut out = format!(
            "## Context Budget\n\n\
             - Tokens used: {used}/{}\n\
             - Sources tracked: {}\n\n",
            self.budget,
            scored.len()
        );

        out.push_str("## Keep (highest relevance)\n\n");
        for s in &keep {
            out.push_str(&format!(
                "- **{}** — {}t, {} accesses, relevance {:.2}\n",
                s.label, s.tokens, s.access_count, s.relevance
            ));
        }

        out.push_str("\n## Drop (lowest relevance)\n\n");
        for s in &drop {
            out.push_str(&format!(
                "- **{}** — {}t, {} accesses, relevance {:.2}\n",
                s.label, s.tokens, s.access_count, s.relevance
            ));
        }

        out.push_str(&format!(
            "\nEstimated savings if dropped: {} tokens\n",
            savings
        ));

        Ok(out)
    }

    /// Reset the session (clear ledger and fired flags).
    pub fn reset(&mut self, conn: &Connection) -> Result<()> {
        ledger::clear(conn)?;
        self.fired = [false; 5];
        self.visibility_alert_fired = false;
        Ok(())
    }
}

fn format_scored_list(sources: &[&ScoredSource]) -> String {
    sources
        .iter()
        .map(|s| format!("{} ({:.2})", s.label, s.relevance))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats;
    use crate::store::schema;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        schema::init_session_schema(&conn).unwrap();
        conn
    }

    fn test_mgr(budget: u64) -> ContextManager {
        ContextManager::new(budget, 30, RelevanceWeights::default())
    }

    #[test]
    fn no_alert_below_40_pct() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);
        // Suppress visibility alert — global stats are shared across parallel tests
        mgr.visibility_alert_fired = true;

        // 39% usage
        ledger::record_return(&conn, 1, "src", 390).unwrap();

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_none());
    }

    #[test]
    fn alert_at_40_pct() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);
        // Suppress visibility alert — global stats are shared across parallel tests
        mgr.visibility_alert_fired = true;

        ledger::record_return(&conn, 1, "src", 400).unwrap();

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
        let text = alert.unwrap();
        assert!(text.contains("40%"));
        assert!(text.contains("400/1000"));
    }

    #[test]
    fn threshold_fires_only_once() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        ledger::record_return(&conn, 1, "src", 400).unwrap();

        let first = mgr.check_alert(&conn).unwrap();
        assert!(first.is_some());

        let second = mgr.check_alert(&conn).unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn skipped_thresholds_fire_highest() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        // Jump straight to 80%
        ledger::record_return(&conn, 1, "src", 800).unwrap();

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
        let text = alert.unwrap();
        // Should fire the 80% alert (highest crossed)
        assert!(text.contains("80%"));
        assert!(text.contains("COMPACT RECOMMENDED"));
    }

    #[test]
    fn alert_60_shows_top_consumers() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        ledger::record_return(&conn, 1, "big-file.rs", 400).unwrap();
        ledger::record_return(&conn, 2, "small.rs", 200).unwrap();

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
        let text = alert.unwrap();
        assert!(text.contains("60%"));
        assert!(text.contains("big-file.rs"));
    }

    #[test]
    fn alert_90_shows_remaining() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        ledger::record_return(&conn, 1, "huge", 900).unwrap();

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
        let text = alert.unwrap();
        assert!(text.contains("90%"));
        assert!(text.contains("CRITICAL"));
        assert!(text.contains("100 tokens remaining"));
    }

    #[test]
    fn reset_clears_ledger_and_fired() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        ledger::record_return(&conn, 1, "src", 500).unwrap();
        mgr.check_alert(&conn).unwrap(); // fires 40%

        mgr.reset(&conn).unwrap();

        assert_eq!(ledger::total_tokens(&conn).unwrap(), 0);

        // After reset, thresholds should be available again
        ledger::record_return(&conn, 1, "src", 500).unwrap();
        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
    }

    #[test]
    fn precompact_recommendation_has_sections() {
        stats::reset();
        let conn = test_conn();
        let mgr = test_mgr(1000);

        ledger::record_return(&conn, 1, "a", 200).unwrap();
        ledger::record_return(&conn, 2, "b", 300).unwrap();
        ledger::record_return(&conn, 3, "c", 100).unwrap();

        let rec = mgr.precompact_recommendation(&conn).unwrap();
        assert!(rec.contains("## Context Budget"));
        assert!(rec.contains("## Keep"));
        assert!(rec.contains("## Drop"));
    }

    #[test]
    fn record_and_check_flow() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(1000);

        // Simulate a series of tool returns
        mgr.record(&conn, "cargo-test", 1600).unwrap(); // 400 tokens
        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some()); // 40% crossed

        mgr.record(&conn, "server.rs", 800).unwrap(); // +200 = 600 total
        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some()); // 60% crossed
        assert!(alert.unwrap().contains("60%"));
    }

    #[test]
    fn test_low_visibility_warning() {
        stats::reset();
        let conn = test_conn();
        let mut mgr = test_mgr(100_000);

        stats::record_indexed(20_000);
        stats::record_visible(500);

        let alert = mgr.check_alert(&conn).unwrap();
        assert!(alert.is_some());
        let text = alert.unwrap();
        assert!(text.contains("bpx visibility"));
        assert!(text.contains("2.5%"));
    }
}
