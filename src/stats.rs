use std::sync::atomic::{AtomicU64, Ordering};

/// Global session statistics for context savings tracking
pub static BYTES_INDEXED: AtomicU64 = AtomicU64::new(0);
pub static BYTES_RETURNED: AtomicU64 = AtomicU64::new(0);
pub static BYTES_VISIBLE: AtomicU64 = AtomicU64::new(0);
pub static COMMANDS_EXECUTED: AtomicU64 = AtomicU64::new(0);
pub static SEARCHES_PERFORMED: AtomicU64 = AtomicU64::new(0);

/// Record bytes that were indexed (kept out of context)
pub fn record_indexed(bytes: u64) {
    BYTES_INDEXED.fetch_add(bytes, Ordering::Relaxed);
}

/// Record bytes that were returned to context
pub fn record_returned(bytes: u64) {
    BYTES_RETURNED.fetch_add(bytes, Ordering::Relaxed);
}

/// Record bytes of indexed content that were actually shown to the agent
pub fn record_visible(bytes: u64) {
    BYTES_VISIBLE.fetch_add(bytes, Ordering::Relaxed);
}

/// Record a command execution
pub fn record_command() {
    COMMANDS_EXECUTED.fetch_add(1, Ordering::Relaxed);
}

/// Record a search
pub fn record_search() {
    SEARCHES_PERFORMED.fetch_add(1, Ordering::Relaxed);
}

/// Get current stats snapshot
pub fn get_stats() -> StatsSnapshot {
    StatsSnapshot {
        bytes_indexed: BYTES_INDEXED.load(Ordering::Relaxed),
        bytes_returned: BYTES_RETURNED.load(Ordering::Relaxed),
        bytes_visible: BYTES_VISIBLE.load(Ordering::Relaxed),
        commands_executed: COMMANDS_EXECUTED.load(Ordering::Relaxed),
        searches_performed: SEARCHES_PERFORMED.load(Ordering::Relaxed),
    }
}

#[derive(Debug)]
pub struct StatsSnapshot {
    pub bytes_indexed: u64,
    pub bytes_returned: u64,
    pub bytes_visible: u64,
    pub commands_executed: u64,
    pub searches_performed: u64,
}

impl StatsSnapshot {
    /// Context savings ratio (higher = more savings)
    pub fn savings_ratio(&self) -> f64 {
        if self.bytes_returned == 0 {
            return 0.0;
        }
        (self.bytes_indexed as f64 + self.bytes_returned as f64) / self.bytes_returned as f64
    }

    /// Estimated tokens saved (approx 4 bytes per token)
    pub fn tokens_saved(&self) -> u64 {
        self.bytes_indexed / 4
    }

    /// Approximate share of indexed bytes that were surfaced to the agent
    pub fn visibility_ratio(&self) -> f64 {
        if self.bytes_indexed == 0 {
            return 0.0;
        }
        self.bytes_visible as f64 / self.bytes_indexed as f64
    }
}

#[cfg(test)]
pub fn reset() {
    BYTES_INDEXED.store(0, Ordering::Relaxed);
    BYTES_RETURNED.store(0, Ordering::Relaxed);
    BYTES_VISIBLE.store(0, Ordering::Relaxed);
    COMMANDS_EXECUTED.store(0, Ordering::Relaxed);
    SEARCHES_PERFORMED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visibility_ratio_calculation() {
        let snapshot = StatsSnapshot {
            bytes_indexed: 10_000,
            bytes_returned: 800,
            bytes_visible: 500,
            commands_executed: 0,
            searches_performed: 0,
        };

        assert!((snapshot.visibility_ratio() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_record_visible_increments() {
        reset();
        record_visible(321);

        let snapshot = get_stats();
        assert_eq!(snapshot.bytes_visible, 321);
    }
}
