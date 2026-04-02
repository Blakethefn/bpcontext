use std::sync::atomic::{AtomicU64, Ordering};

/// Global session statistics for context savings tracking
pub static BYTES_INDEXED: AtomicU64 = AtomicU64::new(0);
pub static BYTES_RETURNED: AtomicU64 = AtomicU64::new(0);
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
        commands_executed: COMMANDS_EXECUTED.load(Ordering::Relaxed),
        searches_performed: SEARCHES_PERFORMED.load(Ordering::Relaxed),
    }
}

#[derive(Debug)]
pub struct StatsSnapshot {
    pub bytes_indexed: u64,
    pub bytes_returned: u64,
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
}
