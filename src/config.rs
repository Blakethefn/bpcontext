use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub general: GeneralConfig,
    pub search: SearchConfig,
    pub fetch: FetchConfig,
    pub integration: IntegrationConfig,
    pub cleanup: CleanupConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Max stdout capture in bytes (default 100KB)
    pub max_stdout_bytes: usize,
    /// Head/tail split ratio for truncation (default 0.6)
    pub head_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Max results per query (default 10)
    pub default_limit: u32,
    /// Max searches per throttle window (default 8)
    pub throttle_max: u32,
    /// Throttle window in seconds (default 60)
    pub throttle_window_secs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FetchConfig {
    /// URL cache TTL in hours (default 24)
    pub cache_ttl_hours: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Path to taskvault binary
    pub taskvault_bin: String,
    /// Path to obsidian vault
    pub vault_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CleanupConfig {
    /// Auto-delete content DBs older than this many days (default 14)
    pub stale_db_days: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                max_stdout_bytes: 102_400,
                head_ratio: 0.6,
            },
            search: SearchConfig {
                default_limit: 10,
                throttle_max: 8,
                throttle_window_secs: 60,
            },
            fetch: FetchConfig {
                cache_ttl_hours: 24,
            },
            integration: IntegrationConfig {
                taskvault_bin: "taskvault".to_string(),
                vault_path: dirs::home_dir()
                    .map(|h| h.join("obsidian-vault").to_string_lossy().to_string())
                    .unwrap_or_else(|| "~/obsidian-vault".to_string()),
            },
            cleanup: CleanupConfig {
                stale_db_days: 14,
            },
        }
    }
}

impl Config {
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Could not find config directory")?
            .join("bpcontext");
        Ok(config_dir.join("config.toml"))
    }

    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config at {}", path.display()))?;
        let config: Config =
            toml::from_str(&content).context("Failed to parse config.toml")?;
        Ok(config)
    }

    pub fn init_default() -> Result<()> {
        let path = Self::config_path()?;
        if path.exists() {
            println!("Config already exists at {}", path.display());
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let default = Config::default();
        let content = toml::to_string_pretty(&default)?;
        fs::write(&path, &content)?;
        println!("Created default config at {}", path.display());
        Ok(())
    }
}
