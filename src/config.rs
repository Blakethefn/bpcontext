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
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub knowledge: KnowledgeConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Max stdout capture in bytes (default 100KB)
    pub max_stdout_bytes: usize,
    /// Head/tail split ratio for truncation (default 0.6)
    pub head_ratio: f64,
    /// Files/outputs under this size bypass truncation entirely (default 8192)
    #[serde(default = "default_preview_threshold_bytes")]
    pub preview_threshold_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Max results per query (default 15)
    pub default_limit: u32,
    /// Max searches per throttle window (default 8)
    pub throttle_max: u32,
    /// Throttle window in seconds (default 60)
    pub throttle_window_secs: u64,
    /// Max bytes per snippet for top results (default 2000)
    #[serde(default = "default_snippet_bytes")]
    pub snippet_bytes: usize,
    /// Max bytes per snippet for lower-ranked results (default 800)
    #[serde(default = "default_secondary_snippet_bytes")]
    pub secondary_snippet_bytes: usize,
    /// Number of top-ranked results that get the larger snippet size (default 3)
    #[serde(default = "default_top_result_count")]
    pub top_result_count: usize,
    /// Multiplier for vector (semantic) RRF scores (default 1.0)
    #[serde(default = "default_weight")]
    pub vector_weight: f64,
    /// Multiplier for keyword (BM25/trigram/fuzzy) RRF scores (default 1.0)
    #[serde(default = "default_weight")]
    pub keyword_weight: f64,
}

fn default_preview_threshold_bytes() -> usize {
    8192
}

fn default_snippet_bytes() -> usize {
    2000
}

fn default_secondary_snippet_bytes() -> usize {
    800
}

fn default_top_result_count() -> usize {
    3
}

fn default_weight() -> f64 {
    1.0
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

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    /// Model name (default: all-MiniLM-L6-v2)
    pub model: String,
    /// Directory for storing downloaded models
    pub model_dir: String,
    /// Batch size for embedding generation (default 32)
    pub batch_size: usize,
    /// Enable/disable embedding generation (default true)
    pub enabled: bool,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        let model_dir = dirs::data_local_dir()
            .map(|d| {
                d.join("bpcontext")
                    .join("models")
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_else(|| "~/.local/share/bpcontext/models".to_string());

        Self {
            model: "all-MiniLM-L6-v2".to_string(),
            model_dir,
            batch_size: 32,
            enabled: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Estimated context window budget in tokens (default 200,000)
    pub budget_tokens: u64,
    /// Minutes before a returned chunk is considered stale (default 30)
    pub stale_threshold_minutes: u64,
    /// Relevance weight for recency (default 0.4)
    #[serde(default = "default_recency_weight")]
    pub recency_weight: f64,
    /// Relevance weight for access frequency (default 0.3)
    #[serde(default = "default_frequency_weight")]
    pub frequency_weight: f64,
    /// Relevance weight for staleness penalty (default 0.3)
    #[serde(default = "default_staleness_weight")]
    pub staleness_weight: f64,
}

fn default_recency_weight() -> f64 {
    0.4
}
fn default_frequency_weight() -> f64 {
    0.3
}
fn default_staleness_weight() -> f64 {
    0.3
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            budget_tokens: 200_000,
            stale_threshold_minutes: 30,
            recency_weight: 0.4,
            frequency_weight: 0.3,
            staleness_weight: 0.3,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    #[serde(default = "default_knowledge_enabled")]
    pub enabled: bool,
    /// Override the default knowledge DB location.
    #[serde(default)]
    pub db_path: Option<String>,
    /// Minutes before a source is considered stale and re-synced (default 60).
    #[serde(default = "default_sync_stale_minutes")]
    pub sync_stale_minutes: u64,
    /// Sources defined in config (synced automatically on first knowledge tool call).
    #[serde(default)]
    pub sources: Vec<KnowledgeSourceConfig>,
}

fn default_knowledge_enabled() -> bool {
    true
}

fn default_sync_stale_minutes() -> u64 {
    60
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: None,
            sync_stale_minutes: 60,
            sources: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSourceConfig {
    pub label: String,
    pub path: String,
    pub glob: Option<String>,
    #[serde(default)]
    pub enrichments: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                max_stdout_bytes: 102_400,
                head_ratio: 0.6,
                preview_threshold_bytes: default_preview_threshold_bytes(),
            },
            search: SearchConfig {
                default_limit: 15,
                throttle_max: 8,
                throttle_window_secs: 60,
                snippet_bytes: default_snippet_bytes(),
                secondary_snippet_bytes: default_secondary_snippet_bytes(),
                top_result_count: default_top_result_count(),
                vector_weight: 1.0,
                keyword_weight: 1.0,
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
            cleanup: CleanupConfig { stale_db_days: 14 },
            embeddings: EmbeddingsConfig::default(),
            context: ContextConfig::default(),
            knowledge: KnowledgeConfig::default(),
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
        let config: Config = toml::from_str(&content).context("Failed to parse config.toml")?;
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
