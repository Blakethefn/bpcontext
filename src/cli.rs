use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bpcontext")]
#[command(about = "Context window optimization CLI and MCP server for Claude Code")]
#[command(version)]
pub struct Cli {
    /// Initialize default config
    #[arg(long)]
    pub init: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the MCP server (JSON-RPC over stdio)
    Serve,

    /// Run a hook (called by Claude Code)
    Hook {
        /// Hook type: pretooluse, posttooluse, precompact
        hook_type: String,
    },

    /// Execute a command and index the output
    Execute {
        /// Shell command to run
        command: String,
        /// Label for indexing
        #[arg(short, long)]
        label: Option<String>,
        /// Timeout in milliseconds
        #[arg(short, long, default_value = "120000")]
        timeout_ms: u64,
    },

    /// Search indexed content
    Search {
        /// Search query
        query: String,
        /// Filter by source label
        #[arg(short, long)]
        source: Option<String>,
        /// Filter by content type (code/prose)
        #[arg(short = 't', long)]
        content_type: Option<String>,
        /// Max results
        #[arg(short, long, default_value = "10")]
        limit: u32,
    },

    /// Index raw text
    Index {
        /// Label for the content
        label: String,
        /// Content to index (reads from stdin if not provided)
        #[arg(short, long)]
        content: Option<String>,
        /// Content type hint
        #[arg(short = 't', long)]
        content_type: Option<String>,
    },

    /// Fetch a URL, convert to markdown, and index it
    Fetch {
        /// URL to fetch
        url: String,
        /// Label for indexing
        #[arg(short, long)]
        label: Option<String>,
    },

    /// Promote search results to an obsidian output note
    Promote {
        /// Search query to find content
        query: String,
        /// Name for the output note
        #[arg(short, long)]
        name: String,
        /// Obsidian project
        #[arg(short, long)]
        project: String,
    },

    /// Show context savings statistics
    Stats,

    /// List indexed sources
    Sources,

    /// Generate embeddings for chunks that don't have one yet
    EmbedBackfill,

    /// Show context budget and per-source breakdown
    ContextStatus,
}
