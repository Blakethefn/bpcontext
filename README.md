# bpcontext

**Context window optimizer for AI coding agents.**

LLM coding agents burn through their context window reading large files, command output, and web pages. bpcontext sits between the agent and those sources — it captures the full content, indexes it into a local SQLite FTS5 database, and returns only a compact preview. The agent can then search the indexed content on demand, pulling back exactly the chunks it needs instead of holding everything in context.

In practice, this cuts context usage by 50-80% for file reads, command output, and web fetches — without losing access to any of the content.

**v2** adds semantic search (local embeddings via Candle) and a smart context manager that tracks utilization and surfaces optimization advice inline.

## The Problem

A typical Claude Code session might:
- Read a 2,600-line file → **55k tokens** into context
- Run `git log --oneline -100` → thousands of tokens for a quick lookup
- Fetch API docs from a URL → entire page dumped into context

These add up fast. By the time you're doing real work, half your context window is occupied by reference material you've already read.

## How bpcontext Solves It

1. **Capture** — Command output, file content, or web pages are captured in full
2. **Chunk** — Content is split into semantically meaningful sections
3. **Index** — Chunks are stored in SQLite with FTS5 full-text indexing
4. **Preview** — A truncated head/tail preview is returned to the agent (configurable ratio)
5. **Search** — The agent retrieves specific chunks on demand via multi-layer search (BM25 + trigram + fuzzy + vector similarity)
6. **Track** — A context ledger tracks what's been returned and alerts when utilization crosses thresholds

The full content is always available. The agent just doesn't need to hold all of it in memory at once.

## Installation

```bash
# CPU only
cargo build --release

# With CUDA support (requires CUDA toolkit)
cargo build --release --features cuda
```

The binary is at `target/release/bpcontext`. Generate a default config:

```bash
bpcontext --init
```

Config location: `~/.config/bpcontext/config.toml`

### First-run model download

On first use, bpcontext downloads the `all-MiniLM-L6-v2` embedding model (~80MB) from Hugging Face to `~/.local/share/bpcontext/models/`. This is a one-time download — no API keys or accounts needed. If the download fails (e.g., no internet), search falls back to keyword-only mode.

## Quick Start

### As an MCP Server (recommended)

Add to your Claude Code `.mcp.json`:

```json
{
  "mcpServers": {
    "bpcontext": {
      "command": "/path/to/bpcontext",
      "args": ["serve"]
    }
  }
}
```

Then configure Claude Code to prefer bpcontext tools over built-in tools. Add to your `CLAUDE.md`:

```markdown
# bpcontext Tool Routing

- **`bpx_execute`** instead of `Bash` for commands producing >20 lines
- **`bpx_execute_file`** instead of `Read` when analyzing a file (not editing it)
- **`bpx_batch_execute`** instead of multiple Bash/Read/Grep calls when exploring
- **`bpx_fetch_and_index`** instead of `WebFetch` for any URL
```

### As a CLI

```bash
# Run a command and index its output
bpcontext execute "git log --oneline -50"

# Search indexed content
bpcontext search "authentication"

# Filter by source or content type
bpcontext search "error" --source "git log" --content-type code --limit 5

# Index raw text from stdin
echo "some notes" | bpcontext index "my-notes"

# Fetch and index a web page
bpcontext fetch https://docs.rs/some-crate

# Check context savings
bpcontext stats

# List what's been indexed this session
bpcontext sources

# Backfill embeddings for previously indexed content
bpcontext embed-backfill

# Show context budget and per-source breakdown
bpcontext context-status
```

## MCP Tools

| Tool | What it does |
|------|-------------|
| `bpx_execute` | Run a shell command, index the output, return a preview |
| `bpx_execute_file` | Read and index a file with optional processing |
| `bpx_batch_execute` | Run multiple commands + search queries in one call |
| `bpx_search` | Search session index and knowledge store (BM25, trigram, fuzzy, vector similarity, merged via RRF) |
| `bpx_fetch_and_index` | Fetch a URL, convert HTML to markdown, index it |
| `bpx_index` | Index raw text for later search |
| `bpx_index_dir` | Index all files in a directory for the current session |
| `bpx_promote` | Export search results to an Obsidian note via TaskVault |
| `bpx_stats` | Show context savings metrics for the session |
| `bpx_context_status` | Show context budget and per-source breakdown |
| `bpx_read_chunks` | Read specific chunks by ID |

**Knowledge store tools (persistent across sessions):**

| Tool | What it does |
|------|-------------|
| `bpx_knowledge_add` | Register a directory as a persistent knowledge source and run initial sync |
| `bpx_knowledge_sync` | Incrementally re-index changed files across all registered sources |
| `bpx_knowledge_status` | List registered sources, chunk counts, and last sync time |
| `bpx_knowledge_remove` | Unregister a knowledge source and delete all its indexed content |

## Knowledge Store (RAG)

The knowledge store is a persistent RAG layer that survives across sessions. Unlike the session-scoped index (which is rebuilt each time), the knowledge store registers directories as durable sources and only re-indexes files that have changed (incremental sync via SHA-256 content hash).

This means you can register your project's source directory once and search it in any future session without re-indexing.

### CLI

```bash
# Register a directory as a knowledge source
bpcontext knowledge add /path/to/project/src --label myproject

# Filter by file type
bpcontext knowledge add /path/to/docs --label mydocs --glob "**/*.md"

# Enable enrichments (for Obsidian vaults)
bpcontext knowledge add /path/to/vault --label vault --enrichments frontmatter,wikilinks,folder_tags

# Re-sync all sources (re-indexes changed files only)
bpcontext knowledge sync

# Re-sync a specific source
bpcontext knowledge sync --label myproject

# Check status
bpcontext knowledge status

# Search
bpcontext knowledge search "authentication flow"

# Remove a source
bpcontext knowledge remove --label myproject
```

### Session index vs. knowledge store

| | Session index | Knowledge store |
|---|---|---|
| Lifetime | Current session only | Persistent across sessions |
| How populated | `bpx_execute`, `bpx_index`, `bpx_index_dir` | `bpx_knowledge_add` / `bpcontext knowledge add` |
| Re-index cost | Full re-index every session | Incremental — only changed files |
| Use case | One-off exploration | Frequently referenced codebases and docs |

`bpx_search` queries **both** layers simultaneously and merges results via RRF, so you get session-context results and persistent knowledge in a single call.

### Enrichments

Enrichments extract structured metadata at index time for richer filtering:

| Enrichment | What it extracts |
|---|---|
| `frontmatter` | YAML frontmatter fields (e.g., `status`, `type`, `tags`) |
| `wikilinks` | Outgoing `[[wikilinks]]` from Obsidian notes |
| `folder_tags` | Parent folder names as implicit tags |

Enrichments are stored alongside each chunk and available for metadata filtering in search results.

### Data storage

The knowledge store uses a single global database separate from per-session content databases:

- **Knowledge DB:** `~/.local/share/bpcontext/knowledge.db` — registered sources, file hashes, chunks, and embeddings

## Claude Code Hooks

bpcontext can also run as Claude Code hooks to automatically intercept tool output:

- **pretooluse** — Intercepts before tool execution
- **posttooluse** — Processes and compresses tool output after execution
- **precompact** — Runs before context compaction

## Semantic Search

bpcontext v2 adds a fourth search layer: vector similarity using local embeddings.

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80MB)
- **Inference:** [Candle](https://github.com/huggingface/candle) (Rust ML framework) — runs on CPU by default, CUDA optional
- **At index time:** each chunk is embedded and stored as a BLOB in SQLite
- **At search time:** the query is embedded and compared against all stored vectors via dot product (brute-force, <1ms for typical session sizes)
- **Fusion:** vector results are merged into the existing RRF (Reciprocal Rank Fusion) pipeline alongside BM25, trigram, and fuzzy results

This means `bpx_search(["authentication flow"])` will find chunks about "login session", "JWT validation", and "token refresh" — even though none of those words appear in the query.

Weights are configurable:
```toml
[search]
vector_weight = 1.0    # multiplier for semantic results
keyword_weight = 1.0   # multiplier for keyword results
```

## Context Manager

bpcontext tracks what it returns to the agent and surfaces optimization advice inline — no extra tool calls needed.

**How it works:** after every tool response, the context ledger checks utilization against the configured budget. When a threshold is crossed, an alert is appended to the response:

| Utilization | What the agent sees |
|-------------|---------------------|
| 40% | Token count and source count |
| 60% | Top consumers, nudge toward search |
| 70% | Stale sources to drop, relevance scores |
| 80% | Explicit compact recommendation with keep/drop lists |
| 90% | Critical alert with remaining tokens |

Each threshold fires once per session. The precompact hook also includes keep/drop recommendations.

## Configuration

```toml
[general]
max_stdout_bytes = 102400   # max bytes captured per command
head_ratio = 0.6            # fraction of preview from head vs tail

[search]
default_limit = 10
throttle_max = 8            # max searches per window
throttle_window_secs = 60
vector_weight = 1.0         # weight for semantic search in RRF fusion
keyword_weight = 1.0        # weight for keyword search in RRF fusion

[fetch]
cache_ttl_hours = 24        # cache fetched URLs

[embeddings]
model = "all-MiniLM-L6-v2"
model_dir = "~/.local/share/bpcontext/models"
batch_size = 32
enabled = true              # set to false to disable embeddings entirely

[context]
budget_tokens = 200000      # estimated context window budget
stale_threshold_minutes = 30

[integration]
taskvault_bin = "/path/to/taskvault"  # optional: for promote command
vault_path = "/path/to/obsidian_docs" # optional: Obsidian vault path

[cleanup]
stale_db_days = 14          # auto-cleanup old session databases
```

## Architecture

```
src/
  cli.rs          — CLI argument parsing (clap)
  config.rs       — TOML config loading
  db.rs           — SQLite connection and session DB management
  fetch.rs        — URL fetching and HTML-to-markdown conversion
  indexdir.rs     — Directory indexing for bpx_index_dir
  promote.rs      — Export results to Obsidian via TaskVault
  stats.rs        — Context savings tracking
  truncate.rs     — Head/tail preview generation
  context/        — Smart context manager (ledger, relevance scoring, alerts)
  embedder/       — Local embedding model (Candle + all-MiniLM-L6-v2)
  executor/       — Command execution and output capture
  hooks/          — Claude Code hook handlers (pre/post tool use, precompact)
  knowledge/      — Persistent knowledge store (RAG): source registry, incremental sync, enrichments, search
  mcp/            — MCP server (JSON-RPC over stdio)
  session/        — Session lifecycle and event tracking
  store/          — Chunking, FTS5 indexing, and multi-layer search (BM25 + trigram + fuzzy + vector)
```

## Data Storage

All runtime data is stored under XDG-standard directories:

- **Content DBs:** `~/.local/share/bpcontext/content/{hash}.db` — per-project FTS5 indexes + embeddings (session-scoped)
- **Session DBs:** `~/.local/share/bpcontext/sessions/{hash}.db` — events + context ledger
- **Knowledge DB:** `~/.local/share/bpcontext/knowledge.db` — persistent knowledge store (source registry, file hashes, chunks, embeddings)
- **Models:** `~/.local/share/bpcontext/models/` — downloaded embedding weights
- **Config:** `~/.config/bpcontext/config.toml`

## Requirements

- Rust 1.70+
- SQLite (bundled via rusqlite)
- For CUDA: CUDA toolkit 12.0+ and a compatible GPU

## License

MIT
