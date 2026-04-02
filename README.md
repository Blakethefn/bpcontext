# bpcontext

**Context window optimizer for AI coding agents.**

LLM coding agents burn through their context window reading large files, command output, and web pages. bpcontext sits between the agent and those sources — it captures the full content, indexes it into a local SQLite FTS5 database, and returns only a compact preview. The agent can then search the indexed content on demand, pulling back exactly the chunks it needs instead of holding everything in context.

In practice, this cuts context usage by 50-80% for file reads, command output, and web fetches — without losing access to any of the content.

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
5. **Search** — The agent retrieves specific chunks on demand via multi-layer search (BM25 + trigram + fuzzy)

The full content is always available. The agent just doesn't need to hold all of it in memory at once.

## Installation

```bash
cargo build --release
```

The binary is at `target/release/bpcontext`. Generate a default config:

```bash
bpcontext --init
```

Config location: `~/.config/bpcontext/config.toml`

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
```

## MCP Tools

| Tool | What it does |
|------|-------------|
| `bpx_execute` | Run a shell command, index the output, return a preview |
| `bpx_execute_file` | Read and index a file with optional processing |
| `bpx_batch_execute` | Run multiple commands + search queries in one call |
| `bpx_search` | Search indexed content (BM25, trigram, fuzzy) |
| `bpx_fetch_and_index` | Fetch a URL, convert HTML to markdown, index it |
| `bpx_index` | Index raw text for later search |
| `bpx_promote` | Export search results to an Obsidian note via TaskVault |
| `bpx_stats` | Show context savings metrics for the session |

## Claude Code Hooks

bpcontext can also run as Claude Code hooks to automatically intercept tool output:

- **pretooluse** — Intercepts before tool execution
- **posttooluse** — Processes and compresses tool output after execution
- **precompact** — Runs before context compaction

## Configuration

```toml
[general]
max_stdout_bytes = 102400   # max bytes captured per command
head_ratio = 0.6            # fraction of preview from head vs tail

[search]
default_limit = 10
throttle_max = 8            # max searches per window
throttle_window_secs = 60

[fetch]
cache_ttl_hours = 24        # cache fetched URLs

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
  promote.rs      — Export results to Obsidian via TaskVault
  stats.rs        — Context savings tracking
  truncate.rs     — Head/tail preview generation
  executor/       — Command execution and output capture
  hooks/          — Claude Code hook handlers (pre/post tool use)
  mcp/            — MCP server (JSON-RPC over stdio)
  session/        — Session lifecycle and event tracking
  store/          — Chunking, FTS5 indexing, and multi-layer search
```

## Requirements

- Rust 1.70+
- SQLite (bundled via rusqlite)

## License

MIT
