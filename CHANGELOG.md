# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2025-04-02

### Added

- CLI with commands: `serve`, `hook`, `execute`, `search`, `index`, `fetch`, `promote`, `stats`, `sources`
- MCP server over JSON-RPC stdio with 8 tools (`bpx_execute`, `bpx_execute_file`, `bpx_batch_execute`, `bpx_search`, `bpx_fetch_and_index`, `bpx_index`, `bpx_promote`, `bpx_stats`)
- SQLite FTS5 indexing with semantic chunking
- Multi-layer search: BM25, trigram, and fuzzy matching
- Head/tail preview generation with configurable ratio
- URL fetching with HTML-to-markdown conversion and caching
- Claude Code hook support (pretooluse, posttooluse, precompact)
- Session-scoped databases with automatic cleanup
- Context savings tracking and stats
- TOML configuration with `--init` scaffolding
- Obsidian/TaskVault integration via `promote` command
