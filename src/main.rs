mod cli;
mod config;
mod context;
mod db;
mod embedder;
mod executor;
mod fetch;
mod hooks;
mod indexdir;
mod mcp;
mod promote;
mod session;
mod stats;
mod store;
mod truncate;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use cli::{Cli, Commands};
use config::Config;
use embedder::Embed;
use store::ContentStore;

fn project_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.init {
        return Config::init_default();
    }

    match cli.command {
        Some(Commands::Serve) => mcp::server::run(&project_dir()),

        Some(Commands::Hook { hook_type }) => hooks::dispatch(&hook_type),

        Some(Commands::Execute {
            command,
            label,
            timeout_ms,
        }) => {
            let config = Config::load()?;
            let store = ContentStore::open(&project_dir())?;
            let label = label.as_deref().unwrap_or(&command);

            stats::record_command();
            let result = executor::execute_command(
                &command,
                config.general.max_stdout_bytes,
                config.general.head_ratio,
                Some(Duration::from_millis(timeout_ms)),
            )?;

            let full_output = if result.stderr.is_empty() {
                result.stdout.clone()
            } else {
                format!("{}\n--- stderr ---\n{}", result.stdout, result.stderr)
            };

            let index_result = store.index(label, &full_output, None)?;
            stats::record_indexed(full_output.len() as u64);

            let preview = truncate::preview(&full_output, 3072);
            stats::record_returned(preview.len() as u64);

            println!("{preview}");
            println!(
                "\n{} {} chunks indexed (label: \"{label}\")",
                "->".green(),
                index_result.chunk_count
            );
            Ok(())
        }

        Some(Commands::Search {
            query,
            source,
            content_type,
            limit,
        }) => {
            let _config = Config::load()?;
            let store = ContentStore::open(&project_dir())?;

            stats::record_search();
            let results =
                store.search(&query, limit, source.as_deref(), content_type.as_deref())?;

            if results.is_empty() {
                println!("{}", "No results found.".yellow());
            } else {
                for result in &results {
                    let snippet = truncate::preview(&result.content, 500);
                    println!(
                        "{} {} (source: {}, type: {}, score: {:.4})",
                        "->".green(),
                        result.title.bold(),
                        result.source.cyan(),
                        result.content_type.dimmed(),
                        result.score
                    );
                    println!("{snippet}\n");
                }
                println!("{} results", results.len());
            }
            Ok(())
        }

        Some(Commands::Index {
            label,
            content,
            content_type,
        }) => {
            let store = ContentStore::open(&project_dir())?;

            let text = match content {
                Some(c) => c,
                None => {
                    let mut buf = String::new();
                    std::io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };

            let index_result = store.index(&label, &text, content_type.as_deref())?;
            stats::record_indexed(text.len() as u64);

            println!(
                "{} {} chunks indexed (label: \"{label}\")",
                "->".green(),
                index_result.chunk_count
            );
            Ok(())
        }

        Some(Commands::Fetch { url, label }) => {
            let store = ContentStore::open(&project_dir())?;
            let label = label.as_deref().unwrap_or(&url);

            let content = fetch::fetch_and_convert(&url)?;
            let index_result = store.index(label, &content, Some("prose"))?;
            stats::record_indexed(content.len() as u64);

            let preview = truncate::preview(&content, 3072);
            stats::record_returned(preview.len() as u64);

            println!("{preview}");
            println!(
                "\n{} {} chunks indexed from URL (label: \"{label}\")",
                "->".green(),
                index_result.chunk_count
            );
            Ok(())
        }

        Some(Commands::Promote {
            query,
            name,
            project,
        }) => {
            let config = Config::load()?;
            let store = ContentStore::open(&project_dir())?;

            let results = store.search(&query, 5, None, None)?;
            if results.is_empty() {
                anyhow::bail!("No content found matching: {query}");
            }

            let content: String = results
                .iter()
                .take(3)
                .map(|r| format!("### {}\n\n{}\n", r.title, r.content))
                .collect();

            let note_path = promote::promote_to_obsidian(&config, &name, &project, &content)?;
            println!("{} Promoted to: {}", "->".green(), note_path);
            Ok(())
        }

        Some(Commands::Stats) => {
            let config = Config::load()?;
            let store = ContentStore::open(&project_dir())?;
            let session_stats = stats::get_stats();
            let total_indexed = store.total_bytes_indexed()?;
            let sources = store.list_sources()?;

            let db_path = db::content_db_path(&project_dir())?;
            let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

            println!("{}", "bpcontext Stats".bold());
            println!();
            println!("{}", "Session:".bold());
            println!("  Commands executed:  {}", session_stats.commands_executed);
            println!("  Searches performed: {}", session_stats.searches_performed);
            println!("  Bytes indexed:      {}", session_stats.bytes_indexed);
            println!("  Bytes returned:     {}", session_stats.bytes_returned);
            println!("  Bytes visible:      {}", session_stats.bytes_visible);
            println!(
                "  Visibility ratio:   {:.1}%",
                session_stats.visibility_ratio() * 100.0
            );
            println!(
                "  Savings ratio:      {:.1}x",
                session_stats.savings_ratio()
            );
            println!("  Est. tokens saved:  {}", session_stats.tokens_saved());
            println!();
            println!("{}", "Content Store:".bold());
            println!("  Total indexed:      {} bytes", total_indexed);
            println!("  Sources:            {}", sources.len());
            println!("  DB size:            {} KB", db_size / 1024);

            // Context budget summary
            if let Ok(session_store) = session::SessionStore::open(&project_dir()) {
                let conn = session_store.conn();
                let ctx_tokens = context::ledger::total_tokens(conn).unwrap_or(0);
                let ctx_sources = context::ledger::source_count(conn).unwrap_or(0);
                let budget = config.context.budget_tokens;
                let pct = if budget > 0 {
                    ctx_tokens as f64 / budget as f64 * 100.0
                } else {
                    0.0
                };

                println!();
                println!("{}", "Context Budget:".bold());
                println!(
                    "  Tokens used:        {} / {} ({:.1}%)",
                    ctx_tokens, budget, pct
                );
                println!("  Sources tracked:    {}", ctx_sources);
            }

            Ok(())
        }

        Some(Commands::Sources) => {
            let store = ContentStore::open(&project_dir())?;
            let sources = store.list_sources()?;

            if sources.is_empty() {
                println!("{}", "No sources indexed.".yellow());
            } else {
                for source in &sources {
                    println!(
                        "{} {} ({} chunks, indexed {})",
                        "->".green(),
                        source.label.bold(),
                        source.chunk_count,
                        source.indexed_at.dimmed()
                    );
                }
            }
            Ok(())
        }

        Some(Commands::EmbedBackfill) => {
            let config = Config::load()?;
            if !config.embeddings.enabled {
                println!("{}", "Embeddings disabled in config.".yellow());
                return Ok(());
            }

            let model_dir =
                PathBuf::from(&config.embeddings.model_dir).join(&config.embeddings.model);
            let emb = embedder::Embedder::new(&model_dir)?;

            let store = ContentStore::open(&project_dir())?;
            let conn = store.conn();

            // Find chunks that don't have embeddings yet
            let mut stmt = conn.prepare(
                "SELECT c.rowid, c.title, c.content
                 FROM chunks c
                 LEFT JOIN chunk_embeddings ce ON c.rowid = ce.chunk_rowid
                 WHERE ce.chunk_rowid IS NULL",
            )?;

            let missing: Vec<(i64, String, String)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
                .filter_map(|r| r.ok())
                .collect();

            if missing.is_empty() {
                println!("{}", "All chunks already have embeddings.".green());
                return Ok(());
            }

            println!(
                "Found {} chunks without embeddings. Generating...",
                missing.len()
            );

            let batch_size = config.embeddings.batch_size;
            let dim = emb.dim() as i32;
            let mut total = 0usize;

            for batch in missing.chunks(batch_size) {
                let texts: Vec<String> = batch
                    .iter()
                    .map(|(_, title, content)| format!("{title} {content}"))
                    .collect();
                let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

                let embeddings = emb.embed_batch(&text_refs)?;

                for ((rowid, _, _), embedding) in batch.iter().zip(embeddings.iter()) {
                    let blob = embedder::embedding_to_bytes(embedding);
                    conn.execute(
                        "INSERT OR REPLACE INTO chunk_embeddings (chunk_rowid, embedding, dim) VALUES (?1, ?2, ?3)",
                        rusqlite::params![rowid, blob, dim],
                    )?;
                }

                total += batch.len();
                println!("  {} {} / {}", "->".green(), total, missing.len());
            }

            println!(
                "{} Backfilled embeddings for {} chunks.",
                "->".green(),
                total
            );
            Ok(())
        }

        Some(Commands::ContextStatus) => {
            let session_store = session::SessionStore::open(&project_dir())?;
            let config = Config::load()?;
            let conn = session_store.conn();

            let total_tokens = context::ledger::total_tokens(conn)?;
            let n_sources = context::ledger::source_count(conn)?;
            let sources = context::ledger::source_breakdown(conn)?;

            let pct = if config.context.budget_tokens > 0 {
                total_tokens as f64 / config.context.budget_tokens as f64 * 100.0
            } else {
                0.0
            };

            println!("{}", "Context Budget".bold());
            println!();
            println!(
                "  Tokens used: {} / {} ({:.1}%)",
                total_tokens, config.context.budget_tokens, pct
            );
            println!("  Sources tracked: {}", n_sources);
            println!();

            if sources.is_empty() {
                println!("{}", "  No sources tracked in this session.".dimmed());
            } else {
                println!("{}", "Per-Source Breakdown:".bold());
                for source in &sources {
                    println!(
                        "  {} {} — {}t, {} accesses, last {}",
                        "->".green(),
                        source.label.bold(),
                        source.tokens,
                        source.access_count,
                        source.last_access.dimmed()
                    );
                }
            }

            // Show relevance scores if there are enough sources
            if sources.len() >= 2 {
                let weights = context::advisor::RelevanceWeights {
                    recency: config.context.recency_weight,
                    frequency: config.context.frequency_weight,
                    staleness: config.context.staleness_weight,
                };
                let scored = context::advisor::score_sources(
                    conn,
                    &weights,
                    config.context.stale_threshold_minutes,
                )?;

                println!();
                println!("{}", "Relevance Scores:".bold());
                for s in &scored {
                    let bar = if s.relevance > 0.6 {
                        "high".green()
                    } else if s.relevance > 0.3 {
                        "mid".yellow()
                    } else {
                        "low".red()
                    };
                    println!(
                        "  {} {} — {:.2} ({})",
                        "->".green(),
                        s.label.bold(),
                        s.relevance,
                        bar
                    );
                }
            }

            Ok(())
        }

        Some(Commands::IndexDir {
            path,
            glob,
            label_prefix,
        }) => {
            let store = ContentStore::open(&project_dir())?;
            let result = indexdir::index_directory(&store, &path, glob.as_deref(), &label_prefix)?;
            stats::record_indexed(result.total_bytes);
            println!(
                "{} Indexed {} files ({} chunks, {} skipped)",
                "->".green(),
                result.files_indexed,
                result.total_chunks,
                result.files_skipped,
            );
            for err in &result.walk_errors {
                eprintln!("{} {}", "warn:".yellow(), err);
            }
            Ok(())
        }

        None => {
            // Default: show stats
            println!(
                "{}",
                "bpcontext - Context window optimization for Claude Code".bold()
            );
            println!("Run 'bpcontext --help' for usage.");
            Ok(())
        }
    }
}
