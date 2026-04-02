mod cli;
mod config;
mod db;
mod executor;
mod fetch;
mod hooks;
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
        Some(Commands::Serve) => {
            mcp::server::run(&project_dir())
        }

        Some(Commands::Hook { hook_type }) => {
            hooks::dispatch(&hook_type)
        }

        Some(Commands::Execute { command, label, timeout_ms }) => {
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

        Some(Commands::Search { query, source, content_type, limit }) => {
            let _config = Config::load()?;
            let store = ContentStore::open(&project_dir())?;

            stats::record_search();
            let results = store.search(
                &query,
                limit,
                source.as_deref(),
                content_type.as_deref(),
            )?;

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

        Some(Commands::Index { label, content, content_type }) => {
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

        Some(Commands::Promote { query, name, project }) => {
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
            println!("  Savings ratio:      {:.1}x", session_stats.savings_ratio());
            println!("  Est. tokens saved:  {}", session_stats.tokens_saved());
            println!();
            println!("{}", "Content Store:".bold());
            println!("  Total indexed:      {} bytes", total_indexed);
            println!("  Sources:            {}", sources.len());
            println!("  DB size:            {} KB", db_size / 1024);
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

        None => {
            // Default: show stats
            println!("{}", "bpcontext - Context window optimization for Claude Code".bold());
            println!("Run 'bpcontext --help' for usage.");
            Ok(())
        }
    }
}
