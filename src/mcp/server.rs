use anyhow::Result;
use serde_json::{json, Value};
use std::io::{self, BufReader};
use std::path::Path;
use std::time::Duration;

use super::tools;
use super::transport;
use crate::config::Config;
use crate::executor;
use crate::fetch;
use crate::promote;
use crate::stats;
use crate::store::ContentStore;
use crate::truncate;

/// Run the MCP server loop over stdio
pub fn run(project_dir: &Path) -> Result<()> {
    let config = Config::load()?;
    let store = ContentStore::open(project_dir)?;

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut stdout = io::stdout().lock();

    loop {
        let message = match transport::read_message(&mut reader)? {
            Some(msg) => msg,
            None => break, // EOF, parent closed stdin
        };

        let request: Value = serde_json::from_str(&message)?;
        let method = request["method"].as_str().unwrap_or("");
        let id = request.get("id").cloned();

        let response = match method {
            "initialize" => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "bpcontext",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            })),

            "notifications/initialized" => None, // No response needed

            "tools/list" => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": tools::tool_definitions()
            })),

            "tools/call" => {
                let tool_name = request["params"]["name"].as_str().unwrap_or("");
                let arguments = &request["params"]["arguments"];
                let result = handle_tool_call(tool_name, arguments, &config, &store, project_dir);

                match result {
                    Ok(content) => Some(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": content
                            }]
                        }
                    })),
                    Err(e) => Some(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": format!("Error: {e}")
                            }],
                            "isError": true
                        }
                    })),
                }
            }

            _ => {
                // Unknown method
                if id.is_some() {
                    Some(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found: {method}")
                        }
                    }))
                } else {
                    None // Notification, no response
                }
            }
        };

        if let Some(resp) = response {
            transport::write_message(&mut stdout, &resp.to_string())?;
        }
    }

    Ok(())
}

/// Dispatch a tool call to the appropriate handler
fn handle_tool_call(
    tool_name: &str,
    args: &Value,
    config: &Config,
    store: &ContentStore,
    project_dir: &Path,
) -> Result<String> {
    match tool_name {
        "bpx_execute" => handle_execute(args, config, store),
        "bpx_batch_execute" => handle_batch_execute(args, config, store),
        "bpx_search" => handle_search(args, config, store),
        "bpx_execute_file" => handle_execute_file(args, config, store),
        "bpx_fetch_and_index" => handle_fetch_and_index(args, config, store),
        "bpx_index" => handle_index(args, store),
        "bpx_promote" => handle_promote(args, config, store),
        "bpx_stats" => handle_stats(store, project_dir),
        _ => anyhow::bail!("Unknown tool: {tool_name}"),
    }
}

fn handle_execute(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let command = args["command"].as_str().unwrap_or("");
    let label = args["label"].as_str().unwrap_or(command);
    let timeout_ms = args["timeout_ms"].as_u64().unwrap_or(120_000);

    stats::record_command();

    let result = executor::execute_command(
        command,
        config.general.max_stdout_bytes,
        config.general.head_ratio,
        Some(Duration::from_millis(timeout_ms)),
    )?;

    let full_output = if result.stderr.is_empty() {
        result.stdout.clone()
    } else {
        format!("{}\n--- stderr ---\n{}", result.stdout, result.stderr)
    };

    // Index the output
    let index_result = store.index(label, &full_output, None)?;
    stats::record_indexed(full_output.len() as u64);

    // Return a preview
    let preview = truncate::preview(&full_output, 3072);
    stats::record_returned(preview.len() as u64);

    Ok(format!(
        "{preview}\n\n---\n{} chunks indexed (label: \"{label}\"). Use bpx_search to find specific content.",
        index_result.chunk_count
    ))
}

fn handle_batch_execute(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let commands = args["commands"].as_array().unwrap_or(&Vec::new()).clone();
    let queries = args["queries"].as_array().unwrap_or(&Vec::new()).clone();

    let mut output = String::new();
    let mut section_labels = Vec::new();

    // Execute all commands
    for cmd in &commands {
        let label = cmd["label"].as_str().unwrap_or("unlabeled");
        let command = cmd["command"].as_str().unwrap_or("");

        stats::record_command();

        let result = executor::execute_command(
            command,
            config.general.max_stdout_bytes,
            config.general.head_ratio,
            Some(Duration::from_secs(60)),
        )?;

        let full = if result.stderr.is_empty() {
            result.stdout.clone()
        } else {
            format!("{}\n--- stderr ---\n{}", result.stdout, result.stderr)
        };

        store.index(label, &full, None)?;
        stats::record_indexed(full.len() as u64);
        section_labels.push(label.to_string());
    }

    output.push_str(&format!("## Indexed Sections\n{}\n\n", section_labels.join(", ")));

    // Run all queries
    for query_val in &queries {
        let query = query_val.as_str().unwrap_or("");
        if query.is_empty() {
            continue;
        }

        stats::record_search();
        let results = store.search(query, config.search.default_limit, None, None)?;

        output.push_str(&format!("### Query: \"{query}\"\n"));
        if results.is_empty() {
            output.push_str("No results found.\n\n");
        } else {
            for result in &results {
                let snippet = truncate::preview(&result.content, 500);
                output.push_str(&format!(
                    "**[{}]** (source: {}, score: {:.4})\n{snippet}\n\n",
                    result.title, result.source, result.score
                ));
            }
        }
    }

    stats::record_returned(output.len() as u64);
    Ok(output)
}

fn handle_search(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let queries = args["queries"].as_array().unwrap_or(&Vec::new()).clone();
    let source = args["source"].as_str();
    let content_type = args["content_type"].as_str();
    let limit = args["limit"].as_u64().unwrap_or(config.search.default_limit as u64) as u32;

    let mut output = String::new();

    for query_val in &queries {
        let query = query_val.as_str().unwrap_or("");
        if query.is_empty() {
            continue;
        }

        stats::record_search();
        let results = store.search(query, limit, source, content_type)?;

        output.push_str(&format!("### \"{query}\"\n"));
        if results.is_empty() {
            output.push_str("No results.\n\n");
        } else {
            for result in &results {
                let snippet = truncate::preview(&result.content, 800);
                output.push_str(&format!(
                    "**[{}]** (source: {}, type: {}, score: {:.4})\n{snippet}\n\n",
                    result.title, result.source, result.content_type, result.score
                ));
            }
        }
    }

    stats::record_returned(output.len() as u64);
    Ok(output)
}

fn handle_execute_file(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let path = args["path"].as_str().unwrap_or("");
    let code = args["code"].as_str();

    let content = if let Some(processing_code) = code {
        // Pipe file through processing command
        let command = format!("cat '{}' | {}", path.replace('\'', "'\\''"), processing_code);
        let result = executor::execute_command(
            &command,
            config.general.max_stdout_bytes,
            config.general.head_ratio,
            Some(Duration::from_secs(60)),
        )?;
        result.stdout
    } else {
        std::fs::read_to_string(path)?
    };

    stats::record_command();

    let label = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);

    let index_result = store.index(label, &content, None)?;
    stats::record_indexed(content.len() as u64);

    let preview = truncate::preview(&content, 3072);
    stats::record_returned(preview.len() as u64);

    Ok(format!(
        "{preview}\n\n---\n{} chunks indexed (label: \"{label}\"). Use bpx_search to find specific content.",
        index_result.chunk_count
    ))
}

fn handle_fetch_and_index(args: &Value, _config: &Config, store: &ContentStore) -> Result<String> {
    let url = args["url"].as_str().unwrap_or("");
    let label = args["label"].as_str().unwrap_or(url);

    let content = fetch::fetch_and_convert(url)?;
    let index_result = store.index(label, &content, Some("prose"))?;
    stats::record_indexed(content.len() as u64);

    let preview = truncate::preview(&content, 3072);
    stats::record_returned(preview.len() as u64);

    Ok(format!(
        "{preview}\n\n---\n{} chunks indexed from URL (label: \"{label}\"). Use bpx_search to find specific content.",
        index_result.chunk_count
    ))
}

fn handle_index(args: &Value, store: &ContentStore) -> Result<String> {
    let content = args["content"].as_str().unwrap_or("");
    let label = args["label"].as_str().unwrap_or("manual");
    let content_type = args["content_type"].as_str();

    let index_result = store.index(label, content, content_type)?;
    stats::record_indexed(content.len() as u64);

    Ok(format!(
        "{} chunks indexed (label: \"{label}\"). Use bpx_search to query this content.",
        index_result.chunk_count
    ))
}

fn handle_promote(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    let name = args["name"].as_str().unwrap_or("");
    let project = args["project"].as_str().unwrap_or("");

    // Search for content to promote
    let results = store.search(query, 5, None, None)?;
    if results.is_empty() {
        anyhow::bail!("No content found matching query: {query}");
    }

    // Combine top results into promotion content
    let content: String = results
        .iter()
        .take(3)
        .map(|r| format!("### {}\n\n{}\n", r.title, r.content))
        .collect();

    let note_path = promote::promote_to_obsidian(config, name, project, &content)?;
    Ok(format!("Promoted to obsidian output note: {note_path}"))
}

fn handle_stats(store: &ContentStore, project_dir: &Path) -> Result<String> {
    let session_stats = stats::get_stats();
    let total_indexed = store.total_bytes_indexed()?;
    let sources = store.list_sources()?;

    let db_path = crate::db::content_db_path(project_dir)?;
    let db_size = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(format!(
        "## bpcontext Stats\n\n\
         **Session:**\n\
         - Commands executed: {}\n\
         - Searches performed: {}\n\
         - Bytes indexed (session): {}\n\
         - Bytes returned (session): {}\n\
         - Savings ratio: {:.1}x\n\
         - Est. tokens saved: {}\n\n\
         **Content Store:**\n\
         - Total bytes indexed: {total_indexed}\n\
         - Sources: {}\n\
         - DB size: {} KB\n",
        session_stats.commands_executed,
        session_stats.searches_performed,
        session_stats.bytes_indexed,
        session_stats.bytes_returned,
        session_stats.savings_ratio(),
        session_stats.tokens_saved(),
        sources.len(),
        db_size / 1024,
    ))
}
