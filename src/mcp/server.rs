use anyhow::Result;
use serde_json::{json, Value};
use std::io::{self, BufReader};
use std::path::Path;
use std::time::Duration;

use super::tools;
use super::transport;
use crate::config::Config;
use crate::context::advisor::RelevanceWeights;
use crate::context::ContextManager;
use crate::embedder::Embedder;
use crate::executor;
use crate::fetch;
use crate::indexdir;
use crate::promote;
use crate::session::SessionStore;
use crate::stats;
use crate::store::ContentStore;
use crate::truncate;

/// Run the MCP server loop over stdio
pub fn run(project_dir: &Path) -> Result<()> {
    let config = Config::load()?;
    let mut store = ContentStore::open(project_dir)?;

    // Lazy embedder initialization
    if config.embeddings.enabled {
        let model_dir =
            std::path::PathBuf::from(&config.embeddings.model_dir).join(&config.embeddings.model);
        match Embedder::new(&model_dir) {
            Ok(embedder) => {
                store.set_embedder(std::sync::Arc::new(embedder));
            }
            Err(e) => {
                eprintln!("[bpcontext] embedder unavailable, falling back to keyword search: {e}");
            }
        }
    }

    // Session store + context manager (reset per conversation)
    let session_store = SessionStore::open(project_dir)?;
    let weights = RelevanceWeights {
        recency: config.context.recency_weight,
        frequency: config.context.frequency_weight,
        staleness: config.context.staleness_weight,
    };
    let mut ctx_mgr = ContextManager::new(
        config.context.budget_tokens,
        config.context.stale_threshold_minutes,
        weights,
    );
    // Reset ledger at session start (per-conversation reset)
    ctx_mgr.reset(session_store.conn())?;

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
                let result = handle_tool_call(
                    tool_name,
                    arguments,
                    &config,
                    &store,
                    project_dir,
                    &session_store,
                );

                match result {
                    Ok(mut content) => {
                        // Record what was returned and check for context alerts
                        let label = tool_label(tool_name, arguments);
                        let _ = ctx_mgr.record(session_store.conn(), &label, content.len());
                        if let Ok(Some(alert)) = ctx_mgr.check_alert(session_store.conn()) {
                            content.push_str("\n\n---\n");
                            content.push_str(&alert);
                        }

                        Some(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{
                                    "type": "text",
                                    "text": content
                                }]
                            }
                        }))
                    }
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
    session_store: &SessionStore,
) -> Result<String> {
    match tool_name {
        "bpx_execute" => handle_execute(args, config, store),
        "bpx_batch_execute" => handle_batch_execute(args, config, store),
        "bpx_search" => handle_search(args, config, store),
        "bpx_execute_file" => handle_execute_file(args, config, store),
        "bpx_read_chunks" => handle_read_chunks(args, config, store),
        "bpx_fetch_and_index" => handle_fetch_and_index(args, config, store),
        "bpx_index" => handle_index(args, store),
        "bpx_promote" => handle_promote(args, config, store),
        "bpx_index_dir" => handle_index_dir(args, config, store),
        "bpx_stats" => handle_stats(config, store, project_dir, session_store),
        "bpx_sources" => handle_sources(store),
        "bpx_context_status" => handle_context_status(config, session_store),
        _ => anyhow::bail!("Unknown tool: {tool_name}"),
    }
}

fn search_weights(config: &Config) -> crate::store::search::SearchWeights {
    crate::store::search::SearchWeights {
        keyword_weight: config.search.keyword_weight,
        vector_weight: config.search.vector_weight,
    }
}

fn adaptive_content(
    content: String,
    threshold: usize,
    preview_bytes: usize,
) -> (String, bool, usize) {
    let len = content.len();
    if len <= threshold {
        (content, true, len)
    } else {
        (
            truncate::preview(&content, preview_bytes),
            false,
            len.min(preview_bytes),
        )
    }
}

fn preview_snippet(content: &str, max_bytes: usize) -> (String, bool, usize) {
    if content.len() <= max_bytes {
        (content.to_string(), false, content.len())
    } else {
        (
            truncate::preview(content, max_bytes),
            true,
            content.len().min(max_bytes),
        )
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
        result.stdout
    } else {
        format!("{}\n--- stderr ---\n{}", result.stdout, result.stderr)
    };

    // Index the output
    let index_result = store.index(label, &full_output, None)?;
    stats::record_indexed(full_output.len() as u64);

    let full_len = full_output.len();
    let (output_text, is_full, visible_bytes) =
        adaptive_content(full_output, config.general.preview_threshold_bytes, 3072);
    stats::record_visible(visible_bytes as u64);
    stats::record_returned(output_text.len() as u64);

    let footer = if is_full {
        format!("Full content returned ({full_len} bytes).")
    } else {
        "Use bpx_search or bpx_read_chunks for specific sections.".to_string()
    };

    Ok(format!(
        "{output_text}\n\n---\n{} chunks indexed (label: \"{label}\"). {footer}",
        index_result.chunk_count,
    ))
}

fn handle_batch_execute(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let commands = args["commands"].as_array().unwrap_or(&Vec::new()).clone();
    let queries = args["queries"].as_array().unwrap_or(&Vec::new()).clone();

    let weights = search_weights(config);

    let mut output = String::new();
    let mut section_labels = Vec::new();
    let mut visible_bytes = 0u64;

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
            result.stdout
        } else {
            format!("{}\n--- stderr ---\n{}", result.stdout, result.stderr)
        };

        store.index(label, &full, None)?;
        stats::record_indexed(full.len() as u64);
        section_labels.push(label.to_string());
    }

    output.push_str(&format!(
        "## Indexed Sections\n{}\n\n",
        section_labels.join(", ")
    ));

    // Run all queries
    for query_val in &queries {
        let query = query_val.as_str().unwrap_or("");
        if query.is_empty() {
            continue;
        }

        stats::record_search();
        let results =
            store.search_with_weights(query, config.search.default_limit, None, None, &weights)?;

        output.push_str(&format!("### Query: \"{query}\"\n"));
        if results.is_empty() {
            output.push_str("No results found.\n\n");
        } else {
            for (i, result) in results.iter().enumerate() {
                let max_bytes = if i < config.search.top_result_count {
                    config.search.snippet_bytes
                } else {
                    config.search.secondary_snippet_bytes
                };
                let (snippet, _, snippet_visible) = preview_snippet(&result.content, max_bytes);
                visible_bytes += snippet_visible as u64;
                let line_info = if result.line_start > 0 {
                    format!(", lines: {}-{}", result.line_start, result.line_end)
                } else {
                    String::new()
                };
                output.push_str(&format!(
                    "**[{}]** (source: {}, score: {:.4}{line_info})\n{snippet}\n\n",
                    result.title, result.source, result.score
                ));
            }
        }
    }

    stats::record_visible(visible_bytes);
    stats::record_returned(output.len() as u64);
    Ok(output)
}

fn handle_search(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let queries = args["queries"].as_array().unwrap_or(&Vec::new()).clone();
    let source = args["source"].as_str();
    let content_type = args["content_type"].as_str();
    let limit = args["limit"]
        .as_u64()
        .unwrap_or(config.search.default_limit as u64) as u32;
    let top_snippet_bytes = args["snippet_bytes"]
        .as_u64()
        .unwrap_or(config.search.snippet_bytes as u64) as usize;

    let weights = search_weights(config);

    let mut output = String::new();
    let mut visible_bytes = 0u64;

    for query_val in &queries {
        let query = query_val.as_str().unwrap_or("");
        if query.is_empty() {
            continue;
        }

        stats::record_search();
        let results = store.search_with_weights(query, limit, source, content_type, &weights)?;

        output.push_str(&format!("### \"{query}\"\n"));
        if results.is_empty() {
            output.push_str("No results.\n\n");
        } else {
            let mut truncated_any = false;
            for (i, result) in results.iter().enumerate() {
                let max_bytes = if i < config.search.top_result_count {
                    top_snippet_bytes
                } else {
                    config.search.secondary_snippet_bytes
                };
                let (snippet, truncated, snippet_visible) =
                    preview_snippet(&result.content, max_bytes);
                truncated_any |= truncated;
                visible_bytes += snippet_visible as u64;
                let line_info = if result.line_start > 0 {
                    format!(", lines: {}-{}", result.line_start, result.line_end)
                } else {
                    String::new()
                };
                output.push_str(&format!(
                    "**[{}]** (source: {}, source_id: {}, type: {}, score: {:.4}{line_info})\n{snippet}\n\n",
                    result.title, result.source, result.source_id, result.content_type, result.score
                ));
            }
            if truncated_any {
                output.push_str(
                    "Some results were truncated. Use bpx_read_chunks with the source label to see full content.\n\n"
                );
            }
        }
    }

    // Negative-claim warning for broad queries
    let is_broad = source.is_none()
        && queries.len() == 1
        && queries[0]
            .as_str()
            .is_some_and(|q| q.split_whitespace().count() <= 2);

    if is_broad {
        output.push_str(
            "**Review note:** These results show where matches were found, not where they are absent. \
             Do not infer \"no endpoint checks X\" or \"all routes do Y\" from search results alone. \
             Use the native Read tool on specific files to confirm presence or absence before making \
             severity claims.\n\n"
        );
    }

    stats::record_visible(visible_bytes);
    stats::record_returned(output.len() as u64);
    Ok(output)
}

fn handle_execute_file(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let path = args["path"].as_str().unwrap_or("");
    let code = args["code"].as_str();
    let query = args["query"].as_str().filter(|q| !q.trim().is_empty());

    let content = if let Some(processing_code) = code {
        // Pipe file through processing command
        let command = format!(
            "cat '{}' | {}",
            path.replace('\'', "'\\''"),
            processing_code
        );
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

    let threshold = config.general.preview_threshold_bytes;
    if content.len() <= threshold {
        stats::record_visible(content.len() as u64);
        stats::record_returned(content.len() as u64);
        return Ok(format!(
            "{content}\n\n---\n{} chunks indexed (label: \"{label}\"). Full content returned ({} bytes).",
            index_result.chunk_count,
            content.len(),
        ));
    }

    if let Some(query) = query {
        let results =
            store.search_with_weights(query, 5, Some(label), None, &search_weights(config))?;
        if !results.is_empty() {
            let mut output = String::new();
            let mut visible_bytes = 0u64;

            for result in &results {
                visible_bytes += result.content.len() as u64;
                output.push_str(&format!("### {}\n{}\n\n", result.title, result.content));
            }

            while output.ends_with('\n') {
                output.pop();
            }

            stats::record_visible(visible_bytes);
            stats::record_returned(output.len() as u64);
            return Ok(format!(
                "{output}\n\n---\n{} chunks indexed (label: \"{label}\"). Returned query-matched chunks for \"{query}\".",
                index_result.chunk_count,
            ));
        }
    }

    let (preview, _, visible_bytes) = adaptive_content(content, 0, 3072);
    stats::record_visible(visible_bytes as u64);
    stats::record_returned(preview.len() as u64);

    Ok(format!(
        "{preview}\n\n---\n{} chunks indexed (label: \"{label}\"). Use bpx_search or bpx_read_chunks for specific sections.",
        index_result.chunk_count,
    ))
}

fn handle_read_chunks(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let label = args["label"].as_str().unwrap_or("");
    let query = args["query"].as_str().filter(|q| !q.trim().is_empty());
    let max_chunks = args["max_chunks"].as_u64().unwrap_or(10) as usize;

    if let Some(query) = query {
        stats::record_search();
        let results = store.search_with_weights(
            query,
            max_chunks as u32,
            Some(label),
            None,
            &search_weights(config),
        )?;

        if results.is_empty() {
            return Ok(format!(
                "No indexed chunks matched \"{query}\" for source label \"{label}\"."
            ));
        }

        let mut output = String::new();
        let mut visible_bytes = 0u64;
        for result in &results {
            visible_bytes += result.content.len() as u64;
            let line_info = if result.line_start > 0 {
                format!(" (lines {}-{})", result.line_start, result.line_end)
            } else {
                String::new()
            };
            output.push_str(&format!(
                "### {}{line_info}\n{}\n\n",
                result.title, result.content
            ));
        }

        while output.ends_with('\n') {
            output.pop();
        }

        stats::record_visible(visible_bytes);
        stats::record_returned(output.len() as u64);
        return Ok(output);
    }

    let chunks = store.get_chunks_by_source(label)?;
    if chunks.is_empty() {
        return Ok(format!(
            "No indexed chunks found for source label \"{label}\"."
        ));
    }

    let mut output = String::new();
    let mut visible_bytes = 0u64;
    for chunk in chunks.iter().take(max_chunks) {
        visible_bytes += chunk.content.len() as u64;
        let line_info = if chunk.line_start > 0 {
            format!(" (lines {}-{})", chunk.line_start, chunk.line_end)
        } else {
            String::new()
        };
        output.push_str(&format!(
            "### {}{line_info}\n{}\n\n",
            chunk.title, chunk.content
        ));
    }

    while output.ends_with('\n') {
        output.pop();
    }

    stats::record_visible(visible_bytes);
    stats::record_returned(output.len() as u64);
    Ok(output)
}

fn handle_fetch_and_index(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let url = args["url"].as_str().unwrap_or("");
    let label = args["label"].as_str().unwrap_or(url);

    let content = fetch::fetch_and_convert(url)?;
    let index_result = store.index(label, &content, Some("prose"))?;
    stats::record_indexed(content.len() as u64);

    let content_len = content.len();
    let (output_text, is_full, visible_bytes) =
        adaptive_content(content, config.general.preview_threshold_bytes, 3072);
    stats::record_visible(visible_bytes as u64);
    stats::record_returned(output_text.len() as u64);

    let footer = if is_full {
        format!("Full content returned ({content_len} bytes).")
    } else {
        "Use bpx_search or bpx_read_chunks for specific sections.".to_string()
    };

    Ok(format!(
        "{output_text}\n\n---\n{} chunks indexed from URL (label: \"{label}\"). {footer}",
        index_result.chunk_count,
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

fn handle_index_dir(args: &Value, config: &Config, store: &ContentStore) -> Result<String> {
    let path = args["path"].as_str().unwrap_or(".");
    let glob = args["glob"].as_str();
    let label_prefix = args["label_prefix"].as_str().unwrap_or("");

    let path = std::path::Path::new(path);
    let result = indexdir::index_directory(store, path, glob, label_prefix)?;
    stats::record_indexed(result.total_bytes);

    let mut output = format!(
        "Indexed {} files ({} chunks, {} skipped).",
        result.files_indexed, result.total_chunks, result.files_skipped
    );

    let avg_file_size = if result.files_indexed > 0 {
        result.total_bytes / result.files_indexed as u64
    } else {
        0
    };
    output.push_str(&format!("\nAverage file size: {avg_file_size} bytes."));

    if !result.walk_errors.is_empty() {
        output.push_str("\n\nWalk errors:");
        for err in &result.walk_errors {
            output.push_str(&format!("\n- {err}"));
        }
    }

    if avg_file_size as usize <= config.general.preview_threshold_bytes {
        output.push_str(
            "\n\nMost files are small enough to Read directly. Use bpx_search to find content across files."
        );
    } else {
        output.push_str(
            "\n\nUse bpx_search to find content, or bpx_read_chunks for full chunks by source label."
        );
    }

    Ok(output)
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

fn handle_stats(
    config: &Config,
    store: &ContentStore,
    project_dir: &Path,
    session_store: &SessionStore,
) -> Result<String> {
    let session_stats = stats::get_stats();
    let total_indexed = store.total_bytes_indexed()?;
    let sources = store.list_sources()?;

    let db_path = crate::db::content_db_path(project_dir)?;
    let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    let mut output = format!(
        "## bpcontext Stats\n\n\
         **Session:**\n\
        - Commands executed: {}\n\
        - Searches performed: {}\n\
        - Bytes indexed (session): {}\n\
        - Bytes returned (session): {}\n\
        - Bytes visible to agent: {}\n\
        - Visibility ratio: {:.1}%\n\
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
        session_stats.bytes_visible,
        session_stats.visibility_ratio() * 100.0,
        session_stats.savings_ratio(),
        session_stats.tokens_saved(),
        sources.len(),
        db_size / 1024,
    );

    // Context budget summary
    let conn = session_store.conn();
    let ctx_tokens = crate::context::ledger::total_tokens(conn).unwrap_or(0);
    let ctx_sources = crate::context::ledger::source_count(conn).unwrap_or(0);
    let budget = config.context.budget_tokens;
    let pct = if budget > 0 {
        ctx_tokens as f64 / budget as f64 * 100.0
    } else {
        0.0
    };

    output.push_str(&format!(
        "\n**Context Budget:**\n\
         - Tokens used: {ctx_tokens}/{budget} ({pct:.1}%)\n\
         - Sources tracked: {ctx_sources}\n"
    ));

    // Per-source breakdown if any sources are tracked
    let breakdown = crate::context::ledger::source_breakdown(conn).unwrap_or_default();
    if !breakdown.is_empty() {
        output.push_str("- Top consumers: ");
        let top: Vec<String> = breakdown
            .iter()
            .take(5)
            .map(|s| format!("{} ({}t)", s.label, s.tokens))
            .collect();
        output.push_str(&top.join(", "));
        output.push('\n');
    }

    Ok(output)
}

fn handle_sources(store: &ContentStore) -> Result<String> {
    let sources = store.list_sources()?;

    if sources.is_empty() {
        return Ok("No sources indexed.".to_string());
    }

    let mut output = format!("## Indexed Sources ({})\n\n", sources.len());
    for source in &sources {
        output.push_str(&format!(
            "- **{}** — {} chunks, indexed {}\n",
            source.label, source.chunk_count, source.indexed_at
        ));
    }
    Ok(output)
}

fn handle_context_status(config: &Config, session_store: &SessionStore) -> Result<String> {
    let conn = session_store.conn();

    let total_tokens = crate::context::ledger::total_tokens(conn)?;
    let n_sources = crate::context::ledger::source_count(conn)?;
    let breakdown = crate::context::ledger::source_breakdown(conn)?;
    let budget = config.context.budget_tokens;
    let pct = if budget > 0 {
        total_tokens as f64 / budget as f64 * 100.0
    } else {
        0.0
    };

    let mut output = format!(
        "## Context Budget\n\n\
         - Tokens used: {total_tokens}/{budget} ({pct:.1}%)\n\
         - Sources tracked: {n_sources}\n"
    );

    if !breakdown.is_empty() {
        output.push_str("\n### Per-Source Breakdown\n\n");
        for source in &breakdown {
            output.push_str(&format!(
                "- **{}** — {}t, {} accesses, last {}\n",
                source.label, source.tokens, source.access_count, source.last_access
            ));
        }
    }

    // Relevance scores if enough sources
    if breakdown.len() >= 2 {
        let weights = crate::context::advisor::RelevanceWeights {
            recency: config.context.recency_weight,
            frequency: config.context.frequency_weight,
            staleness: config.context.staleness_weight,
        };
        let scored = crate::context::advisor::score_sources(
            conn,
            &weights,
            config.context.stale_threshold_minutes,
        )?;

        if !scored.is_empty() {
            output.push_str("\n### Relevance Scores\n\n");
            for s in &scored {
                let level = if s.relevance > 0.6 {
                    "high"
                } else if s.relevance > 0.3 {
                    "mid"
                } else {
                    "low"
                };
                output.push_str(&format!(
                    "- **{}** — {:.2} ({})\n",
                    s.label, s.relevance, level
                ));
            }
        }
    }

    Ok(output)
}

/// Extract a human-readable label from a tool call for the context ledger.
fn tool_label(tool_name: &str, args: &Value) -> String {
    match tool_name {
        "bpx_execute" => args["label"]
            .as_str()
            .or_else(|| args["command"].as_str())
            .unwrap_or("execute")
            .to_string(),
        "bpx_execute_file" => args["path"]
            .as_str()
            .and_then(|p| std::path::Path::new(p).file_name().and_then(|n| n.to_str()))
            .unwrap_or("file")
            .to_string(),
        "bpx_read_chunks" => args["label"].as_str().unwrap_or("read-chunks").to_string(),
        "bpx_search" => "search".to_string(),
        "bpx_batch_execute" => "batch".to_string(),
        "bpx_fetch_and_index" => args["label"]
            .as_str()
            .or_else(|| args["url"].as_str())
            .unwrap_or("fetch")
            .to_string(),
        "bpx_index" => args["label"].as_str().unwrap_or("index").to_string(),
        "bpx_promote" => "promote".to_string(),
        "bpx_index_dir" => args["path"].as_str().unwrap_or("index-dir").to_string(),
        "bpx_stats" => "stats".to_string(),
        "bpx_sources" => "sources".to_string(),
        "bpx_context_status" => "context-status".to_string(),
        _ => tool_name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionStore;
    use crate::stats;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn reset_stats() {
        stats::reset();
    }

    fn test_config() -> Config {
        Config::default()
    }

    fn make_store() -> (tempfile::TempDir, ContentStore) {
        std::env::set_var("XDG_DATA_HOME", "/tmp/bpcontext-test-data");
        let _ = fs::create_dir_all("/tmp/bpcontext-test-data");
        let dir = tempdir().unwrap();
        let store = ContentStore::open(dir.path()).unwrap();
        (dir, store)
    }

    fn make_store_with_session() -> (tempfile::TempDir, ContentStore, SessionStore) {
        std::env::set_var("XDG_DATA_HOME", "/tmp/bpcontext-test-data");
        let _ = fs::create_dir_all("/tmp/bpcontext-test-data");
        let dir = tempdir().unwrap();
        let store = ContentStore::open(dir.path()).unwrap();
        let session = SessionStore::open(dir.path()).unwrap();
        (dir, store, session)
    }

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    fn shell_quote_path(path: &Path) -> String {
        path.display().to_string().replace('\'', "'\\''")
    }

    fn cat_command(path: &Path) -> String {
        format!("cat '{}'", shell_quote_path(path))
    }

    fn repeated_line(line: &str, count: usize) -> String {
        line.repeat(count)
    }

    fn two_section_content(first_body: &str, second_body: &str) -> String {
        format!("# Intro\n\n{first_body}\n\n# Details\n\n{second_body}\n")
    }

    #[test]
    fn test_execute_file_small_returns_full() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let path = write_file(dir.path(), "small.txt", "small file contents");

        let result = handle_execute_file(&json!({ "path": path }), &config, &store).unwrap();

        assert!(result.contains("small file contents"));
        assert!(result.contains("Full content returned"));
    }

    #[test]
    fn test_execute_file_large_truncates() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let path = write_file(dir.path(), "large.txt", &"a".repeat(9000));

        let result = handle_execute_file(&json!({ "path": path }), &config, &store).unwrap();

        assert!(result.contains("[truncated"));
        assert!(result.contains("Use bpx_search or bpx_read_chunks"));
    }

    #[test]
    fn test_execute_small_output_returns_full() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let path = write_file(dir.path(), "cmd.txt", "command output");

        let result = handle_execute(
            &json!({ "command": cat_command(&path), "label": "cmd" }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("command output"));
        assert!(result.contains("Full content returned"));
    }

    #[test]
    fn test_execute_large_output_truncates() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let path = write_file(dir.path(), "cmd-large.txt", &"b".repeat(9000));

        let result = handle_execute(
            &json!({ "command": cat_command(&path), "label": "cmd-large" }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("[truncated"));
        assert!(result.contains("Use bpx_search or bpx_read_chunks"));
    }

    #[test]
    fn test_adaptive_threshold_default_8192() {
        assert_eq!(Config::default().general.preview_threshold_bytes, 8192);
    }

    #[test]
    fn test_adaptive_output_message_differs() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let small = write_file(dir.path(), "small-message.txt", "small body");
        let large = write_file(dir.path(), "large-message.txt", &"c".repeat(9000));

        let full = handle_execute_file(&json!({ "path": small }), &config, &store).unwrap();
        let truncated = handle_execute_file(&json!({ "path": large }), &config, &store).unwrap();

        assert!(full.contains("Full content returned"));
        assert!(!full.contains("Use bpx_search or bpx_read_chunks"));
        assert!(truncated.contains("Use bpx_search or bpx_read_chunks"));
    }

    #[test]
    fn test_read_chunks_returns_full_content() {
        reset_stats();
        let (_dir, store) = make_store();
        let content = two_section_content(
            &repeated_line("intro line\n", 350),
            &format!("{}\nFULL_CHUNK_MARKER", repeated_line("detail line\n", 350)),
        );
        store.index("manual.md", &content, None).unwrap();

        let result =
            handle_read_chunks(&json!({ "label": "manual.md" }), &test_config(), &store).unwrap();

        assert!(result.contains("FULL_CHUNK_MARKER"));
        assert!(!result.contains("[truncated"));
    }

    #[test]
    fn test_read_chunks_with_query_filters() {
        reset_stats();
        let (_dir, store) = make_store();
        let content = two_section_content(
            "alpha section\nalpha section\nalpha section",
            "needle-target\nneedle-target\nneedle-target",
        );
        store.index("filtered.md", &content, None).unwrap();

        let result = handle_read_chunks(
            &json!({ "label": "filtered.md", "query": "needle-target" }),
            &test_config(),
            &store,
        )
        .unwrap();

        assert!(result.contains("needle-target"));
        assert!(!result.contains("alpha section"));
    }

    #[test]
    fn test_search_tiered_snippets() {
        reset_stats();
        let (_dir, store) = make_store();
        let mut config = test_config();
        config.search.top_result_count = 2;
        config.search.snippet_bytes = 2000;
        config.search.secondary_snippet_bytes = 800;

        let top1 = format!(
            "{}{}{}",
            "needle ".repeat(6),
            "a".repeat(1200),
            "TOP_SNIPPET_MARKER_ONE"
        );
        let top2 = format!(
            "{}{}{}",
            "needle ".repeat(5),
            "b".repeat(1200),
            "TOP_SNIPPET_MARKER_TWO"
        );
        let low1 = format!("{}{}{}", "needle ", "c".repeat(1200), "LOW_MARKER_THREE");
        let low2 = format!("{}{}{}", "needle ", "d".repeat(1200), "LOW_MARKER_FOUR");

        store.index("one", &top1, None).unwrap();
        store.index("two", &top2, None).unwrap();
        store.index("three", &low1, None).unwrap();
        store.index("four", &low2, None).unwrap();

        let result = handle_search(
            &json!({ "queries": ["needle"], "limit": 4 }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("TOP_SNIPPET_MARKER_ONE"));
        assert!(result.contains("TOP_SNIPPET_MARKER_TWO"));
        assert!(!result.contains("LOW_MARKER_THREE"));
        assert!(!result.contains("LOW_MARKER_FOUR"));
    }

    #[test]
    fn test_search_snippet_bytes_param() {
        reset_stats();
        let (_dir, store) = make_store();
        let mut config = test_config();
        config.search.top_result_count = 1;
        config.search.snippet_bytes = 2000;
        let content = format!(
            "{}{}{}",
            "needle ".repeat(5),
            "x".repeat(300),
            "PARAM_MARKER"
        );
        store.index("param", &content, None).unwrap();

        let result = handle_search(
            &json!({ "queries": ["needle"], "limit": 1, "snippet_bytes": 120 }),
            &config,
            &store,
        )
        .unwrap();

        assert!(!result.contains("PARAM_MARKER"));
    }

    #[test]
    fn test_search_default_limit_15() {
        assert_eq!(Config::default().search.default_limit, 15);
    }

    #[test]
    fn test_batch_tiered_snippets() {
        reset_stats();
        let (dir, store) = make_store();
        let mut config = test_config();
        config.search.top_result_count = 2;
        config.search.snippet_bytes = 2000;
        config.search.secondary_snippet_bytes = 800;

        let path1 = write_file(
            dir.path(),
            "one.txt",
            &format!(
                "{}{}{}",
                "needle ".repeat(6),
                "a".repeat(1200),
                "BATCH_MARKER_ONE"
            ),
        );
        let path2 = write_file(
            dir.path(),
            "two.txt",
            &format!(
                "{}{}{}",
                "needle ".repeat(5),
                "b".repeat(1200),
                "BATCH_MARKER_TWO"
            ),
        );
        let path3 = write_file(
            dir.path(),
            "three.txt",
            &format!("{}{}{}", "needle ", "c".repeat(1200), "BATCH_MARKER_THREE"),
        );
        let path4 = write_file(
            dir.path(),
            "four.txt",
            &format!("{}{}{}", "needle ", "d".repeat(1200), "BATCH_MARKER_FOUR"),
        );

        let result = handle_batch_execute(
            &json!({
                "commands": [
                    { "label": "one", "command": cat_command(&path1) },
                    { "label": "two", "command": cat_command(&path2) },
                    { "label": "three", "command": cat_command(&path3) },
                    { "label": "four", "command": cat_command(&path4) }
                ],
                "queries": ["needle"]
            }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("BATCH_MARKER_ONE"));
        assert!(result.contains("BATCH_MARKER_TWO"));
        assert!(!result.contains("BATCH_MARKER_THREE"));
        assert!(!result.contains("BATCH_MARKER_FOUR"));
    }

    #[test]
    fn test_execute_file_query_returns_relevant() {
        reset_stats();
        let (dir, store) = make_store();
        let mut config = test_config();
        config.general.preview_threshold_bytes = 1024;
        let content = two_section_content(
            &repeated_line("intro filler line\n", 350),
            &format!(
                "{}\nAUTH_GUARD_MARKER",
                repeated_line("relevant line\n", 350)
            ),
        );
        let path = write_file(dir.path(), "audit.md", &content);

        let result = handle_execute_file(
            &json!({ "path": path, "query": "AUTH_GUARD_MARKER" }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("AUTH_GUARD_MARKER"));
        assert!(result.contains("Returned query-matched chunks"));
    }

    #[test]
    fn test_execute_file_query_small_still_full() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        let path = write_file(dir.path(), "small-query.txt", "small query file");

        let result =
            handle_execute_file(&json!({ "path": path, "query": "query" }), &config, &store)
                .unwrap();

        assert!(result.contains("small query file"));
        assert!(result.contains("Full content returned"));
        assert!(!result.contains("Returned query-matched chunks"));
    }

    #[test]
    fn test_execute_file_query_no_match_fallback() {
        reset_stats();
        let (dir, store) = make_store();
        let mut config = test_config();
        config.general.preview_threshold_bytes = 1024;
        let path = write_file(dir.path(), "nomatch.txt", &"z".repeat(9000));

        let result = handle_execute_file(
            &json!({ "path": path, "query": "missing-marker" }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("[truncated"));
        assert!(result.contains("Use bpx_search or bpx_read_chunks"));
        assert!(!result.contains("Returned query-matched chunks"));
    }

    #[test]
    fn test_index_dir_small_file_guidance() {
        reset_stats();
        let (dir, store) = make_store();
        let config = test_config();
        write_file(dir.path(), "main.rs", "fn main() {}\n");
        write_file(dir.path(), "lib.rs", "pub fn hi() {}\n");

        let result = handle_index_dir(
            &json!({ "path": dir.path().display().to_string() }),
            &config,
            &store,
        )
        .unwrap();

        assert!(result.contains("Average file size"));
        assert!(result.contains("Most files are small enough to Read directly"));
    }

    #[test]
    fn test_search_truncation_hint() {
        reset_stats();
        let (_dir, store) = make_store();
        let mut config = test_config();
        config.search.top_result_count = 1;
        config.search.snippet_bytes = 100;
        config.search.secondary_snippet_bytes = 50;
        let content = format!("{}{}", "needle ".repeat(5), "h".repeat(1500));
        store.index("hint", &content, None).unwrap();

        let result = handle_search(&json!({ "queries": ["needle"] }), &config, &store).unwrap();

        assert!(result.contains("Some results were truncated"));
        assert!(result.contains("bpx_read_chunks"));
    }

    #[test]
    fn test_stats_includes_visibility() {
        reset_stats();
        let (dir, store, session_store) = make_store_with_session();
        let config = test_config();
        store
            .index("stats-source", "visible content", None)
            .unwrap();
        stats::record_indexed(1000);
        stats::record_visible(120);

        let result = handle_stats(&config, &store, dir.path(), &session_store).unwrap();

        // Global stats are shared across parallel tests, so check label presence not exact values
        assert!(result.contains("Bytes visible to agent:"));
        assert!(result.contains("Visibility ratio"));
    }

    #[test]
    fn test_sources_lists_indexed() {
        let (_dir, store) = make_store();
        store.index("alpha/main.rs", "fn main() {}", None).unwrap();
        store.index("alpha/lib.rs", "pub mod utils;", None).unwrap();

        let result = handle_sources(&store).unwrap();

        assert!(result.contains("Indexed Sources (2)"));
        assert!(result.contains("alpha/main.rs"));
        assert!(result.contains("alpha/lib.rs"));
    }

    #[test]
    fn test_sources_empty() {
        let (_dir, store) = make_store();

        let result = handle_sources(&store).unwrap();

        assert!(result.contains("No sources indexed"));
    }

    #[test]
    fn test_context_status_shows_budget() {
        let (_dir, _store, session_store) = make_store_with_session();
        let config = test_config();

        let result = handle_context_status(&config, &session_store).unwrap();

        assert!(result.contains("Context Budget"));
        assert!(result.contains("Tokens used:"));
        assert!(result.contains("Sources tracked:"));
    }
}
