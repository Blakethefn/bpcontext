use anyhow::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use super::{read_hook_input, write_hook_output};
use crate::config::Config;
use crate::knowledge::enrichment;
use crate::knowledge::sync;
use crate::knowledge::KnowledgeStore;
use crate::session::events::{self, Priority};
use crate::session::SessionStore;

/// Handle PostToolUse hook
///
/// Captures tool call responses and indexes them as session events
/// for continuity across context compactions. Also triggers write-through
/// knowledge re-indexing when files inside registered sources are modified.
pub fn handle() -> Result<()> {
    let input = read_hook_input()?;

    let tool_name = input["tool_name"].as_str().unwrap_or("unknown");
    let tool_response = input.get("tool_response");

    // Determine priority based on tool type
    let priority = match tool_name {
        "Bash" | "Read" | "Write" | "Edit" => Priority::Normal,
        "Agent" => Priority::High,
        _ => Priority::Low,
    };

    // Try to get project dir from environment
    let project_dir = std::env::current_dir()?;
    let session_store = SessionStore::open(&project_dir)?;

    // Build event data
    let event_data = json!({
        "tool": tool_name,
        "response_preview": truncate_response(tool_response),
    });

    events::insert_event(
        session_store.conn(),
        "tool_call",
        Some(tool_name),
        &event_data.to_string(),
        priority,
    )?;

    // Evict if over threshold
    events::evict_if_needed(session_store.conn(), 500)?;

    // Write-through knowledge re-indexing (best-effort, never fails the hook)
    knowledge_write_through(&input);

    write_hook_output(&json!({ "decision": "approve" }))
}

/// Truncate a tool response to a reasonable size for event storage
fn truncate_response(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(v) => {
            let s = v.to_string();
            if s.len() > 500 {
                format!("{}...", &s[..497])
            } else {
                s
            }
        }
        None => String::from("(no response)"),
    }
}

// ── Write-Through Knowledge Re-Indexing ────────────────────────────────────

/// Check if any written files are inside a knowledge source and re-index them.
/// Best-effort — errors are logged to stderr but never fail the hook.
fn knowledge_write_through(input: &Value) {
    let paths = extract_written_paths(input);
    if paths.is_empty() {
        return;
    }

    let config = match Config::load() {
        Ok(c) => c,
        Err(_) => return,
    };
    if !config.knowledge.enabled {
        return;
    }

    let store = match KnowledgeStore::open() {
        Ok(s) => s,
        Err(_) => return, // No knowledge DB or error — skip silently
    };

    write_through_inner(&paths, &store);
}

/// Core write-through logic: check each path against registered sources
/// and re-index matches. Returns the count of files reindexed.
fn write_through_inner(paths: &[PathBuf], store: &KnowledgeStore) -> usize {
    let mut reindexed = 0;
    for path in paths {
        if !path.exists() {
            continue; // File was deleted or never existed — skip
        }

        let source = match store.source_for_path(path) {
            Ok(Some(s)) => s,
            _ => continue, // Not inside any source, or DB error — skip
        };

        let enrichment_fn =
            enrichment::build_enrichment_fn(&source.enrichments, Path::new(&source.path));
        let enrich_ref = enrichment_fn
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn(&str, &Path) -> Value);

        if let Err(e) = sync::reindex_file(store, &source, path, enrich_ref) {
            eprintln!(
                "bpcontext: knowledge write-through failed for {}: {}",
                path.display(),
                e
            );
        } else {
            reindexed += 1;
        }
    }
    reindexed
}

// ── Path Extraction ────────────────────────────────────────────────────────

/// Extract file paths that were written by this tool call.
/// Returns absolute paths only.
fn extract_written_paths(input: &Value) -> Vec<PathBuf> {
    let tool_name = input["tool_name"].as_str().unwrap_or("");
    let tool_input = &input["tool_input"];

    match tool_name {
        "Write" | "Edit" => {
            if let Some(path) = tool_input["file_path"].as_str() {
                vec![PathBuf::from(path)]
            } else {
                vec![]
            }
        }
        "Bash" => {
            let command = tool_input["command"].as_str().unwrap_or("");
            let output = input["tool_response"]
                .as_str()
                .or_else(|| input["tool_response"]["stdout"].as_str())
                .unwrap_or("");
            extract_bash_write_paths(command, output)
        }
        _ => vec![],
    }
}

/// Extract file paths that a Bash command may have written.
/// Uses string scanning heuristics — no regex dependency.
///
/// Detected patterns:
/// - Shell redirects: `> /path`, `>> /path`
/// - Tee commands: `tee /path`, `tee -a /path`
/// - Copy/move destinations: `cp src /dest`, `mv src /dest`
/// - Tool output: "Created /path", "Wrote /path", etc.
/// - Taskvault output: `.md` paths containing obsidian/tasks/outputs
fn extract_bash_write_paths(command: &str, output: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Redirect targets: > /path or >> /path
    {
        let bytes = command.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            if bytes[i] == b'>' {
                let mut after = i + 1;
                // Skip second > in >>
                if after < len && bytes[after] == b'>' {
                    after += 1;
                }
                if let Some(path) = extract_path_after(command, after) {
                    paths.push(path);
                }
                i = after;
            } else {
                i += 1;
            }
        }
    }

    // 2. Tee targets: tee /path or tee -a /path
    {
        let mut search_from = 0;
        while let Some(pos) = command[search_from..].find("tee ") {
            let abs_pos = search_from + pos;
            // Ensure "tee" is at a word boundary (not inside "guarantee" etc.)
            if abs_pos > 0 {
                let prev = command.as_bytes()[abs_pos - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' {
                    search_from = abs_pos + 4;
                    continue;
                }
            }
            let after_tee = abs_pos + 4;
            // Skip -a flag if present
            let arg_start = if command[after_tee..].starts_with("-a ") {
                after_tee + 3
            } else {
                after_tee
            };
            if let Some(path) = extract_path_after(command, arg_start) {
                paths.push(path);
            }
            search_from = after_tee;
        }
    }

    // 3. cp/mv destination: last absolute-path argument
    {
        let trimmed = command.trim();
        if trimmed.starts_with("cp ") || trimmed.starts_with("mv ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(last) = parts.last() {
                if last.starts_with('/') {
                    paths.push(PathBuf::from(*last));
                }
            }
        }
    }

    // 4. Tool output patterns: "Created /path", "Wrote /path", etc.
    {
        let markers = [
            "File created at: ",
            "File created at ",
            "Created: ",
            "Created ",
            "Wrote: ",
            "Wrote ",
            "Written to: ",
            "Written to ",
        ];
        for marker in &markers {
            let mut search_from = 0;
            while let Some(pos) = output[search_from..].find(marker) {
                let after = search_from + pos + marker.len();
                if let Some(path) = extract_path_after(output, after) {
                    paths.push(path);
                }
                search_from = after;
            }
        }
    }

    // 5. Taskvault/obsidian paths: .md files mentioning obsidian/tasks/outputs
    {
        let mut search_from = 0;
        while search_from < output.len() {
            if let Some(pos) = output[search_from..].find('/') {
                let abs_pos = search_from + pos;
                if let Some(token) = scan_path_token(output, abs_pos) {
                    if token.ends_with(".md")
                        && (token.contains("obsidian")
                            || token.contains("tasks/")
                            || token.contains("outputs/"))
                    {
                        paths.push(PathBuf::from(token));
                    }
                }
                search_from = abs_pos + 1;
            } else {
                break;
            }
        }
    }

    paths
}

/// Extract an absolute path from `s` starting after position `from`.
/// Skips leading whitespace and optional quotes. Returns `None` if
/// no absolute path (starting with `/`) is found.
fn extract_path_after(s: &str, from: usize) -> Option<PathBuf> {
    if from >= s.len() {
        return None;
    }
    let rest = &s[from..];
    let trimmed = rest.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    // Skip optional opening quote
    let (content, end_char): (&str, Option<u8>) = match trimmed.as_bytes()[0] {
        b'"' => (&trimmed[1..], Some(b'"')),
        b'\'' => (&trimmed[1..], Some(b'\'')),
        _ => (trimmed, None),
    };
    if content.is_empty() || !content.starts_with('/') {
        return None;
    }
    let path_str = scan_until_terminator(content, end_char);
    if path_str.len() > 1 {
        Some(PathBuf::from(path_str))
    } else {
        None
    }
}

/// Scan from the start of `s` until a terminator character is found.
/// If `quote_end` is `Some`, the terminator is the matching quote.
/// Otherwise, whitespace and shell metacharacters terminate the scan.
fn scan_until_terminator(s: &str, quote_end: Option<u8>) -> &str {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if let Some(q) = quote_end {
            if b == q {
                return &s[..i];
            }
        } else {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' | b';' | b'|' | b'&' | b')' | b',' => {
                    return &s[..i];
                }
                _ => {}
            }
        }
    }
    s
}

/// Scan a path-like token starting at position `from`.
/// Collects characters until whitespace or shell metacharacter.
fn scan_path_token(s: &str, from: usize) -> Option<&str> {
    let bytes = s.as_bytes();
    if from >= bytes.len() || bytes[from] != b'/' {
        return None;
    }
    let mut end = from + 1;
    while end < bytes.len() {
        match bytes[end] {
            b' ' | b'\t' | b'\n' | b'\r' | b';' | b'|' | b'&' | b')' | b'"' | b'\'' | b','
            | b'(' => {
                break;
            }
            _ => end += 1,
        }
    }
    if end > from + 1 {
        Some(&s[from..end])
    } else {
        None
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{self, sync};
    use rusqlite::params;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    // ── extract_written_paths: Write/Edit ──────────────────────────────────

    #[test]
    fn write_tool_extracts_file_path() {
        let input = json!({
            "tool_name": "Write",
            "tool_input": { "file_path": "/tmp/test.md" }
        });
        let paths = extract_written_paths(&input);
        assert_eq!(paths, vec![PathBuf::from("/tmp/test.md")]);
    }

    #[test]
    fn edit_tool_extracts_file_path() {
        let input = json!({
            "tool_name": "Edit",
            "tool_input": { "file_path": "/home/user/code.rs" }
        });
        let paths = extract_written_paths(&input);
        assert_eq!(paths, vec![PathBuf::from("/home/user/code.rs")]);
    }

    #[test]
    fn write_tool_missing_file_path_returns_empty() {
        let input = json!({
            "tool_name": "Write",
            "tool_input": {}
        });
        assert!(extract_written_paths(&input).is_empty());
    }

    #[test]
    fn read_tool_returns_empty() {
        let input = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "/tmp/read.md" }
        });
        assert!(extract_written_paths(&input).is_empty());
    }

    #[test]
    fn unknown_tool_returns_empty() {
        let input = json!({
            "tool_name": "Agent",
            "tool_input": {}
        });
        assert!(extract_written_paths(&input).is_empty());
    }

    // ── extract_bash_write_paths: redirects ────────────────────────────────

    #[test]
    fn bash_redirect_single_gt() {
        let paths = extract_bash_write_paths("echo foo > /tmp/out.txt", "");
        assert!(paths.contains(&PathBuf::from("/tmp/out.txt")));
    }

    #[test]
    fn bash_redirect_double_gt() {
        let paths = extract_bash_write_paths("echo bar >> /tmp/append.log", "");
        assert!(paths.contains(&PathBuf::from("/tmp/append.log")));
    }

    #[test]
    fn bash_redirect_quoted_path() {
        let paths = extract_bash_write_paths("echo data > \"/tmp/my file.txt\"", "");
        assert!(paths.contains(&PathBuf::from("/tmp/my file.txt")));
    }

    #[test]
    fn bash_redirect_no_absolute_path() {
        let paths = extract_bash_write_paths("echo foo > relative.txt", "");
        assert!(paths.is_empty(), "only absolute paths should be extracted");
    }

    // ── extract_bash_write_paths: tee ──────────────────────────────────────

    #[test]
    fn bash_tee_simple() {
        let paths = extract_bash_write_paths("cmd | tee /tmp/tee-out.log", "");
        assert!(paths.contains(&PathBuf::from("/tmp/tee-out.log")));
    }

    #[test]
    fn bash_tee_append() {
        let paths = extract_bash_write_paths("cmd | tee -a /tmp/append.log", "");
        assert!(paths.contains(&PathBuf::from("/tmp/append.log")));
    }

    #[test]
    fn bash_tee_not_inside_word() {
        // "guarantee" contains "tee" but should not match
        let paths = extract_bash_write_paths("echo guarantee /tmp/nope", "");
        assert!(!paths.contains(&PathBuf::from("/tmp/nope")));
    }

    // ── extract_bash_write_paths: cp/mv ────────────────────────────────────

    #[test]
    fn bash_cp_destination() {
        let paths = extract_bash_write_paths("cp /tmp/src.txt /tmp/dest.txt", "");
        assert!(paths.contains(&PathBuf::from("/tmp/dest.txt")));
    }

    #[test]
    fn bash_mv_destination() {
        let paths = extract_bash_write_paths("mv /tmp/old.txt /tmp/new.txt", "");
        assert!(paths.contains(&PathBuf::from("/tmp/new.txt")));
    }

    #[test]
    fn bash_cp_relative_dest_not_extracted() {
        let paths = extract_bash_write_paths("cp /tmp/src.txt relative.txt", "");
        assert!(
            !paths.iter().any(|p| p == Path::new("relative.txt")),
            "relative paths should not be extracted by cp/mv heuristic"
        );
    }

    // ── extract_bash_write_paths: output patterns ──────────────────────────

    #[test]
    fn bash_output_created_pattern() {
        let paths = extract_bash_write_paths("", "Created /tmp/new-note.md");
        assert!(paths.contains(&PathBuf::from("/tmp/new-note.md")));
    }

    #[test]
    fn bash_output_wrote_pattern() {
        let paths = extract_bash_write_paths("", "Wrote: /tmp/data.json");
        assert!(paths.contains(&PathBuf::from("/tmp/data.json")));
    }

    #[test]
    fn bash_output_file_created_at() {
        let paths = extract_bash_write_paths("", "File created at: /tmp/file.md");
        assert!(paths.contains(&PathBuf::from("/tmp/file.md")));
    }

    // ── extract_bash_write_paths: taskvault/obsidian ───────────────────────

    #[test]
    fn bash_output_taskvault_md_path() {
        let output = "Note at /home/user/obsidian_docs/tasks/my-task.md done";
        let paths = extract_bash_write_paths("taskvault new-task", output);
        assert!(paths
            .iter()
            .any(|p| p.to_string_lossy().contains("tasks/my-task.md")));
    }

    #[test]
    fn bash_output_obsidian_outputs_path() {
        let output = "/vault/outputs/2026-04-07-report.md done";
        let paths = extract_bash_write_paths("", output);
        assert!(paths.contains(&PathBuf::from("/vault/outputs/2026-04-07-report.md")));
    }

    // ── extract_bash_write_paths: edge cases ───────────────────────────────

    #[test]
    fn bash_empty_command_and_output() {
        assert!(extract_bash_write_paths("", "").is_empty());
    }

    #[test]
    fn bash_no_write_command() {
        assert!(extract_bash_write_paths("ls -la /tmp", "file1 file2").is_empty());
    }

    // ── extract_path_after ─────────────────────────────────────────────────

    #[test]
    fn extract_path_absolute() {
        let result = extract_path_after("/tmp/file.txt rest", 0);
        assert_eq!(result, Some(PathBuf::from("/tmp/file.txt")));
    }

    #[test]
    fn extract_path_with_leading_space() {
        let result = extract_path_after("  /tmp/file.txt", 0);
        assert_eq!(result, Some(PathBuf::from("/tmp/file.txt")));
    }

    #[test]
    fn extract_path_not_absolute() {
        let result = extract_path_after("relative/path.txt", 0);
        assert!(result.is_none());
    }

    #[test]
    fn extract_path_empty() {
        assert!(extract_path_after("", 0).is_none());
    }

    #[test]
    fn extract_path_beyond_length() {
        assert!(extract_path_after("short", 100).is_none());
    }

    // ── write_through_inner (integration with in-memory store) ─────────────

    #[test]
    fn write_through_reindexes_file_in_source() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("note.md");
        fs::write(&file_path, "# Test Note\nSearchable content here").unwrap();

        store
            .add_source("test", dir.path().to_str().unwrap(), None, &[])
            .unwrap();
        let source = store.get_source("test").unwrap().unwrap();
        sync::sync_source(&store, &source, None).unwrap();

        // Modify the file
        fs::write(&file_path, "# Updated Note\nNew content entirely").unwrap();

        let count = write_through_inner(&[file_path], &store);
        assert_eq!(count, 1, "one file should be reindexed");

        let source = store.get_source("test").unwrap().unwrap();
        let chunks: Vec<String> = {
            let mut stmt = store
                .conn()
                .prepare(
                    "SELECT kc.content FROM knowledge_chunks kc
                     JOIN knowledge_chunk_meta kcm ON kcm.chunk_rowid = kc.rowid
                     WHERE kcm.source_id = ?1",
                )
                .unwrap();
            stmt.query_map(params![source.id], |row| row.get(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };
        let all = chunks.join("\n");
        assert!(
            all.contains("New content"),
            "updated content should be indexed"
        );
    }

    #[test]
    fn write_through_skips_path_outside_sources() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let source_dir = tempdir().unwrap();
        let other_dir = tempdir().unwrap();

        store
            .add_source("test", source_dir.path().to_str().unwrap(), None, &[])
            .unwrap();

        let outside_file = other_dir.path().join("outside.md");
        fs::write(&outside_file, "# Outside\nNot in any source").unwrap();

        let count = write_through_inner(&[outside_file], &store);
        assert_eq!(count, 0, "file outside sources should not be reindexed");
    }

    #[test]
    fn write_through_skips_nonexistent_file() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();
        store
            .add_source("test", dir.path().to_str().unwrap(), None, &[])
            .unwrap();

        let ghost = dir.path().join("ghost.md");
        let count = write_through_inner(&[ghost], &store);
        assert_eq!(count, 0, "nonexistent file should be skipped");
    }

    #[test]
    fn write_through_adds_new_file_to_index() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();

        store
            .add_source("test", dir.path().to_str().unwrap(), None, &[])
            .unwrap();

        let new_file = dir.path().join("brand-new.md");
        fs::write(&new_file, "# Brand New\nFresh file via write-through").unwrap();

        let count = write_through_inner(&[new_file], &store);
        assert_eq!(count, 1, "new file should be indexed");

        let source = store.get_source("test").unwrap().unwrap();
        let file_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM knowledge_files WHERE source_id = ?1",
                params![source.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(file_count, 1, "file record should exist in DB");
    }

    #[test]
    fn write_through_with_enrichments() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();

        let enrichments = vec!["frontmatter".to_string()];
        store
            .add_source(
                "test",
                dir.path().to_str().unwrap(),
                None,
                &enrichments,
            )
            .unwrap();

        let file = dir.path().join("enriched.md");
        fs::write(
            &file,
            "---\ntype: task\nstatus: active\n---\n# Task\nBody",
        )
        .unwrap();

        let count = write_through_inner(&[file], &store);
        assert_eq!(count, 1);

        let source = store.get_source("test").unwrap().unwrap();
        let metadata: String = store
            .conn()
            .query_row(
                "SELECT metadata FROM knowledge_chunk_meta WHERE source_id = ?1 LIMIT 1",
                params![source.id],
                |row| row.get(0),
            )
            .unwrap();
        let parsed: Value = serde_json::from_str(&metadata).unwrap();
        assert_eq!(parsed["frontmatter"]["type"], "task");
    }

    #[test]
    fn write_through_respects_glob_filter() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();

        store
            .add_source("test", dir.path().to_str().unwrap(), Some("**/*.md"), &[])
            .unwrap();

        // .rs file should not match the glob
        let rs_file = dir.path().join("code.rs");
        fs::write(&rs_file, "fn main() { println!(\"hello\"); }").unwrap();
        let count = write_through_inner(&[rs_file], &store);
        assert_eq!(count, 0, ".rs file should not match **/*.md glob");

        // .md file should match
        let md_file = dir.path().join("note.md");
        fs::write(&md_file, "# Note\nContent").unwrap();
        let count = write_through_inner(&[md_file], &store);
        assert_eq!(count, 1, ".md file should match glob");
    }

    #[test]
    fn write_through_no_sources_registered() {
        let store = knowledge::open_in_memory().expect("in-memory store");
        let dir = tempdir().unwrap();
        let file = dir.path().join("orphan.md");
        fs::write(&file, "# Orphan\nNo sources registered").unwrap();

        let count = write_through_inner(&[file], &store);
        assert_eq!(count, 0, "no sources means no reindexing");
    }
}
