use anyhow::Result;
use serde_json::json;

use super::{read_hook_input, write_hook_output};

/// Handle PreToolUse hook
///
/// Intercepts Bash, Read, Grep, and WebFetch calls that are likely
/// to produce large output, and suggests routing through bpx_ equivalents.
pub fn handle() -> Result<()> {
    let input = read_hook_input()?;

    let tool_name = input["tool_name"].as_str().unwrap_or("");
    let tool_input = &input["tool_input"];

    let guidance = match tool_name {
        "Bash" => check_bash_command(tool_input),
        "Read" => check_read_command(tool_input),
        "WebFetch" => Some(suggest_fetch(tool_input)),
        _ => None,
    };

    match guidance {
        Some(msg) => write_hook_output(&json!({
            "decision": "block",
            "reason": msg
        })),
        None => write_hook_output(&json!({
            "decision": "approve"
        })),
    }
}

/// Check if a Bash command is likely to produce large output
fn check_bash_command(tool_input: &serde_json::Value) -> Option<String> {
    let command = tool_input["command"].as_str()?;

    // Commands likely to produce large output
    let large_output_patterns = [
        "git log",
        "git diff",
        "find ",
        "cat ",
        "head -",
        "tail -",
        "ls -la",
        "ls -R",
        "tree ",
        "npm test",
        "cargo test",
        "pytest",
        "make ",
        "cargo build",
    ];

    // Short safe commands that should pass through
    let safe_patterns = [
        "git status",
        "git branch",
        "git checkout",
        "git add",
        "git commit",
        "git push",
        "git pull",
        "mkdir",
        "rm ",
        "mv ",
        "cp ",
        "cd ",
        "echo ",
        "pwd",
        "which ",
        "whoami",
    ];

    // Allow safe commands through
    for pattern in &safe_patterns {
        if command.starts_with(pattern) {
            return None;
        }
    }

    // Suggest redirect for large-output commands
    for pattern in &large_output_patterns {
        if command.contains(pattern) {
            return Some(format!(
                "Command '{}' may produce large output. Use bpx_execute instead to keep output indexed and out of context.",
                truncate_for_message(command)
            ));
        }
    }

    None
}

/// Check if a Read command targets a large file
fn check_read_command(tool_input: &serde_json::Value) -> Option<String> {
    let _path = tool_input["file_path"].as_str()?;

    // We can't check file size in the hook without I/O,
    // so we provide guidance as a context hint rather than blocking
    None
}

/// Suggest using bpx_fetch_and_index instead of WebFetch
fn suggest_fetch(tool_input: &serde_json::Value) -> String {
    let url = tool_input["url"].as_str().unwrap_or("the URL");
    format!(
        "Use bpx_fetch_and_index instead of WebFetch for '{url}' to index the content and keep it out of context."
    )
}

fn truncate_for_message(s: &str) -> String {
    if s.len() <= 80 {
        s.to_string()
    } else {
        format!("{}...", &s[..77])
    }
}
