use anyhow::{Context, Result};
use std::process::Command;
use std::time::Duration;

use crate::truncate::truncate_output;

/// Result of executing a command
#[derive(Debug)]
#[allow(dead_code)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub truncated: bool,
    pub original_bytes: usize,
}

/// Execute a shell command with output capture and truncation
pub fn execute_command(
    command: &str,
    max_bytes: usize,
    head_ratio: f64,
    timeout: Option<Duration>,
) -> Result<ExecResult> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);

    // Set timeout via a wrapper if specified
    let output = if let Some(timeout) = timeout {
        let timeout_secs = timeout.as_secs();
        let mut timeout_cmd = Command::new("timeout");
        timeout_cmd
            .arg(format!("{timeout_secs}"))
            .arg("sh")
            .arg("-c")
            .arg(command);
        timeout_cmd
            .output()
            .with_context(|| format!("Failed to execute command: {command}"))?
    } else {
        cmd.output()
            .with_context(|| format!("Failed to execute command: {command}"))?
    };

    let raw_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let raw_stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let original_bytes = raw_stdout.len() + raw_stderr.len();

    let combined = if raw_stderr.is_empty() {
        raw_stdout
    } else {
        format!("{raw_stdout}\n--- stderr ---\n{raw_stderr}")
    };

    let truncated = combined.len() > max_bytes;
    let final_output = if truncated {
        truncate_output(&combined, max_bytes, head_ratio)
    } else {
        combined.clone()
    };

    // Split back into stdout/stderr for the result
    let (stdout, stderr) = if final_output.contains("\n--- stderr ---\n") {
        let parts: Vec<&str> = final_output.splitn(2, "\n--- stderr ---\n").collect();
        (
            parts[0].to_string(),
            parts.get(1).unwrap_or(&"").to_string(),
        )
    } else {
        (final_output, String::new())
    };

    Ok(ExecResult {
        stdout,
        stderr,
        exit_code: output.status.code(),
        truncated,
        original_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_simple_command() {
        let result = execute_command("echo hello", 102400, 0.6, None).unwrap();
        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, Some(0));
        assert!(!result.truncated);
    }

    #[test]
    fn test_execute_with_stderr() {
        let result = execute_command("echo err >&2", 102400, 0.6, None).unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[test]
    fn test_execute_truncation() {
        // Generate large output
        let result = execute_command("seq 1 100000", 1024, 0.6, None).unwrap();
        assert!(result.truncated);
    }
}
