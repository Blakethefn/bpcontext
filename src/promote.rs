use anyhow::{Context, Result};
use std::process::Command;

use crate::config::Config;

/// Promote content to an obsidian output note via taskvault
pub fn promote_to_obsidian(
    config: &Config,
    name: &str,
    project: &str,
    content: &str,
) -> Result<String> {
    // Create the output note via taskvault
    let output = Command::new(&config.integration.taskvault_bin)
        .args(["new-output", name, "--project", project])
        .output()
        .with_context(|| {
            format!(
                "Failed to run taskvault at {}",
                config.integration.taskvault_bin
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("taskvault failed: {stderr}");
    }

    // Get the path of the created note
    let path_output = Command::new(&config.integration.taskvault_bin)
        .args(["path", "output", name])
        .output()
        .context("Failed to get output note path from taskvault")?;

    let note_path = String::from_utf8_lossy(&path_output.stdout)
        .trim()
        .to_string();

    if note_path.is_empty() {
        anyhow::bail!("taskvault returned empty path for output note");
    }

    // Append the promoted content to the note body
    let existing = std::fs::read_to_string(&note_path)
        .with_context(|| format!("Failed to read note at {note_path}"))?;

    let updated = format!("{existing}\n## Promoted Content\n\n{content}\n");
    std::fs::write(&note_path, updated)
        .with_context(|| format!("Failed to write to note at {note_path}"))?;

    Ok(note_path)
}
