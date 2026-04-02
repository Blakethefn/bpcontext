pub mod pretooluse;
pub mod posttooluse;
pub mod precompact;

use anyhow::Result;
use serde_json::Value;
use std::io::{self, Read};

/// Read JSON input from stdin (as provided by Claude Code hooks)
pub fn read_hook_input() -> Result<Value> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let value: Value = serde_json::from_str(&input)?;
    Ok(value)
}

/// Write JSON output to stdout (response to Claude Code)
pub fn write_hook_output(value: &Value) -> Result<()> {
    let output = serde_json::to_string(value)?;
    println!("{output}");
    Ok(())
}

/// Dispatch a hook by type
pub fn dispatch(hook_type: &str) -> Result<()> {
    match hook_type {
        "pretooluse" => pretooluse::handle(),
        "posttooluse" => posttooluse::handle(),
        "precompact" => precompact::handle(),
        _ => anyhow::bail!("Unknown hook type: {hook_type}"),
    }
}
