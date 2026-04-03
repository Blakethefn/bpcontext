use anyhow::{Context, Result};

/// Fetch a URL and convert HTML to markdown
pub fn fetch_and_convert(url: &str) -> Result<String> {
    let response =
        reqwest::blocking::get(url).with_context(|| format!("Failed to fetch URL: {url}"))?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} fetching {url}");
    }

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = response.text().context("Failed to read response body")?;

    // Convert HTML to markdown if it looks like HTML
    if content_type.contains("html") || body.trim_start().starts_with('<') {
        Ok(htmd::convert(&body).unwrap_or(body))
    } else {
        Ok(body)
    }
}
