use anyhow::{Context, Result};

/// Fetch a URL and convert HTML to markdown
pub fn fetch_and_convert(url: &str) -> Result<String> {
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to fetch URL: {url}"))?;

    let status = response.status();
    if status.as_u16() < 200 || status.as_u16() >= 300 {
        anyhow::bail!("HTTP {} fetching {url}", status.as_u16());
    }

    let content_type = response
        .headers()
        .get("Content-Type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = response
        .into_body()
        .read_to_string()
        .context("Failed to read response body")?;

    // Convert HTML to markdown if it looks like HTML
    if content_type.contains("html") || body.trim_start().starts_with('<') {
        Ok(htmd::convert(&body).unwrap_or(body))
    } else {
        Ok(body)
    }
}
