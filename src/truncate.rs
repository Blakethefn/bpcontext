/// Truncate output using head+tail strategy
///
/// Keeps the first `head_ratio` of the max bytes and the last `(1 - head_ratio)`,
/// inserting a marker in the middle showing how many bytes were omitted.
pub fn truncate_output(content: &str, max_bytes: usize, head_ratio: f64) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }

    let head_bytes = (max_bytes as f64 * head_ratio) as usize;
    let tail_bytes = max_bytes - head_bytes;

    let head = safe_truncate(content, head_bytes);
    let tail = safe_truncate_end(content, tail_bytes);
    let omitted = content.len() - head.len() - tail.len();

    format!("{head}\n\n... [{omitted} bytes omitted] ...\n\n{tail}")
}

/// Generate a preview (first N bytes) of content
pub fn preview(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let truncated = safe_truncate(content, max_bytes);
    format!(
        "{truncated}\n\n... [truncated, {total} bytes total]",
        total = content.len()
    )
}

/// Truncate a string at a UTF-8 safe boundary from the start
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Find the last newline before max_bytes for cleaner cuts
    let slice = &s[..max_bytes.min(s.len())];
    match slice.rfind('\n') {
        Some(pos) if pos > max_bytes / 2 => &s[..pos],
        _ => {
            // Fall back to char boundary
            let mut end = max_bytes.min(s.len());
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            &s[..end]
        }
    }
}

/// Truncate a string at a UTF-8 safe boundary from the end
fn safe_truncate_end(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let start = s.len() - max_bytes;
    // Find the first newline after start for cleaner cuts
    match s[start..].find('\n') {
        Some(pos) if pos < max_bytes / 2 => &s[start + pos + 1..],
        _ => {
            let mut begin = start;
            while begin < s.len() && !s.is_char_boundary(begin) {
                begin += 1;
            }
            &s[begin..]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_truncation_needed() {
        let short = "hello world";
        assert_eq!(truncate_output(short, 100, 0.6), short);
    }

    #[test]
    fn test_truncation_preserves_head_and_tail() {
        let content = "a".repeat(1000);
        let result = truncate_output(&content, 200, 0.6);
        assert!(result.len() < 1000);
        assert!(result.contains("bytes omitted"));
    }

    #[test]
    fn test_preview() {
        let content = "a".repeat(500);
        let result = preview(&content, 100);
        assert!(result.contains("truncated"));
        assert!(result.contains("500 bytes total"));
    }
}
