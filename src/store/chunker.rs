/// Maximum chunk size in bytes
const MAX_CHUNK_BYTES: usize = 4096;

/// A chunk of content ready for indexing
#[derive(Debug)]
pub struct Chunk {
    pub title: String,
    pub content: String,
    pub is_code: bool,
}

/// Split content into indexable chunks
///
/// Strategy:
/// - Markdown: split on headings, keep code blocks intact
/// - Plain text: split on double newlines (paragraphs)
/// - Oversized chunks get split at paragraph boundaries
pub fn chunk_content(content: &str) -> Vec<Chunk> {
    if looks_like_markdown(content) {
        chunk_markdown(content)
    } else {
        chunk_plain_text(content)
    }
}

/// Check if content appears to be markdown
fn looks_like_markdown(content: &str) -> bool {
    content.lines().any(|line| {
        line.starts_with('#')
            || line.starts_with("```")
            || line.starts_with("- ")
            || line.starts_with("* ")
    })
}

/// Chunk markdown by headings
fn chunk_markdown(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_title = String::from("(untitled)");
    let mut current_lines: Vec<&str> = Vec::new();
    let mut in_code_block = false;

    for line in content.lines() {
        // Track code block boundaries
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            current_lines.push(line);
            continue;
        }

        // Split on headings (only outside code blocks)
        if !in_code_block && line.starts_with('#') {
            // Flush previous section
            if !current_lines.is_empty() {
                let body = current_lines.join("\n");
                flush_chunk(&mut chunks, &current_title, &body);
                current_lines.clear();
            }
            current_title = line.trim_start_matches('#').trim().to_string();
            current_lines.push(line);
        } else {
            current_lines.push(line);
        }
    }

    // Flush final section
    if !current_lines.is_empty() {
        let body = current_lines.join("\n");
        flush_chunk(&mut chunks, &current_title, &body);
    }

    if chunks.is_empty() {
        chunks.push(Chunk {
            title: "(untitled)".to_string(),
            content: content.to_string(),
            is_code: detect_code(content),
        });
    }

    chunks
}

/// Chunk plain text by paragraphs
fn chunk_plain_text(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let paragraphs: Vec<&str> = content.split("\n\n").collect();

    let mut current = String::new();
    let mut idx = 0u32;

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        if current.len() + para.len() + 2 > MAX_CHUNK_BYTES && !current.is_empty() {
            idx += 1;
            chunks.push(Chunk {
                title: format!("section-{idx}"),
                content: std::mem::take(&mut current),
                is_code: detect_code(&current),
            });
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }

    if !current.is_empty() {
        idx += 1;
        let is_code = detect_code(&current);
        chunks.push(Chunk {
            title: format!("section-{idx}"),
            content: current,
            is_code,
        });
    }

    chunks
}

/// Flush accumulated content as one or more chunks (splitting if oversized)
fn flush_chunk(chunks: &mut Vec<Chunk>, title: &str, content: &str) {
    if content.len() <= MAX_CHUNK_BYTES {
        chunks.push(Chunk {
            title: title.to_string(),
            content: content.to_string(),
            is_code: detect_code(content),
        });
        return;
    }

    // Split oversized content at paragraph boundaries
    let paragraphs: Vec<&str> = content.split("\n\n").collect();
    let mut current = String::new();
    let mut part = 0u32;

    for para in paragraphs {
        if current.len() + para.len() + 2 > MAX_CHUNK_BYTES && !current.is_empty() {
            part += 1;
            chunks.push(Chunk {
                title: format!("{title} (part {part})"),
                content: std::mem::take(&mut current),
                is_code: detect_code(&current),
            });
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }

    if !current.is_empty() {
        part += 1;
        let is_code = detect_code(&current);
        chunks.push(Chunk {
            title: if part > 1 {
                format!("{title} (part {part})")
            } else {
                title.to_string()
            },
            content: current,
            is_code,
        });
    }
}

/// Heuristic: does this content look like code?
fn detect_code(content: &str) -> bool {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return false;
    }

    // Contains code fences
    if content.contains("```") {
        return true;
    }

    // >60% of lines start with common code patterns
    let code_lines = lines.iter().filter(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("fn ")
            || trimmed.starts_with("pub ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with('#')
            || trimmed.starts_with('{')
            || trimmed.starts_with('}')
            || trimmed.ends_with(';')
            || trimmed.ends_with('{')
    }).count();

    code_lines as f64 / lines.len() as f64 > 0.6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_markdown_splits_on_headings() {
        let md = "# Title\nSome text\n## Section\nMore text";
        let chunks = chunk_content(md);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].title, "Title");
        assert_eq!(chunks[1].title, "Section");
    }

    #[test]
    fn test_chunk_plain_text() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_content(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_detect_code() {
        assert!(detect_code("fn main() {\n    println!(\"hello\");\n}"));
        assert!(!detect_code("This is just a regular sentence about things."));
    }

    #[test]
    fn test_code_blocks_stay_intact() {
        let md = "# Example\n```rust\nfn main() {\n    // code\n}\n```\nSome text after.";
        let chunks = chunk_content(md);
        // The code block should be in the same chunk as its heading
        assert!(chunks[0].content.contains("fn main()"));
    }
}
