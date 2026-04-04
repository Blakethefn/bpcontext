/// Maximum chunk size in bytes
const MAX_CHUNK_BYTES: usize = 4096;

/// Minimum chunk size — avoid splitting too aggressively
const MIN_CHUNK_BYTES: usize = 256;

/// A chunk of content ready for indexing
#[derive(Debug)]
pub struct Chunk {
    pub title: String,
    pub content: String,
    pub is_code: bool,
    /// 1-based line number where this chunk starts (0 if unknown)
    pub line_start: u32,
    /// 1-based line number where this chunk ends, inclusive (0 if unknown)
    pub line_end: u32,
}

/// Result of single-pass content classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentType {
    Markdown,
    Code,
    PlainText,
}

/// Classify content in a single pass over all lines.
///
/// Tracks markdown indicators and code indicators simultaneously,
/// then applies markdown-first priority: if markdown passes, return
/// Markdown even if code would also match.
fn classify_content(content: &str) -> ContentType {
    // Quick structural checks for code (these inspect the whole string, not per-line)
    let has_php_tag = content.contains("<?php") || content.contains("<?=");
    let has_script_pair = content.contains("<script") && content.contains("</script>");
    let has_style_pair = content.contains("<style") && content.contains("</style>");

    let mut md_score: u32 = 0;
    let mut md_code_counter: u32 = 0; // code counter-indicators for markdown rejection
    let mut line_count: u32 = 0;

    let mut non_empty_lines: u32 = 0;
    let mut code_endings: u32 = 0;
    let mut has_func = false;

    for line in content.lines() {
        line_count += 1;
        let trimmed = line.trim();

        // --- Markdown indicators ---
        if trimmed.starts_with("# ")
            || trimmed.starts_with("## ")
            || trimmed.starts_with("### ")
            || trimmed.starts_with("#### ")
        {
            md_score += 1;
        }
        if trimmed.starts_with("```") {
            md_score += 1;
        }
        if (trimmed.starts_with("- ") || trimmed.starts_with("* "))
            && !trimmed.contains('{') && !trimmed.contains(';') {
                md_score += 1;
            }

        // --- Code counter-indicators (for markdown rejection) ---
        if trimmed.contains("<?php")
            || trimmed.starts_with("<script")
            || trimmed.starts_with("<style")
            || trimmed.starts_with("<html")
            || trimmed.starts_with("<!DOCTYPE")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("package ")
            || trimmed.starts_with("use ")
        {
            md_code_counter += 1;
        }
        if trimmed.ends_with(';') || trimmed.ends_with('{') {
            md_code_counter += 1;
        }

        // --- Code detection indicators ---
        if !trimmed.is_empty() {
            non_empty_lines += 1;
            if trimmed.ends_with(';') || trimmed.ends_with('{') || trimmed.ends_with('}') {
                code_endings += 1;
            }
            if !has_func {
                has_func = trimmed.starts_with("function ")
                    || trimmed.starts_with("def ")
                    || trimmed.starts_with("fn ")
                    || trimmed.starts_with("pub fn ")
                    || trimmed.contains("function(")
                    || trimmed.contains("function (")
                    || trimmed.starts_with("class ");
            }
        }
    }

    // --- Markdown check (priority) ---
    if line_count > 0 && md_score > 0 {
        let md_passes = (md_score as f64 / line_count as f64) > 0.02
            && (md_code_counter <= md_score * 2);
        if md_passes {
            return ContentType::Markdown;
        }
    }

    // --- Code check ---
    if has_php_tag || has_script_pair || has_style_pair {
        return ContentType::Code;
    }
    if non_empty_lines > 0 {
        let code_ratio = code_endings as f64 / non_empty_lines as f64;
        if code_ratio > 0.4 || (has_func && code_endings > 5) {
            return ContentType::Code;
        }
    }

    ContentType::PlainText
}

/// Split content into indexable chunks
///
/// Strategy:
/// - Markdown: split on headings, keep code blocks intact
/// - Code: split on structural boundaries (functions, classes, tags)
/// - Plain text: split on double newlines (paragraphs)
/// - All paths enforce MAX_CHUNK_BYTES via newline fallback
pub fn chunk_content(content: &str) -> Vec<Chunk> {
    match classify_content(content) {
        ContentType::Markdown => chunk_markdown(content),
        ContentType::Code => chunk_code(content),
        ContentType::PlainText => chunk_plain_text(content),
    }
}

#[cfg(test)]
fn looks_like_markdown(content: &str) -> bool {
    classify_content(content) == ContentType::Markdown
}

#[cfg(test)]
fn looks_like_code(content: &str) -> bool {
    classify_content(content) == ContentType::Code
}

/// Check if a line is a structural boundary in code
fn is_structural_boundary(line: &str) -> bool {
    let trimmed = line.trim();

    // PHP / JS functions
    trimmed.starts_with("function ")
        || trimmed.starts_with("function(")
        || trimmed.starts_with("public function ")
        || trimmed.starts_with("private function ")
        || trimmed.starts_with("protected function ")
        || trimmed.starts_with("static function ")
        || trimmed.starts_with("async function ")
        // Classes, traits, interfaces
        || trimmed.starts_with("class ")
        || trimmed.starts_with("abstract class ")
        || trimmed.starts_with("trait ")
        || trimmed.starts_with("interface ")
        // Python
        || trimmed.starts_with("def ")
        || trimmed.starts_with("async def ")
        // Rust
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub(crate) fn ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        // HTML embedded sections
        || trimmed.starts_with("<script")
        || trimmed.starts_with("<style")
        || trimmed.starts_with("</script>")
        || trimmed.starts_with("</style>")
        // PHP tags
        || trimmed.starts_with("<?php")
        || trimmed == "?>"
        // CSS at-rules
        || trimmed.starts_with("@media")
        || trimmed.starts_with("@keyframes")
        // JS arrow/const functions (common pattern)
        || (trimmed.starts_with("const ") && (trimmed.contains("=> {") || trimmed.contains("= function")))
        || (trimmed.starts_with("let ") && (trimmed.contains("=> {") || trimmed.contains("= function")))
}

/// Chunk source code by structural boundaries
fn chunk_code(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_bytes: usize = 0;
    let mut current_title: Option<String> = None;
    let mut line_number: u32 = 0;
    let mut chunk_line_start: u32 = 1;

    for line in content.lines() {
        line_number += 1;
        let line_bytes = line.len() + 1; // +1 for newline

        if is_structural_boundary(line) && current_bytes >= MIN_CHUNK_BYTES {
            // Flush current buffer
            let body = current_lines.join("\n");
            let title = current_title
                .take()
                .unwrap_or_else(|| make_code_title(&current_lines));
            let chunk_line_end = line_number - 1;
            flush_chunk_with_lines(&mut chunks, &title, &body, chunk_line_start, chunk_line_end);
            current_lines.clear();
            current_bytes = 0;
            chunk_line_start = line_number;
        }

        // Set title from first structural boundary in this chunk
        if current_title.is_none() && is_structural_boundary(line) {
            let trimmed = line.trim();
            let label = if trimmed.len() > 60 {
                format!("{}...", &trimmed[..57])
            } else {
                trimmed.to_string()
            };
            current_title = Some(label);
        }

        current_lines.push(line);
        current_bytes += line_bytes;
    }

    // Flush remaining
    if !current_lines.is_empty() {
        let body = current_lines.join("\n");
        let title = current_title.unwrap_or_else(|| make_code_title(&current_lines));
        flush_chunk_with_lines(&mut chunks, &title, &body, chunk_line_start, line_number);
    }

    if chunks.is_empty() {
        chunks.push(Chunk {
            title: "code".to_string(),
            content: content.to_string(),
            is_code: true,
            line_start: 1,
            line_end: line_number.max(1),
        });
    }

    chunks
}

/// Generate a title from the first meaningful line of a code chunk
fn make_code_title(lines: &[&str]) -> String {
    for line in lines.iter().take(5) {
        let trimmed = line.trim();
        if !trimmed.is_empty() && trimmed.len() > 2 {
            let label = if trimmed.len() > 60 {
                format!("{}...", &trimmed[..57])
            } else {
                trimmed.to_string()
            };
            return label;
        }
    }
    "code-section".to_string()
}

/// Chunk markdown content by headings
fn chunk_markdown(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_title = String::from("(untitled)");
    let mut current_lines: Vec<&str> = Vec::new();
    let mut in_code_block = false;
    let mut line_number: u32 = 0;
    let mut chunk_line_start: u32 = 1;

    for line in content.lines() {
        line_number += 1;

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
                let chunk_line_end = line_number - 1;
                flush_chunk_with_lines(
                    &mut chunks,
                    &current_title,
                    &body,
                    chunk_line_start,
                    chunk_line_end,
                );
                current_lines.clear();
            }
            current_title = line.trim_start_matches('#').trim().to_string();
            chunk_line_start = line_number;
            current_lines.push(line);
        } else {
            current_lines.push(line);
        }
    }

    // Flush final section
    if !current_lines.is_empty() {
        let body = current_lines.join("\n");
        flush_chunk_with_lines(
            &mut chunks,
            &current_title,
            &body,
            chunk_line_start,
            line_number,
        );
    }

    if chunks.is_empty() {
        chunks.push(Chunk {
            title: "(untitled)".to_string(),
            content: content.to_string(),
            is_code: detect_code(content),
            line_start: 1,
            line_end: line_number.max(1),
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
    // Track cumulative line offset for line number attribution
    let mut cumulative_lines: u32 = 0;
    let mut chunk_line_start: u32 = 1;

    for para in &paragraphs {
        let para_trimmed = para.trim();
        let para_lines = para.lines().count() as u32;

        if para_trimmed.is_empty() {
            continue;
        }

        if current.len() + para_trimmed.len() + 2 > MAX_CHUNK_BYTES && !current.is_empty() {
            idx += 1;
            let chunk_line_end = cumulative_lines.saturating_sub(1).max(chunk_line_start);
            let taken = std::mem::take(&mut current);
            let is_code = detect_code(&taken);
            chunks.push(Chunk {
                title: format!("section-{idx}"),
                content: taken,
                is_code,
                line_start: chunk_line_start,
                line_end: chunk_line_end,
            });
            chunk_line_start = cumulative_lines + 1;
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para_trimmed);
        cumulative_lines += para_lines + 2; // +2 for the \n\n separator
    }

    if !current.is_empty() {
        // If the accumulated content is oversized (no \n\n splits helped), use newline fallback
        if current.len() > MAX_CHUNK_BYTES {
            split_at_newlines_with_lines(
                &format!("section-{}", idx + 1),
                &current,
                &mut chunks,
                chunk_line_start,
            );
        } else {
            idx += 1;
            let total_lines = content.lines().count() as u32;
            let is_code = detect_code(&current);
            chunks.push(Chunk {
                title: format!("section-{idx}"),
                content: current,
                is_code,
                line_start: chunk_line_start,
                line_end: total_lines.max(chunk_line_start),
            });
        }
    }

    chunks
}

/// Flush accumulated content as one or more chunks (splitting if oversized), with line info
fn flush_chunk_with_lines(
    chunks: &mut Vec<Chunk>,
    title: &str,
    content: &str,
    line_start: u32,
    line_end: u32,
) {
    if content.len() <= MAX_CHUNK_BYTES {
        chunks.push(Chunk {
            title: title.to_string(),
            content: content.to_string(),
            is_code: detect_code(content),
            line_start,
            line_end,
        });
        return;
    }

    // Try splitting at paragraph boundaries first
    let paragraphs: Vec<&str> = content.split("\n\n").collect();

    // If there's only one "paragraph" (no \n\n), go straight to newline splitting
    if paragraphs.len() <= 1 {
        split_at_newlines_with_lines(title, content, chunks, line_start);
        return;
    }

    let mut current = String::new();
    let mut part = 0u32;
    // Track which line within content we're at for distributing line ranges
    let mut para_line_offset: u32 = 0;
    let mut part_line_start = line_start;

    for para in &paragraphs {
        let para_line_count = para.lines().count() as u32;

        if current.len() + para.len() + 2 > MAX_CHUNK_BYTES && !current.is_empty() {
            part += 1;
            let part_line_end = (line_start + para_line_offset)
                .saturating_sub(1)
                .max(part_line_start);
            let taken = std::mem::take(&mut current);
            if taken.len() > MAX_CHUNK_BYTES {
                split_at_newlines_with_lines(
                    &format!("{title} (part {part})"),
                    &taken,
                    chunks,
                    part_line_start,
                );
            } else {
                let is_code = detect_code(&taken);
                chunks.push(Chunk {
                    title: format!("{title} (part {part})"),
                    content: taken,
                    is_code,
                    line_start: part_line_start,
                    line_end: part_line_end,
                });
            }
            part_line_start = line_start + para_line_offset;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
        para_line_offset += para_line_count + 2; // +2 for \n\n
    }

    if !current.is_empty() {
        part += 1;
        let part_title = if part > 1 {
            format!("{title} (part {part})")
        } else {
            title.to_string()
        };
        if current.len() > MAX_CHUNK_BYTES {
            split_at_newlines_with_lines(&part_title, &current, chunks, part_line_start);
        } else {
            let is_code = detect_code(&current);
            chunks.push(Chunk {
                title: part_title,
                content: current,
                is_code,
                line_start: part_line_start,
                line_end,
            });
        }
    }
}

/// Hard split at single newline boundaries with line number tracking
fn split_at_newlines_with_lines(
    title: &str,
    content: &str,
    chunks: &mut Vec<Chunk>,
    base_line_start: u32,
) {
    let mut current = String::new();
    let mut part = 0u32;
    let mut line_offset: u32 = 0;
    let mut part_line_start = base_line_start;

    for line in content.lines() {
        line_offset += 1;
        // +1 for the newline we'll add
        if current.len() + line.len() + 1 > MAX_CHUNK_BYTES && !current.is_empty() {
            part += 1;
            // line_offset is 1-based and has already advanced past the overflow line,
            // so subtract 2 to get the last line actually included in the current chunk.
            let part_line_end = (base_line_start + line_offset)
                .saturating_sub(2)
                .max(part_line_start);
            let taken = std::mem::take(&mut current);
            let is_code = detect_code(&taken);
            chunks.push(Chunk {
                title: format!("{title} (part {part})"),
                content: taken,
                is_code,
                line_start: part_line_start,
                line_end: part_line_end,
            });
            part_line_start = base_line_start + line_offset - 1;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);

        // Handle single lines longer than MAX_CHUNK_BYTES — hard byte split
        while current.len() > MAX_CHUNK_BYTES {
            // Find a char boundary at or before MAX_CHUNK_BYTES
            let mut split_at = MAX_CHUNK_BYTES;
            while split_at > 0 && !current.is_char_boundary(split_at) {
                split_at -= 1;
            }
            if split_at == 0 {
                split_at = current.len().min(MAX_CHUNK_BYTES);
            }
            part += 1;
            let piece: String = current.drain(..split_at).collect();
            let is_code = detect_code(&piece);
            chunks.push(Chunk {
                title: format!("{title} (part {part})"),
                content: piece,
                is_code,
                line_start: part_line_start,
                line_end: base_line_start + line_offset - 1,
            });
            part_line_start = base_line_start + line_offset - 1;
        }
    }

    if !current.is_empty() {
        part += 1;
        let part_line_end = base_line_start + line_offset - 1;
        let is_code = detect_code(&current);
        chunks.push(Chunk {
            title: if part > 1 {
                format!("{title} (part {part})")
            } else {
                title.to_string()
            },
            content: current,
            is_code,
            line_start: part_line_start,
            line_end: part_line_end,
        });
    }
}

/// Heuristic: does this content look like code?
fn detect_code(content: &str) -> bool {
    // Contains code fences
    if content.contains("```") {
        return true;
    }

    let mut total_lines: u32 = 0;
    let mut code_lines: u32 = 0;

    for line in content.lines() {
        total_lines += 1;
        let trimmed = line.trim();
        if trimmed.starts_with("fn ")
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
        {
            code_lines += 1;
        }
    }

    if total_lines == 0 {
        return false;
    }

    code_lines as f64 / total_lines as f64 > 0.6
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
        assert!(!detect_code(
            "This is just a regular sentence about things."
        ));
    }

    #[test]
    fn test_code_blocks_stay_intact() {
        let md = "# Example\n```rust\nfn main() {\n    // code\n}\n```\nSome text after.";
        let chunks = chunk_content(md);
        // The code block should be in the same chunk as its heading
        assert!(chunks[0].content.contains("fn main()"));
    }

    #[test]
    fn test_markdown_false_positive_rejected_for_css() {
        let css_content = ".header {\n    color: #FFD700;\n    background: #1a1a1a;\n}\n\n* {\n    margin: 0;\n    padding: 0;\n}\n\n.container {\n    max-width: 1400px;\n}";
        assert!(!looks_like_markdown(css_content));
    }

    #[test]
    fn test_markdown_false_positive_rejected_for_php() {
        let php = "<?php\nrequire_once 'config.php';\n$db = db();\n$result = $db->query('SELECT * FROM users');\nif (!$result) {\n    die('Error');\n}\n?>";
        assert!(!looks_like_markdown(php));
        assert!(looks_like_code(php));
    }

    #[test]
    fn test_looks_like_code_detects_php() {
        let php = "<?php\nfunction getUser($id) {\n    return $db->query('SELECT * FROM users WHERE id = ?', [$id]);\n}\n?>";
        assert!(looks_like_code(php));
    }

    #[test]
    fn test_looks_like_code_detects_html_with_script() {
        let html = "<html>\n<head>\n<script>\nfunction init() {\n    console.log('hello');\n}\n</script>\n</head>\n</html>";
        assert!(looks_like_code(html));
    }

    #[test]
    fn test_structural_boundary_detection() {
        assert!(is_structural_boundary("function getUser($id) {"));
        assert!(is_structural_boundary("public function handle() {"));
        assert!(is_structural_boundary("class UserController {"));
        assert!(is_structural_boundary("<script>"));
        assert!(is_structural_boundary("<style>"));
        assert!(is_structural_boundary("<?php"));
        assert!(is_structural_boundary("def process(data):"));
        assert!(is_structural_boundary("pub fn new() -> Self {"));
        assert!(is_structural_boundary("  const handler = () => {"));
        assert!(!is_structural_boundary("    $x = 1;"));
        assert!(!is_structural_boundary("    // just a comment"));
    }

    #[test]
    fn test_code_chunking_produces_small_chunks() {
        // Build a PHP-like file with multiple functions
        let mut content = String::from("<?php\nrequire 'config.php';\n\n");
        for i in 0..20 {
            content.push_str(&format!(
                "function func_{i}() {{\n    $x = {i};\n    return $x * 2;\n}}\n\n"
            ));
        }
        content.push_str("?>");

        let chunks = chunk_content(&content);

        // Should produce multiple chunks, not 1-3
        assert!(
            chunks.len() >= 5,
            "Expected >=5 chunks, got {}",
            chunks.len()
        );

        // Every chunk must respect MAX_CHUNK_BYTES
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= MAX_CHUNK_BYTES,
                "Chunk '{}' is {} bytes, exceeds {}",
                chunk.title,
                chunk.content.len(),
                MAX_CHUNK_BYTES
            );
        }
    }

    #[test]
    fn test_no_chunk_exceeds_max_bytes_dense_content() {
        // Dense content with no blank lines — simulates minified-ish code
        let mut content = String::new();
        for i in 0..500 {
            content.push_str(&format!("var x_{i} = {i};\n"));
        }

        let chunks = chunk_content(&content);

        for chunk in &chunks {
            assert!(
                chunk.content.len() <= MAX_CHUNK_BYTES,
                "Chunk '{}' is {} bytes, exceeds {}",
                chunk.title,
                chunk.content.len(),
                MAX_CHUNK_BYTES
            );
        }
    }

    #[test]
    fn test_no_chunk_exceeds_max_bytes_no_newlines() {
        // Single long line with no newlines at all
        let content = "x".repeat(10000);
        let chunks = chunk_content(&content);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= MAX_CHUNK_BYTES,
                "Chunk '{}' is {} bytes, exceeds {}",
                chunk.title,
                chunk.content.len(),
                MAX_CHUNK_BYTES
            );
        }
    }

    #[test]
    fn test_real_markdown_still_detected() {
        let md = "# Title\n\nSome paragraph text.\n\n## Section Two\n\nMore text here.\n\n### Subsection\n\n- Item one\n- Item two\n";
        assert!(looks_like_markdown(md));
    }

    #[test]
    fn test_code_chunks_have_line_numbers() {
        // Build a Rust-like file with multiple functions
        let content = "fn first() {\n    let x = 1;\n    let y = 2;\n    let z = x + y;\n}\n\nfn second() {\n    let a = 10;\n    let b = 20;\n    let c = a + b;\n}\n\nfn third() {\n    let p = 100;\n    let q = 200;\n    let r = p + q;\n}";
        // Force code detection by making it look like code
        let chunks = chunk_code(content);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                chunk.line_start > 0,
                "line_start should be > 0 for code chunks"
            );
            assert!(
                chunk.line_end >= chunk.line_start,
                "line_end should be >= line_start"
            );
        }
        // First chunk should start at line 1
        assert_eq!(
            chunks[0].line_start, 1,
            "First chunk should start at line 1"
        );
    }

    #[test]
    fn test_markdown_chunks_have_line_numbers() {
        let md = "# Section One\nFirst section content.\nMore content here.\n## Section Two\nSecond section content.\nEven more content.\n### Subsection\nSubsection content.";
        let chunks = chunk_markdown(md);
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks from markdown with headings"
        );
        // First chunk should start at line 1
        assert_eq!(
            chunks[0].line_start, 1,
            "First markdown chunk should start at line 1"
        );
        // Each chunk should have valid line numbers
        for chunk in &chunks {
            assert!(chunk.line_start > 0, "line_start should be > 0");
            assert!(
                chunk.line_end >= chunk.line_start,
                "line_end should be >= line_start"
            );
        }
        // Chunks should be in ascending order
        for i in 1..chunks.len() {
            assert!(
                chunks[i].line_start > chunks[i - 1].line_start,
                "Chunk line_start values should be strictly increasing"
            );
        }
    }

    #[test]
    fn test_plain_text_chunks_have_line_numbers() {
        let text = "First paragraph with some text.\nMore lines here.\n\nSecond paragraph with text.\nAdditional lines.\n\nThird paragraph.";
        let chunks = chunk_plain_text(text);
        assert!(!chunks.is_empty());
        // First chunk should start at line 1
        assert_eq!(
            chunks[0].line_start, 1,
            "First plain text chunk should start at line 1"
        );
        for chunk in &chunks {
            assert!(chunk.line_start > 0, "line_start should be > 0");
            assert!(
                chunk.line_end >= chunk.line_start,
                "line_end should be >= line_start"
            );
        }
    }

    #[test]
    fn test_line_numbers_cover_full_content() {
        let content = "# Introduction\nLine 2.\nLine 3.\n## Main Section\nLine 5.\nLine 6.\nLine 7.\n## Conclusion\nLine 9.\nLine 10.";
        let total_lines = content.lines().count() as u32;
        let chunks = chunk_markdown(content);

        assert!(!chunks.is_empty());
        // First chunk starts at line 1
        assert_eq!(chunks[0].line_start, 1, "Coverage should start at line 1");
        // Last chunk ends at or near total line count
        let last = chunks.last().unwrap();
        assert!(
            last.line_end >= total_lines,
            "Last chunk line_end ({}) should cover total lines ({})",
            last.line_end,
            total_lines
        );
    }
}
