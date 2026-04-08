//! Enrichment pipeline for knowledge store metadata extraction.
//!
//! Enrichments are optional metadata extractors that run during sync to produce
//! structured key-value pairs stored alongside each chunk. They enable filtered
//! search — e.g., "find all active task notes about the prediction engine".
//!
//! Three built-in enrichments:
//! - **frontmatter** — parses YAML between `---` fences at file start
//! - **wikilinks** — extracts `[[target]]` and `[[target|alias]]` patterns
//! - **folder_tags** — derives tags from directory path components

use serde_json::Value;
use std::collections::HashSet;
use std::path::Path;

/// Valid enrichment names.
const KNOWN_ENRICHMENTS: &[&str] = &["frontmatter", "wikilinks", "folder_tags"];

/// Build an enrichment function from a list of enabled enrichment names.
/// Returns `None` if the list is empty.
///
/// Valid names: `"frontmatter"`, `"wikilinks"`, `"folder_tags"`.
/// Unknown names are logged as warnings and ignored.
pub fn build_enrichment_fn(
    enrichments: &[String],
    source_base_path: &Path,
) -> Option<Box<dyn Fn(&str, &Path) -> Value>> {
    if enrichments.is_empty() {
        return None;
    }

    let do_frontmatter = enrichments.iter().any(|e| e == "frontmatter");
    let do_wikilinks = enrichments.iter().any(|e| e == "wikilinks");
    let do_folder_tags = enrichments.iter().any(|e| e == "folder_tags");
    let base = source_base_path.to_path_buf();

    // Log unknown enrichment names
    for e in enrichments {
        if !KNOWN_ENRICHMENTS.contains(&e.as_str()) {
            eprintln!("Warning: unknown enrichment '{}', ignoring", e);
        }
    }

    Some(Box::new(move |content: &str, path: &Path| {
        let mut obj = serde_json::Map::new();

        if do_frontmatter {
            let fm = extract_frontmatter(content);
            if !fm.is_null() && fm.as_object().map_or(false, |o| !o.is_empty()) {
                obj.insert("frontmatter".to_string(), fm);
            }
        }
        if do_wikilinks {
            let links = extract_wikilinks(content);
            if let Value::Array(ref arr) = links {
                if !arr.is_empty() {
                    obj.insert("wikilinks".to_string(), links);
                }
            }
        }
        if do_folder_tags {
            let tags = extract_folder_tags(path, &base);
            if let Value::Array(ref arr) = tags {
                if !arr.is_empty() {
                    obj.insert("folder_tags".to_string(), tags);
                }
            }
        }

        Value::Object(obj)
    }))
}

/// Extract YAML frontmatter from markdown content.
///
/// Parses the YAML block between `---` delimiters at the start of the file.
/// Returns a JSON object with the parsed fields, or an empty object if:
/// - No frontmatter found
/// - Malformed YAML
/// - YAML parses to a non-object type
pub fn extract_frontmatter(content: &str) -> Value {
    let empty = || Value::Object(serde_json::Map::new());
    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        return empty();
    }

    let after_first = &trimmed[3..];
    let close_pos = after_first.find("\n---");

    match close_pos {
        Some(pos) => {
            let yaml_str = after_first[..pos].trim();
            if yaml_str.is_empty() {
                return empty();
            }
            match serde_yaml::from_str::<Value>(yaml_str) {
                Ok(val) if val.is_object() => val,
                Ok(_) => empty(),  // Non-object YAML (e.g. bare string, list)
                Err(_) => empty(), // Malformed YAML
            }
        }
        None => empty(), // No closing ---
    }
}

/// Extract wikilinks from content.
///
/// Finds `[[target]]` and `[[target|alias]]` patterns. Returns a JSON array
/// of unique link targets (without aliases). Targets are deduplicated.
pub fn extract_wikilinks(content: &str) -> Value {
    let mut links = Vec::new();
    let mut seen = HashSet::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len.saturating_sub(1) {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            let start = i + 2;
            if let Some(end_offset) = content[start..].find("]]") {
                let link_content = &content[start..start + end_offset];
                // Extract target (before | if alias exists)
                let target = link_content.split('|').next().unwrap_or(link_content).trim();
                if !target.is_empty() && seen.insert(target.to_string()) {
                    links.push(Value::String(target.to_string()));
                }
                i = start + end_offset + 2;
            } else {
                i += 2; // No closing ]], skip
            }
        } else {
            i += 1;
        }
    }

    Value::Array(links)
}

/// Derive folder tags from a file path relative to the source base.
///
/// Returns a JSON array of directory path components between the source base
/// and the file. The filename itself is excluded.
///
/// Example: file `/vault/01-projects/bpcontext/specs/foo.md` with base `/vault`
/// yields `["01-projects", "bpcontext", "specs"]`.
pub fn extract_folder_tags(file_path: &Path, source_base: &Path) -> Value {
    let rel = match file_path.strip_prefix(source_base) {
        Ok(r) => r,
        Err(_) => return Value::Array(vec![]),
    };

    let tags: Vec<Value> = rel
        .parent()
        .map(|p| {
            p.components()
                .filter_map(|c| c.as_os_str().to_str())
                .map(|s| Value::String(s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Value::Array(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    // ── extract_frontmatter ──────────────────────────────────────────────────

    #[test]
    fn frontmatter_standard_fields() {
        let content = "---\ntype: task\nstatus: active\n---\nBody text";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({"type": "task", "status": "active"}));
    }

    #[test]
    fn frontmatter_with_tags_array() {
        let content = "---\ntags:\n  - foo\n  - bar\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({"tags": ["foo", "bar"]}));
    }

    #[test]
    fn frontmatter_with_wikilink_value() {
        let content = "---\nproject: \"[[link]]\"\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({"project": "[[link]]"}));
    }

    #[test]
    fn frontmatter_no_frontmatter() {
        let content = "No frontmatter here";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({}));
    }

    #[test]
    fn frontmatter_malformed_yaml() {
        let content = "---\ninvalid: yaml: [broken\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({}));
    }

    #[test]
    fn frontmatter_empty_between_fences() {
        let content = "---\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({}));
    }

    #[test]
    fn frontmatter_non_object_yaml() {
        let content = "---\njust a string\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({}));
    }

    #[test]
    fn frontmatter_no_closing_fence() {
        let content = "---\ntype: task\nstatus: active\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({}));
    }

    #[test]
    fn frontmatter_leading_whitespace() {
        let content = "  ---\ntype: task\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({"type": "task"}));
    }

    #[test]
    fn frontmatter_multiple_types() {
        let content = "---\ncount: 42\nactive: true\nname: test\n---\n";
        let result = extract_frontmatter(content);
        assert_eq!(result, json!({"count": 42, "active": true, "name": "test"}));
    }

    // ── extract_wikilinks ────────────────────────────────────────────────────

    #[test]
    fn wikilinks_simple() {
        let content = "See [[projects/foo]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["projects/foo"]));
    }

    #[test]
    fn wikilinks_with_alias() {
        let content = "Link [[target|alias]] here";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["target"]));
    }

    #[test]
    fn wikilinks_multiple() {
        let content = "Two [[a]] and [[b]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["a", "b"]));
    }

    #[test]
    fn wikilinks_nested_brackets() {
        // First ]] closes the link — inner [[ is just content
        let content = "Nested [[a|[[b]]]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["a"]));
    }

    #[test]
    fn wikilinks_none() {
        let content = "No links here";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!([]));
    }

    #[test]
    fn wikilinks_deduplicated() {
        let content = "Duplicate [[a]] and [[a]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["a"]));
    }

    #[test]
    fn wikilinks_unclosed() {
        let content = "Unclosed [[link";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!([]));
    }

    #[test]
    fn wikilinks_empty_target() {
        let content = "Empty [[]] link";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!([]));
    }

    #[test]
    fn wikilinks_adjacent() {
        let content = "[[a]][[b]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["a", "b"]));
    }

    #[test]
    fn wikilinks_in_frontmatter() {
        // Wikilinks inside frontmatter are still extracted (by design)
        let content = "---\nproject: \"[[myproject]]\"\n---\nSee [[other]]";
        let result = extract_wikilinks(content);
        assert_eq!(result, json!(["myproject", "other"]));
    }

    // ── extract_folder_tags ──────────────────────────────────────────────────

    #[test]
    fn folder_tags_multi_level() {
        let file = Path::new("/vault/01-projects/bpcontext/specs/foo.md");
        let base = Path::new("/vault");
        let result = extract_folder_tags(file, base);
        assert_eq!(result, json!(["01-projects", "bpcontext", "specs"]));
    }

    #[test]
    fn folder_tags_single_level() {
        let file = Path::new("/vault/tasks/task.md");
        let base = Path::new("/vault");
        let result = extract_folder_tags(file, base);
        assert_eq!(result, json!(["tasks"]));
    }

    #[test]
    fn folder_tags_root_file() {
        let file = Path::new("/vault/root-note.md");
        let base = Path::new("/vault");
        let result = extract_folder_tags(file, base);
        assert_eq!(result, json!([]));
    }

    #[test]
    fn folder_tags_path_not_under_base() {
        let file = Path::new("/other/path/file.md");
        let base = Path::new("/vault");
        let result = extract_folder_tags(file, base);
        assert_eq!(result, json!([]));
    }

    // ── build_enrichment_fn ──────────────────────────────────────────────────

    #[test]
    fn build_empty_list_returns_none() {
        let result = build_enrichment_fn(&[], Path::new("/base"));
        assert!(result.is_none());
    }

    #[test]
    fn build_all_three_enrichments() {
        let enrichments = vec![
            "frontmatter".to_string(),
            "wikilinks".to_string(),
            "folder_tags".to_string(),
        ];
        let func = build_enrichment_fn(&enrichments, Path::new("/vault"))
            .expect("should return Some");

        let content = "---\ntype: task\n---\nSee [[other]]";
        let path = Path::new("/vault/01-projects/note.md");
        let result = func(content, path);

        let obj = result.as_object().expect("should be object");
        assert!(obj.contains_key("frontmatter"));
        assert!(obj.contains_key("wikilinks"));
        assert!(obj.contains_key("folder_tags"));

        assert_eq!(obj["frontmatter"], json!({"type": "task"}));
        assert_eq!(obj["wikilinks"], json!(["other"]));
        assert_eq!(obj["folder_tags"], json!(["01-projects"]));
    }

    #[test]
    fn build_frontmatter_only() {
        let enrichments = vec!["frontmatter".to_string()];
        let func = build_enrichment_fn(&enrichments, Path::new("/base"))
            .expect("should return Some");

        let content = "---\nstatus: done\n---\nBody";
        let result = func(content, Path::new("/base/file.md"));

        let obj = result.as_object().expect("should be object");
        assert!(obj.contains_key("frontmatter"));
        assert!(!obj.contains_key("wikilinks"));
        assert!(!obj.contains_key("folder_tags"));
    }

    #[test]
    fn build_wikilinks_only() {
        let enrichments = vec!["wikilinks".to_string()];
        let func = build_enrichment_fn(&enrichments, Path::new("/base"))
            .expect("should return Some");

        let content = "Link to [[target]]";
        let result = func(content, Path::new("/base/file.md"));

        let obj = result.as_object().expect("should be object");
        assert!(!obj.contains_key("frontmatter"));
        assert!(obj.contains_key("wikilinks"));
        assert!(!obj.contains_key("folder_tags"));
    }

    #[test]
    fn build_folder_tags_only() {
        let enrichments = vec!["folder_tags".to_string()];
        let func = build_enrichment_fn(&enrichments, Path::new("/vault"))
            .expect("should return Some");

        let content = "No links, no frontmatter";
        let result = func(content, Path::new("/vault/deep/path/file.md"));

        let obj = result.as_object().expect("should be object");
        assert!(!obj.contains_key("frontmatter"));
        assert!(!obj.contains_key("wikilinks"));
        assert!(obj.contains_key("folder_tags"));
        assert_eq!(obj["folder_tags"], json!(["deep", "path"]));
    }

    #[test]
    fn build_unknown_enrichment_ignored() {
        // Unknown enrichments are ignored; valid ones still work
        let enrichments = vec![
            "frontmatter".to_string(),
            "nonexistent".to_string(),
        ];
        let func = build_enrichment_fn(&enrichments, Path::new("/base"))
            .expect("should return Some");

        let content = "---\ntype: test\n---\nBody";
        let result = func(content, Path::new("/base/file.md"));

        let obj = result.as_object().expect("should be object");
        assert!(obj.contains_key("frontmatter"));
        assert_eq!(obj["frontmatter"], json!({"type": "test"}));
    }

    #[test]
    fn build_only_unknown_returns_empty_object() {
        let enrichments = vec!["bogus".to_string()];
        let func = build_enrichment_fn(&enrichments, Path::new("/base"))
            .expect("should return Some for non-empty list");

        let content = "Hello world";
        let result = func(content, Path::new("/base/file.md"));
        assert_eq!(result, json!({}));
    }

    #[test]
    fn build_omits_empty_enrichment_keys() {
        // When content has no frontmatter and no wikilinks, those keys
        // should be absent from the output (not present with empty values).
        let enrichments = vec![
            "frontmatter".to_string(),
            "wikilinks".to_string(),
            "folder_tags".to_string(),
        ];
        let func = build_enrichment_fn(&enrichments, Path::new("/vault"))
            .expect("should return Some");

        let content = "Plain text, no frontmatter, no links";
        let path = Path::new("/vault/root.md");
        let result = func(content, path);

        let obj = result.as_object().expect("should be object");
        // No frontmatter → key absent
        assert!(!obj.contains_key("frontmatter"));
        // No wikilinks → key absent
        assert!(!obj.contains_key("wikilinks"));
        // Root file → no folder tags → key absent
        assert!(!obj.contains_key("folder_tags"));
    }
}
