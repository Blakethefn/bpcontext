//! Metadata filter parser and SQL generation for knowledge search.
//!
//! Parses filter strings like `"type:task status:active"` into SQL WHERE clauses
//! that restrict search results based on enrichment metadata stored in
//! `knowledge_chunk_meta.metadata`.

/// A parsed filter predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPredicate {
    /// Filter by source label (joined to knowledge_sources, not metadata).
    Source(String),
    /// Filter by frontmatter field equality:
    /// `json_extract(metadata, '$.frontmatter.<key>') = '<value>'`
    Frontmatter { key: String, value: String },
    /// Filter by tag membership:
    /// `value IN json_each(json_extract(metadata, '$.frontmatter.tags'))`
    Tag(String),
    /// Filter by folder tag:
    /// `value IN json_each(json_extract(metadata, '$.folder_tags'))`
    Folder(String),
}

/// Validate that a metadata key contains only safe characters for JSON path
/// construction. Allows alphanumeric, underscore, and hyphen.
fn validate_key(key: &str) -> bool {
    !key.is_empty() && key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Parse a filter string into a list of predicates.
///
/// Filter syntax: space-separated `key:value` pairs. Multiple pairs are ANDed.
/// Tokens without `:` are ignored. Empty/None input returns an empty vec.
///
/// Special keys:
/// - `source:` — filters by `knowledge_sources.label`
/// - `tag:` — checks membership in `frontmatter.tags` array
/// - `folder:` — checks membership in `folder_tags` array
/// - Everything else — treated as a frontmatter field equality check
pub fn parse_filter(filter: Option<&str>) -> Vec<FilterPredicate> {
    let filter = match filter {
        Some(f) if !f.trim().is_empty() => f.trim(),
        _ => return vec![],
    };

    let mut predicates = Vec::new();

    for token in filter.split_whitespace() {
        if let Some((key, value)) = token.split_once(':') {
            if key.is_empty() || value.is_empty() {
                continue;
            }
            let pred = match key {
                "source" => FilterPredicate::Source(value.to_string()),
                "tag" => FilterPredicate::Tag(value.to_string()),
                "folder" => FilterPredicate::Folder(value.to_string()),
                other => {
                    if !validate_key(other) {
                        continue; // Skip keys with unsafe characters
                    }
                    FilterPredicate::Frontmatter {
                        key: other.to_string(),
                        value: value.to_string(),
                    }
                }
            };
            predicates.push(pred);
        }
        // Tokens without ':' are silently ignored
    }

    predicates
}

/// Generate SQL WHERE clause fragments and parameter values for a set of predicates.
///
/// Returns `(where_clause, params)` where `where_clause` is a string to append
/// after an existing WHERE (joined with ` AND `), and `params` are the bind values.
///
/// The `param_offset` is the starting `?N` index for parameter placeholders
/// (e.g., if you already have `?1` and `?2` bound, pass `param_offset = 2`).
pub fn predicates_to_sql(predicates: &[FilterPredicate], param_offset: usize) -> (String, Vec<String>) {
    let mut fragments = Vec::new();
    let mut params: Vec<String> = Vec::new();

    for pred in predicates {
        let idx = param_offset + params.len() + 1;
        match pred {
            FilterPredicate::Source(label) => {
                fragments.push(format!(
                    "kcm.source_id = (SELECT id FROM knowledge_sources WHERE label = ?{idx})"
                ));
                params.push(label.clone());
            }
            FilterPredicate::Frontmatter { key, value } => {
                // SQLite json_extract path can't be parameterized, so we build the
                // path string in Rust with a validated key and embed it in the SQL.
                // Only the value is a bound parameter.
                let json_path = format!("$.frontmatter.{key}");
                fragments.push(format!(
                    "json_extract(kcm.metadata, '{json_path}') = ?{idx}"
                ));
                params.push(value.clone());
            }
            FilterPredicate::Tag(tag) => {
                fragments.push(format!(
                    "EXISTS (SELECT 1 FROM json_each(json_extract(kcm.metadata, '$.frontmatter.tags')) WHERE value = ?{idx})"
                ));
                params.push(tag.clone());
            }
            FilterPredicate::Folder(folder) => {
                fragments.push(format!(
                    "EXISTS (SELECT 1 FROM json_each(json_extract(kcm.metadata, '$.folder_tags')) WHERE value = ?{idx})"
                ));
                params.push(folder.clone());
            }
        }
    }

    let where_clause = fragments.join(" AND ");
    (where_clause, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_filter ─────────────────────────────────────────────────────────

    #[test]
    fn parse_none_returns_empty() {
        assert!(parse_filter(None).is_empty());
    }

    #[test]
    fn parse_empty_string_returns_empty() {
        assert!(parse_filter(Some("")).is_empty());
    }

    #[test]
    fn parse_whitespace_only_returns_empty() {
        assert!(parse_filter(Some("   ")).is_empty());
    }

    #[test]
    fn parse_no_colon_tokens_ignored() {
        assert!(parse_filter(Some("invalid_no_colon")).is_empty());
    }

    #[test]
    fn parse_source_predicate() {
        let preds = parse_filter(Some("source:vault"));
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0], FilterPredicate::Source("vault".to_string()));
    }

    #[test]
    fn parse_tag_predicate() {
        let preds = parse_filter(Some("tag:implementation"));
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0], FilterPredicate::Tag("implementation".to_string()));
    }

    #[test]
    fn parse_folder_predicate() {
        let preds = parse_filter(Some("folder:01-projects"));
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0],
            FilterPredicate::Folder("01-projects".to_string())
        );
    }

    #[test]
    fn parse_frontmatter_predicate() {
        let preds = parse_filter(Some("type:task"));
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0],
            FilterPredicate::Frontmatter {
                key: "type".to_string(),
                value: "task".to_string(),
            }
        );
    }

    #[test]
    fn parse_multiple_predicates_anded() {
        let preds = parse_filter(Some("type:task status:active"));
        assert_eq!(preds.len(), 2);
        assert_eq!(
            preds[0],
            FilterPredicate::Frontmatter {
                key: "type".to_string(),
                value: "task".to_string(),
            }
        );
        assert_eq!(
            preds[1],
            FilterPredicate::Frontmatter {
                key: "status".to_string(),
                value: "active".to_string(),
            }
        );
    }

    #[test]
    fn parse_mixed_predicates() {
        let preds = parse_filter(Some("source:vault type:task tag:foo folder:01-projects"));
        assert_eq!(preds.len(), 4);
        assert_eq!(preds[0], FilterPredicate::Source("vault".to_string()));
        assert_eq!(
            preds[1],
            FilterPredicate::Frontmatter {
                key: "type".to_string(),
                value: "task".to_string(),
            }
        );
        assert_eq!(preds[2], FilterPredicate::Tag("foo".to_string()));
        assert_eq!(
            preds[3],
            FilterPredicate::Folder("01-projects".to_string())
        );
    }

    #[test]
    fn parse_skips_empty_key_or_value() {
        // ":value" has empty key, "key:" has empty value
        let preds = parse_filter(Some(":value key:"));
        assert!(preds.is_empty());
    }

    #[test]
    fn parse_key_with_special_chars_rejected() {
        // Keys with special characters should be rejected (SQL injection prevention)
        let preds = parse_filter(Some("key';DROP:evil"));
        assert!(preds.is_empty());

        let preds = parse_filter(Some("key.sub:value"));
        assert!(preds.is_empty());

        let preds = parse_filter(Some("key space:value"));
        // "key" has no colon, "space:value" is valid → 1 pred
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0],
            FilterPredicate::Frontmatter {
                key: "space".to_string(),
                value: "value".to_string(),
            }
        );
    }

    #[test]
    fn parse_key_with_hyphen_accepted() {
        let preds = parse_filter(Some("my-key:value"));
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0],
            FilterPredicate::Frontmatter {
                key: "my-key".to_string(),
                value: "value".to_string(),
            }
        );
    }

    #[test]
    fn parse_key_with_underscore_accepted() {
        let preds = parse_filter(Some("my_key:value"));
        assert_eq!(preds.len(), 1);
    }

    // ── validate_key ─────────────────────────────────────────────────────────

    #[test]
    fn validate_key_alphanumeric() {
        assert!(validate_key("type"));
        assert!(validate_key("status123"));
    }

    #[test]
    fn validate_key_with_underscore_hyphen() {
        assert!(validate_key("my_key"));
        assert!(validate_key("my-key"));
    }

    #[test]
    fn validate_key_rejects_special_chars() {
        assert!(!validate_key("key'"));
        assert!(!validate_key("key;"));
        assert!(!validate_key("key.sub"));
        assert!(!validate_key("key space"));
        assert!(!validate_key(""));
    }

    // ── predicates_to_sql ────────────────────────────────────────────────────

    #[test]
    fn sql_empty_predicates() {
        let (clause, params) = predicates_to_sql(&[], 0);
        assert!(clause.is_empty());
        assert!(params.is_empty());
    }

    #[test]
    fn sql_source_predicate() {
        let preds = vec![FilterPredicate::Source("vault".to_string())];
        let (clause, params) = predicates_to_sql(&preds, 0);
        assert!(clause.contains("knowledge_sources"));
        assert!(clause.contains("label = ?1"));
        assert_eq!(params, vec!["vault"]);
    }

    #[test]
    fn sql_frontmatter_predicate() {
        let preds = vec![FilterPredicate::Frontmatter {
            key: "type".to_string(),
            value: "task".to_string(),
        }];
        let (clause, params) = predicates_to_sql(&preds, 0);
        assert!(clause.contains("json_extract(kcm.metadata, '$.frontmatter.type')"));
        assert!(clause.contains("= ?1"));
        assert_eq!(params, vec!["task"]);
    }

    #[test]
    fn sql_tag_predicate() {
        let preds = vec![FilterPredicate::Tag("impl".to_string())];
        let (clause, params) = predicates_to_sql(&preds, 0);
        assert!(clause.contains("json_each"));
        assert!(clause.contains("'$.frontmatter.tags'"));
        assert!(clause.contains("value = ?1"));
        assert_eq!(params, vec!["impl"]);
    }

    #[test]
    fn sql_folder_predicate() {
        let preds = vec![FilterPredicate::Folder("01-projects".to_string())];
        let (clause, params) = predicates_to_sql(&preds, 0);
        assert!(clause.contains("json_each"));
        assert!(clause.contains("'$.folder_tags'"));
        assert!(clause.contains("value = ?1"));
        assert_eq!(params, vec!["01-projects"]);
    }

    #[test]
    fn sql_multiple_predicates_joined_with_and() {
        let preds = vec![
            FilterPredicate::Frontmatter {
                key: "type".to_string(),
                value: "task".to_string(),
            },
            FilterPredicate::Frontmatter {
                key: "status".to_string(),
                value: "active".to_string(),
            },
        ];
        let (clause, params) = predicates_to_sql(&preds, 0);
        assert!(clause.contains(" AND "));
        assert!(clause.contains("?1"));
        assert!(clause.contains("?2"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn sql_param_offset_respected() {
        let preds = vec![FilterPredicate::Source("vault".to_string())];
        let (clause, params) = predicates_to_sql(&preds, 3);
        // With offset 3, first param should be ?4
        assert!(clause.contains("?4"));
        assert_eq!(params.len(), 1);
    }
}
