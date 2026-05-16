use anyhow::Result;

use crate::config::Config;
use crate::knowledge::filter::parse_filter;
use crate::knowledge::{KnowledgeStore, LinkDirection};
use crate::search_memory::{self, SearchRunHit};
use crate::search_model::SearchProfile;
use crate::store::search::SearchWeights;
use crate::store::ContentStore;

#[derive(Debug, Clone)]
pub struct QueryRequest<'a> {
    pub query: &'a str,
    pub source: Option<&'a str>,
    pub content_type: Option<&'a str>,
    pub knowledge_filter: Option<&'a str>,
    pub limit: u32,
    pub snippet_bytes: usize,
    pub profile: SearchProfile,
    pub explain: bool,
    pub use_memory: bool,
    pub include_knowledge: bool,
}

#[derive(Debug, Clone)]
pub struct QueryOutcome {
    pub text: String,
    pub visible_bytes: u64,
    pub total_chunks: usize,
    pub estimated_bytes: u64,
    pub sources_matched: Vec<String>,
    pub profile: SearchProfile,
    pub memory_applied: bool,
}

#[derive(Debug, Clone)]
struct UnifiedResult {
    title: String,
    content: String,
    score: f64,
    source_label: String,
    source_type: &'static str,
    file_path: Option<String>,
    source_id: Option<i64>,
    line_info: String,
    why: crate::search_model::SearchWhy,
    raw_len: usize,
}

pub fn run_query(
    request: QueryRequest<'_>,
    config: &Config,
    store: &ContentStore,
    session_conn: &rusqlite::Connection,
    knowledge_store: Option<&KnowledgeStore>,
    count_only: bool,
) -> Result<QueryOutcome> {
    let explicit_profile = request.profile.is_explicit();
    let mut effective_query = if explicit_profile {
        request.profile.effective_query(request.query)
    } else {
        request.query.trim().to_string()
    };
    let mut source_hint = request.source.map(str::to_string);
    let mut filter_hint = request.knowledge_filter.map(str::to_string);
    let mut memory_applied = false;

    if request.use_memory && explicit_profile {
        if let Some(ks) = knowledge_store {
            if let Some(memory) =
                search_memory::lookup_memory(ks.conn(), request.query, request.profile)?
            {
                let memory_source_hint = memory
                    .learned_source_hint
                    .clone()
                    .or(memory.winning_source_label.clone());
                if effective_query == request.query.trim()
                    && memory.learned_query != request.query.trim()
                {
                    effective_query = memory.learned_query;
                    memory_applied = true;
                }
                if source_hint.is_none() && memory_source_hint.is_some() {
                    source_hint = memory_source_hint;
                    memory_applied = true;
                }
                if filter_hint.is_none() && memory.learned_filter_hint.is_some() {
                    filter_hint = memory.learned_filter_hint;
                    memory_applied = true;
                }
            }
        }
    }

    let mut weights = SearchWeights {
        keyword_weight: config.search.keyword_weight * request.profile.keyword_weight_multiplier(),
        vector_weight: config.search.vector_weight * request.profile.vector_weight_multiplier(),
    };
    if !explicit_profile {
        weights = SearchWeights {
            keyword_weight: config.search.keyword_weight,
            vector_weight: config.search.vector_weight,
        };
    }

    let session_results = store.search_with_weights(
        &effective_query,
        request.limit,
        source_hint.as_deref(),
        request.content_type,
        &weights,
    )?;

    let mut knowledge_results = if request.include_knowledge {
        if let Some(ks) = knowledge_store {
            let mut results = crate::knowledge::search::knowledge_search(
                ks,
                &effective_query,
                filter_hint.as_deref(),
                source_hint.as_deref(),
                request.limit,
                &weights,
                request.snippet_bytes,
            )?;
            let scoped = source_hint.is_some() || filter_hint.is_some();
            apply_knowledge_bias(&mut results, request.profile);
            apply_graph_rerank(
                ks,
                &mut results,
                request.profile,
                scoped,
                filter_hint.as_deref(),
            )?;
            if filter_hint.is_some() {
                for result in &mut results {
                    result.why.metadata_match = true;
                }
            }
            results
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if memory_applied {
        for result in &mut knowledge_results {
            if effective_query != request.query.trim() {
                result.why.memory_query_hint = true;
            }
            if source_hint.as_deref() != request.source {
                result.why.memory_source_hint = true;
            }
            if filter_hint.as_deref() != request.knowledge_filter {
                result.why.memory_filter_hint = true;
            }
        }
    }

    let mut unified = Vec::new();
    let top_snippet_bytes = request
        .profile
        .top_snippet_override()
        .unwrap_or(request.snippet_bytes);
    let secondary_snippet_bytes = request
        .profile
        .secondary_snippet_override()
        .unwrap_or(config.search.secondary_snippet_bytes);
    let top_result_count = config.search.top_result_count;

    for mut result in session_results {
        if memory_applied {
            if effective_query != request.query.trim() {
                result.why.memory_query_hint = true;
            }
            if source_hint.as_deref() != request.source {
                result.why.memory_source_hint = true;
            }
        }
        unified.push(UnifiedResult {
            title: result.title,
            content: result.content.clone(),
            score: result.score,
            source_label: result.source,
            source_type: "session",
            file_path: None,
            source_id: Some(result.source_id),
            line_info: format_lines(result.line_start, result.line_end),
            why: result.why,
            raw_len: result.content.len(),
        });
    }

    for mut result in knowledge_results {
        if memory_applied {
            if effective_query != request.query.trim() {
                result.why.memory_query_hint = true;
            }
            if source_hint.as_deref() != request.source {
                result.why.memory_source_hint = true;
            }
            if filter_hint.as_deref() != request.knowledge_filter {
                result.why.memory_filter_hint = true;
            }
        }
        unified.push(UnifiedResult {
            title: result.title,
            content: result.snippet.clone(),
            score: result.score,
            source_label: result.source_label,
            source_type: "knowledge",
            file_path: Some(result.file_path),
            source_id: Some(result.source_id),
            line_info: format!(" (lines {})", result.lines),
            why: result.why,
            raw_len: result.snippet.len(),
        });
    }

    unified.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    unified.truncate(request.limit as usize);

    let mut sources_matched: Vec<String> = unified
        .iter()
        .map(|r| r.source_label.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sources_matched.sort();

    let total_chunks = unified.len();
    let estimated_bytes = unified.iter().map(|r| r.raw_len as u64).sum();

    if count_only {
        let estimated_tokens = (estimated_bytes as f64 / 4.0).ceil() as u32;
        let recommendation = if estimated_tokens <= 20_000 {
            "inline"
        } else {
            "delegate"
        };
        let text = serde_json::to_string_pretty(&serde_json::json!({
            "count_only": true,
            "query": request.query,
            "profile": request.profile.as_str(),
            "memory_applied": memory_applied,
            "total_chunks": total_chunks,
            "estimated_bytes": estimated_bytes,
            "estimated_tokens": estimated_tokens,
            "sources_matched": sources_matched,
            "recommendation": recommendation
        }))?;
        return Ok(QueryOutcome {
            text,
            visible_bytes: 0,
            total_chunks,
            estimated_bytes,
            sources_matched,
            profile: request.profile,
            memory_applied,
        });
    }

    let text = format_results(
        request.query,
        &effective_query,
        request.explain,
        memory_applied,
        &unified,
        top_snippet_bytes,
        secondary_snippet_bytes,
        top_result_count,
    );

    let hits: Vec<SearchRunHit> = unified
        .iter()
        .map(|result| SearchRunHit {
            source_type: result.source_type.to_string(),
            source_label: result.source_label.clone(),
            file_path: result.file_path.clone(),
            title: result.title.clone(),
        })
        .collect();
    let _ = search_memory::record_search_run(
        session_conn,
        request.query,
        request.profile,
        &effective_query,
        source_hint.as_deref(),
        filter_hint.as_deref(),
        memory_applied,
        &hits,
    )?;

    Ok(QueryOutcome {
        visible_bytes: estimate_visible_bytes(
            &unified,
            top_snippet_bytes,
            secondary_snippet_bytes,
            top_result_count,
        ),
        text,
        total_chunks,
        estimated_bytes,
        sources_matched,
        profile: request.profile,
        memory_applied,
    })
}

fn apply_knowledge_bias(
    results: &mut [crate::knowledge::search::KnowledgeSearchResult],
    profile: SearchProfile,
) {
    if matches!(profile, SearchProfile::Default) {
        return;
    }
    let multiplier = profile.knowledge_bias_multiplier();
    for result in results {
        let boosted = result.score * multiplier;
        result.why.layers.knowledge_bias += boosted - result.score;
        result.score = boosted;
    }
}

fn apply_graph_rerank(
    ks: &KnowledgeStore,
    results: &mut Vec<crate::knowledge::search::KnowledgeSearchResult>,
    profile: SearchProfile,
    scoped_filtering: bool,
    filter_hint: Option<&str>,
) -> Result<()> {
    let boost = profile.graph_boost(scoped_filtering);
    if boost <= 0.0 || results.is_empty() {
        return Ok(());
    }

    let predicates = parse_filter(filter_hint);
    let preds = if predicates.is_empty() {
        None
    } else {
        Some(predicates.as_slice())
    };

    let seed_ids: Vec<i64> = results.iter().take(3).map(|r| r.file_id).collect();
    let mut linked_chunks = Vec::new();
    for file_id in seed_ids {
        linked_chunks.extend(ks.get_related_chunks_filtered(
            file_id,
            LinkDirection::Both,
            6,
            preds,
        )?);
    }

    let mut neighbor_by_file: std::collections::HashMap<i64, crate::knowledge::RelatedChunk> =
        std::collections::HashMap::new();
    for chunk in linked_chunks {
        neighbor_by_file.entry(chunk.file_id).or_insert(chunk);
    }

    let existing_files: std::collections::HashSet<i64> =
        results.iter().map(|r| r.file_id).collect();
    for result in results.iter_mut() {
        if neighbor_by_file.contains_key(&result.file_id) {
            result.score += boost;
            result.why.layers.graph_boost += boost;
            result.why.graph_neighbor = true;
        }
    }

    if profile.inject_neighbors() && !neighbor_by_file.is_empty() {
        let baseline = results.last().map(|r| r.score * 0.92).unwrap_or(boost);
        let mut injected = 0usize;
        for (file_id, chunk) in neighbor_by_file {
            if injected >= 2 || existing_files.contains(&file_id) {
                continue;
            }
            let (source_label, source_id) = knowledge_source_for_file(ks, file_id)?;
            let mut why = crate::search_model::SearchWhy::default();
            why.layers.graph_boost = boost;
            why.graph_neighbor = true;
            results.push(crate::knowledge::search::KnowledgeSearchResult {
                title: chunk.title,
                snippet: chunk.content,
                score: baseline + boost,
                source_label,
                file_path: chunk.rel_path,
                file_id,
                source_id,
                lines: String::new(),
                metadata: serde_json::Value::Object(Default::default()),
                source_type: "knowledge".to_string(),
                why,
            });
            injected += 1;
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(())
}

fn knowledge_source_for_file(ks: &KnowledgeStore, file_id: i64) -> Result<(String, i64)> {
    Ok(ks.conn().query_row(
        "SELECT ks.label, ks.id
         FROM knowledge_files kf
         JOIN knowledge_sources ks ON ks.id = kf.source_id
         WHERE kf.id = ?1",
        rusqlite::params![file_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?)
}

fn format_results(
    original_query: &str,
    effective_query: &str,
    explain: bool,
    memory_applied: bool,
    results: &[UnifiedResult],
    top_snippet_bytes: usize,
    secondary_snippet_bytes: usize,
    top_result_count: usize,
) -> String {
    let mut output = String::new();
    let mut truncated_any = false;
    output.push_str(&format!("### \"{original_query}\"\n"));
    if memory_applied {
        output.push_str(&format!(
            "Search memory adjusted this query plan. Effective query: \"{effective_query}\".\n\n"
        ));
    }

    if results.is_empty() {
        output.push_str("No results.\n");
        return output;
    }

    for (idx, result) in results.iter().enumerate() {
        let max_bytes = if idx < top_result_count {
            top_snippet_bytes
        } else {
            secondary_snippet_bytes
        };
        let snippet = crate::truncate::preview(&result.content, max_bytes);
        if result.content.len() > max_bytes {
            truncated_any = true;
        }
        match result.source_type {
            "session" => {
                output.push_str(&format!(
                    "**[{}]** (source: {}, source_id: {}, type: session, score: {:.4}{})\n{}\n{}\n\n",
                    result.title,
                    result.source_label,
                    result.source_id.unwrap_or_default(),
                    result.score,
                    result.line_info,
                    result.why.compact_line(),
                    snippet
                ));
            }
            _ => {
                output.push_str(&format!(
                    "**[{}]** (source: {}, type: knowledge, file: {}, score: {:.4}{})\n{}\n{}\n\n",
                    result.title,
                    result.source_label,
                    result.file_path.as_deref().unwrap_or(""),
                    result.score,
                    result.line_info,
                    result.why.compact_line(),
                    snippet
                ));
            }
        }
        if explain {
            output.push_str(&format!("{}\n\n", result.why.detailed_line(result.score)));
        }
    }

    if truncated_any {
        output.push_str("Some results were truncated. Use bpx_read_chunks with the source label to see full content.\n");
    }

    output
}

fn estimate_visible_bytes(
    results: &[UnifiedResult],
    top_snippet_bytes: usize,
    secondary_snippet_bytes: usize,
    top_result_count: usize,
) -> u64 {
    results
        .iter()
        .enumerate()
        .map(|(idx, result)| {
            let max_bytes = if idx < top_result_count {
                top_snippet_bytes
            } else {
                secondary_snippet_bytes
            };
            result.content.len().min(max_bytes) as u64
        })
        .sum()
}

fn format_lines(line_start: u32, line_end: u32) -> String {
    if line_start > 0 {
        format!(" (lines {}-{})", line_start, line_end)
    } else {
        String::new()
    }
}
