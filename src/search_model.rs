use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchProfile {
    Default,
    CodeReview,
    Debug,
    Planning,
    VaultNavigation,
}

impl SearchProfile {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("") => Ok(Self::Default),
            Some("code_review") => Ok(Self::CodeReview),
            Some("debug") => Ok(Self::Debug),
            Some("planning") => Ok(Self::Planning),
            Some("vault_navigation") => Ok(Self::VaultNavigation),
            Some(other) => Err(anyhow!(
                "Unknown profile '{other}'. Expected one of: code_review, debug, planning, vault_navigation"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::CodeReview => "code_review",
            Self::Debug => "debug",
            Self::Planning => "planning",
            Self::VaultNavigation => "vault_navigation",
        }
    }

    pub fn is_explicit(self) -> bool {
        !matches!(self, Self::Default)
    }

    pub fn effective_query(self, query: &str) -> String {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        match self {
            Self::Default | Self::CodeReview => trimmed.to_string(),
            Self::Debug => {
                let lower = trimmed.to_lowercase();
                let mut extras = Vec::new();
                if lower.contains("panic") || lower.contains("exception") || lower.contains("error")
                {
                    extras.push("failure");
                    extras.push("root cause");
                }
                if lower.contains("trace") || lower.contains("stack") {
                    extras.push("stack trace");
                }
                if extras.is_empty() {
                    trimmed.to_string()
                } else {
                    format!("{trimmed} {}", extras.join(" "))
                }
            }
            Self::Planning => format!("{trimmed} architecture plan steps"),
            Self::VaultNavigation => format!("{trimmed} related notes links"),
        }
    }

    pub fn keyword_weight_multiplier(self) -> f64 {
        match self {
            Self::Default => 1.0,
            Self::CodeReview => 1.35,
            Self::Debug => 1.1,
            Self::Planning => 1.0,
            Self::VaultNavigation => 0.95,
        }
    }

    pub fn vector_weight_multiplier(self) -> f64 {
        match self {
            Self::Default => 1.0,
            Self::CodeReview => 0.9,
            Self::Debug => 1.0,
            Self::Planning => 1.1,
            Self::VaultNavigation => 1.2,
        }
    }

    pub fn top_snippet_override(self) -> Option<usize> {
        match self {
            Self::Default => None,
            Self::CodeReview => Some(1400),
            Self::Debug => Some(1800),
            Self::Planning => Some(2400),
            Self::VaultNavigation => Some(1200),
        }
    }

    pub fn secondary_snippet_override(self) -> Option<usize> {
        match self {
            Self::Default => None,
            Self::CodeReview => Some(600),
            Self::Debug => Some(900),
            Self::Planning => Some(1400),
            Self::VaultNavigation => Some(700),
        }
    }

    pub fn knowledge_bias_multiplier(self) -> f64 {
        match self {
            Self::Default => 1.0,
            Self::CodeReview => 1.0,
            Self::Debug => 1.05,
            Self::Planning => 1.15,
            Self::VaultNavigation => 1.25,
        }
    }

    pub fn graph_boost(self, scoped_filtering: bool) -> f64 {
        match self {
            Self::Default => 0.0,
            Self::CodeReview => {
                if scoped_filtering {
                    0.0015
                } else {
                    0.0
                }
            }
            Self::Debug => 0.0015,
            Self::Planning => 0.003,
            Self::VaultNavigation => 0.0045,
        }
    }

    pub fn inject_neighbors(self) -> bool {
        matches!(self, Self::Planning | Self::VaultNavigation)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerScores {
    pub bm25: f64,
    pub trigram: f64,
    pub fuzzy: f64,
    pub vector: f64,
    pub proximity: f64,
    pub knowledge_bias: f64,
    pub graph_boost: f64,
}

impl LayerScores {
    pub fn matched_layers(&self) -> Vec<&'static str> {
        let mut layers = Vec::new();
        if self.bm25 > 0.0 {
            layers.push("bm25");
        }
        if self.trigram > 0.0 {
            layers.push("trigram");
        }
        if self.fuzzy > 0.0 {
            layers.push("fuzzy");
        }
        if self.vector > 0.0 {
            layers.push("vector");
        }
        if self.proximity > 0.0 {
            layers.push("proximity");
        }
        if self.knowledge_bias > 0.0 {
            layers.push("knowledge-bias");
        }
        if self.graph_boost > 0.0 {
            layers.push("graph");
        }
        layers
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchWhy {
    pub layers: LayerScores,
    pub metadata_match: bool,
    pub graph_neighbor: bool,
    pub memory_query_hint: bool,
    pub memory_source_hint: bool,
    pub memory_filter_hint: bool,
}

impl SearchWhy {
    pub fn compact_line(&self) -> String {
        let mut parts: Vec<String> = self
            .layers
            .matched_layers()
            .into_iter()
            .map(str::to_string)
            .collect();

        if self.metadata_match {
            parts.push("metadata".to_string());
        }
        if self.graph_neighbor {
            parts.push("graph-neighbor".to_string());
        }
        if self.memory_query_hint {
            parts.push("memory-query".to_string());
        }
        if self.memory_source_hint {
            parts.push("memory-source".to_string());
        }
        if self.memory_filter_hint {
            parts.push("memory-filter".to_string());
        }

        if parts.is_empty() {
            "why: ranking-defaults".to_string()
        } else {
            format!("why: {}", parts.join(", "))
        }
    }

    pub fn detailed_line(&self, final_score: f64) -> String {
        let mut parts = Vec::new();
        if self.layers.bm25 > 0.0 {
            parts.push(format!("bm25={:.4}", self.layers.bm25));
        }
        if self.layers.trigram > 0.0 {
            parts.push(format!("trigram={:.4}", self.layers.trigram));
        }
        if self.layers.fuzzy > 0.0 {
            parts.push(format!("fuzzy={:.4}", self.layers.fuzzy));
        }
        if self.layers.vector > 0.0 {
            parts.push(format!("vector={:.4}", self.layers.vector));
        }
        if self.layers.proximity > 0.0 {
            parts.push(format!("proximity={:.4}", self.layers.proximity));
        }
        if self.layers.knowledge_bias > 0.0 {
            parts.push(format!("knowledge-bias={:.4}", self.layers.knowledge_bias));
        }
        if self.layers.graph_boost > 0.0 {
            parts.push(format!("graph={:.4}", self.layers.graph_boost));
        }
        if self.metadata_match {
            parts.push("metadata=true".to_string());
        }
        if self.graph_neighbor {
            parts.push("graph-neighbor=true".to_string());
        }
        if self.memory_query_hint {
            parts.push("memory-query=true".to_string());
        }
        if self.memory_source_hint {
            parts.push("memory-source=true".to_string());
        }
        if self.memory_filter_hint {
            parts.push("memory-filter=true".to_string());
        }
        parts.push(format!("final={final_score:.4}"));
        format!("why: {}", parts.join(", "))
    }
}

pub fn normalize_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}
