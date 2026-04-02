use serde_json::{json, Value};

/// Generate the MCP tools/list response with all bpx_ tool definitions
pub fn tool_definitions() -> Value {
    json!({
        "tools": [
            {
                "name": "bpx_execute",
                "description": "Run a shell command, index the output, and return a preview. Use this instead of Bash for commands producing large output.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        },
                        "label": {
                            "type": "string",
                            "description": "Label for indexing (defaults to the command)"
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "description": "Timeout in milliseconds (default: 120000)"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "bpx_batch_execute",
                "description": "Run multiple commands and search in one call. Replaces multiple bpx_execute + bpx_search calls.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": { "type": "string" },
                                    "command": { "type": "string" }
                                },
                                "required": ["label", "command"]
                            },
                            "description": "Commands to execute"
                        },
                        "queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Search queries to run against indexed output"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Total timeout in milliseconds"
                        }
                    },
                    "required": ["commands", "queries"]
                }
            },
            {
                "name": "bpx_search",
                "description": "Search indexed content using multi-layer search (BM25, trigram, fuzzy, semantic vector similarity).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Search queries"
                        },
                        "source": {
                            "type": "string",
                            "description": "Filter by source label"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["code", "prose"],
                            "description": "Filter by content type"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results per query (default: 10)"
                        }
                    },
                    "required": ["queries"]
                }
            },
            {
                "name": "bpx_execute_file",
                "description": "Read a file, optionally process it, and index the results.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read and index"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language for syntax-aware processing"
                        },
                        "code": {
                            "type": "string",
                            "description": "Optional processing script to pipe the file through"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "bpx_fetch_and_index",
                "description": "Fetch a URL, convert HTML to markdown, and index it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        },
                        "label": {
                            "type": "string",
                            "description": "Label for indexing (defaults to URL)"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "bpx_index",
                "description": "Index raw text for later search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Text content to index"
                        },
                        "label": {
                            "type": "string",
                            "description": "Label for the indexed content"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["code", "prose"],
                            "description": "Content type hint"
                        }
                    },
                    "required": ["content", "label"]
                }
            },
            {
                "name": "bpx_promote",
                "description": "Promote indexed content to an obsidian output note via taskvault.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find content to promote"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name for the output note"
                        },
                        "project": {
                            "type": "string",
                            "description": "Obsidian project to associate with"
                        }
                    },
                    "required": ["query", "name", "project"]
                }
            },
            {
                "name": "bpx_stats",
                "description": "Show context savings metrics for this session.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    })
}
