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
                "description": "Primary retrieval tool after indexing. Search indexed content using multi-layer search and use the returned source_id for exact follow-up reads via bpx_read_chunks. REVIEW RULE: bpx_search is for discovery, not for repo-wide claims. Before stating 'no endpoint checks X' or 'all routes do Y', you MUST confirm with exact reads (Read tool or bpx_read_chunks) on the specific files. Search results show where matches exist — absence from results does NOT prove absence from the codebase.",
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
                            "description": "Max results per query (default: 15)"
                        },
                        "snippet_bytes": {
                            "type": "integer",
                            "description": "Max bytes per snippet for top results (default: 2000)"
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
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query. For large files, returns relevant chunks instead of head-only preview."
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "bpx_read_chunks",
                "description": "Retrieve full indexed content by source label. Returns chunks in original order and does not truncate them.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Exact source label to read back from the index"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional query to filter to only matching chunks within that source"
                        },
                        "max_chunks": {
                            "type": "integer",
                            "description": "Maximum chunks to return (default: 10)"
                        }
                    },
                    "required": ["label"]
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
                "name": "bpx_index_dir",
                "description": "Index all source files in a directory. Walks the directory respecting .gitignore, skips binary files, and indexes everything into the content store for later bpx_search. Run this at the start of exploration tasks to seed the search index.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to index (e.g. \"./src\", \"/absolute/path/to/project\")"
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern to filter files (e.g. \"*.rs\", \"*.py\"). Matches at any depth. If omitted, all non-binary text files are indexed."
                        },
                        "label_prefix": {
                            "type": "string",
                            "description": "Prefix for all labels (e.g. \"myproject/\"). Defaults to empty string."
                        }
                    },
                    "required": ["path"]
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
            },
            {
                "name": "bpx_sources",
                "description": "List all indexed sources with chunk counts and timestamps. Use this to discover labels for bpx_read_chunks or bpx_search --source.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "bpx_context_status",
                "description": "Show context budget usage, per-source token breakdown, and relevance scores.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    })
}
