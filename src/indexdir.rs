use anyhow::{Context, Result};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::store::ContentStore;

pub struct IndexDirResult {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub total_chunks: usize,
    pub total_bytes: u64,
    pub walk_errors: Vec<String>,
}

/// Index all files in a directory into the content store.
///
/// Walks the directory respecting .gitignore, skips binary files,
/// and optionally filters by glob pattern.
pub fn index_directory(
    store: &ContentStore,
    path: &Path,
    glob: Option<&str>,
    label_prefix: &str,
) -> Result<IndexDirResult> {
    let mut builder = WalkBuilder::new(path);
    builder.hidden(true).git_ignore(true).git_global(true);

    if let Some(pattern) = glob {
        let mut overrides = OverrideBuilder::new(path);
        overrides
            .add(pattern)
            .context("invalid glob pattern")?;
        builder.overrides(overrides.build()?);
    }

    let mut result = IndexDirResult {
        files_indexed: 0,
        files_skipped: 0,
        total_chunks: 0,
        total_bytes: 0,
        walk_errors: Vec::new(),
    };

    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                result.walk_errors.push(e.to_string());
                result.files_skipped += 1;
                continue;
            }
        };

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let file_path = entry.path();

        // Binary detection: check first 1KB for null bytes
        match is_binary(file_path) {
            Ok(true) => {
                result.files_skipped += 1;
                continue;
            }
            Err(_) => {
                result.files_skipped += 1;
                continue;
            }
            Ok(false) => {}
        }

        // Read as UTF-8, skip if invalid
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => {
                result.files_skipped += 1;
                continue;
            }
        };

        if content.is_empty() {
            result.files_skipped += 1;
            continue;
        }

        let relative = file_path
            .strip_prefix(path)
            .unwrap_or(file_path);
        let label = format!("{}{}", label_prefix, relative.display());

        let index_result = store
            .index(&label, &content, None)
            .with_context(|| format!("failed to index {}", file_path.display()))?;

        result.files_indexed += 1;
        result.total_chunks += index_result.chunk_count as usize;
        result.total_bytes += content.len() as u64;
    }

    Ok(result)
}

/// Check if a file is binary by looking for null bytes in the first 1KB.
fn is_binary(path: &Path) -> Result<bool> {
    let mut file = fs::File::open(path)?;
    let mut buf = [0u8; 1024];
    let bytes_read = file.read(&mut buf)?;
    Ok(buf[..bytes_read].contains(&0u8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_store() -> (tempfile::TempDir, ContentStore) {
        let dir = tempdir().unwrap();
        let store = ContentStore::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn test_binary_detection() {
        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("binary.dat");
        fs::write(&bin_path, b"hello\x00world").unwrap();
        assert!(is_binary(&bin_path).unwrap());

        let txt_path = dir.path().join("text.txt");
        fs::write(&txt_path, "hello world").unwrap();
        assert!(!is_binary(&txt_path).unwrap());
    }

    #[test]
    fn test_index_directory_basic() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}").unwrap();
        fs::write(src.join("lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(dir.path().join("README.md"), "# Hello").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), None, "").unwrap();
        assert_eq!(result.files_indexed, 3);
        assert!(result.total_chunks >= 3);
    }

    #[test]
    fn test_glob_filter() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("notes.txt"), "some notes").unwrap();
        fs::write(dir.path().join("data.json"), "{}").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), Some("*.rs"), "").unwrap();
        assert_eq!(result.files_indexed, 1);
    }

    #[test]
    fn test_glob_filter_nested() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(src.join("lib.rs"), "pub fn f() {}").unwrap();
        fs::write(dir.path().join("notes.txt"), "some notes").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), Some("*.rs"), "").unwrap();
        // *.rs should match at all depths
        assert_eq!(result.files_indexed, 2, "glob *.rs should match nested src/lib.rs too");
    }

    #[test]
    fn test_label_prefix() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "content").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), None, "myproject/").unwrap();
        assert_eq!(result.files_indexed, 1);

        // Verify the label by searching
        let search_results = store.search("content", 10, Some("myproject/file.txt"), None).unwrap();
        assert!(!search_results.is_empty());
    }

    #[test]
    fn test_skips_binary_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("text.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("image.png"), b"PNG\x00\x00\x00").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), None, "").unwrap();
        assert_eq!(result.files_indexed, 1);
        assert_eq!(result.files_skipped, 1);
    }

    #[test]
    fn test_gitignore_respected() {
        let dir = tempdir().unwrap();
        // ignore crate requires a .git dir to recognize .gitignore
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::write(dir.path().join(".gitignore"), "*.log\ntarget/\n").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("debug.log"), "log output").unwrap();
        let target = dir.path().join("target");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("build.rs"), "build artifact").unwrap();

        let (_store_dir, store) = make_store();
        let result = index_directory(&store, dir.path(), None, "").unwrap();
        // Only main.rs should be indexed (debug.log and target/ ignored by gitignore,
        // .gitignore itself skipped as hidden file)
        assert_eq!(result.files_indexed, 1);
    }
}
