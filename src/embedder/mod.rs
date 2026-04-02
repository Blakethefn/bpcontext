pub mod model;
pub mod pool;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use model::CandleEmbedder;

/// The embedding dimension for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = model::EMBEDDING_DIM;

/// Trait for embedding text into vectors. Enables mocking in tests.
#[allow(dead_code)]
pub trait Embed: Send + Sync {
    /// Embed a single text, returning an L2-normalized vector.
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in batches, returning L2-normalized vectors.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// The dimensionality of the output vectors.
    fn dim(&self) -> usize;
}

/// Production embedder backed by Candle + all-MiniLM-L6-v2.
///
/// Lazily loads the model on first use. Downloads via hf-hub if not cached.
pub struct Embedder {
    inner: CandleEmbedder,
}

impl Embedder {
    /// Create a new Embedder, loading the model from the given directory.
    ///
    /// If the model files are not present, downloads them via hf-hub first.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_dir = ensure_model_downloaded(model_dir)?;
        let inner = CandleEmbedder::load(&model_dir)
            .context("failed to load embedding model")?;
        Ok(Self { inner })
    }

    /// Default model directory: `~/.local/share/bpcontext/models/all-MiniLM-L6-v2`
    #[allow(dead_code)]
    pub fn default_model_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .context("could not determine local data directory")?;
        Ok(data_dir.join("bpcontext").join("models").join("all-MiniLM-L6-v2"))
    }
}

impl Embed for Embedder {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed_one(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch(texts)
    }

    fn dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Ensure model files exist in `model_dir`, downloading via hf-hub if needed.
///
/// Returns the path where the model files are located.
fn ensure_model_downloaded(model_dir: &Path) -> Result<PathBuf> {
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if config_path.exists() && weights_path.exists() && tokenizer_path.exists() {
        return Ok(model_dir.to_path_buf());
    }

    eprintln!("[bpcontext] Downloading all-MiniLM-L6-v2 model (~80MB, one-time)...");

    let api = hf_hub::api::sync::Api::new()
        .context("failed to initialize hf-hub API")?;
    let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());

    // Download each required file
    let files = ["config.json", "model.safetensors", "tokenizer.json"];
    for filename in &files {
        let downloaded_path = repo
            .get(filename)
            .with_context(|| format!("failed to download {filename} from HuggingFace"))?;

        let target = model_dir.join(filename);
        if downloaded_path != target {
            // hf-hub caches files in its own location — copy to our model_dir
            std::fs::create_dir_all(model_dir)
                .with_context(|| format!("failed to create {}", model_dir.display()))?;
            std::fs::copy(&downloaded_path, &target)
                .with_context(|| format!("failed to copy {filename} to model dir"))?;
        }
    }

    eprintln!("[bpcontext] Model downloaded successfully.");
    Ok(model_dir.to_path_buf())
}

/// Convert a f32 embedding vector to raw bytes for SQLite BLOB storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert raw bytes from SQLite BLOB back to a f32 vector.
#[allow(dead_code)]
pub fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Mock embedder for unit tests — returns deterministic vectors.
    pub(crate) struct MockEmbedder {
        pub dim: usize,
    }

    impl MockEmbedder {
        pub fn new() -> Self {
            Self { dim: EMBEDDING_DIM }
        }
    }

    impl Embed for MockEmbedder {
        fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
            Ok(mock_embedding(text, self.dim))
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| mock_embedding(t, self.dim)).collect())
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    /// Generate a deterministic mock embedding from text.
    /// Uses a simple hash-based approach so similar inputs get different vectors.
    fn mock_embedding(text: &str, dim: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dim];
        for (i, byte) in text.bytes().enumerate() {
            vec[i % dim] += byte as f32;
        }
        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for val in &mut vec {
            *val /= norm;
        }
        vec
    }

    #[test]
    fn embedding_bytes_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.125, 0.0, -0.001];
        let bytes = embedding_to_bytes(&original);
        let recovered = bytes_to_embedding(&bytes);

        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn mock_embedder_returns_correct_dim() {
        let embedder = MockEmbedder::new();
        let vec = embedder.embed_one("hello world").unwrap();
        assert_eq!(vec.len(), EMBEDDING_DIM);
    }

    #[test]
    fn mock_embedder_returns_normalized_vectors() {
        let embedder = MockEmbedder::new();
        let vec = embedder.embed_one("test input").unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mock_embedder_batch_matches_individual() {
        let embedder = MockEmbedder::new();
        let texts = &["hello", "world"];
        let batch = embedder.embed_batch(texts).unwrap();
        let individual: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| embedder.embed_one(t).unwrap())
            .collect();

        assert_eq!(batch, individual);
    }
}
