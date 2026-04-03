use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use tokenizers::Tokenizer;

use super::pool;

/// The embedding dimension for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length for all-MiniLM-L6-v2.
const MAX_SEQ_LEN: usize = 256;

/// Batch size for embedding multiple texts at once.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Loads and runs the all-MiniLM-L6-v2 BERT model via Candle.
pub struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEmbedder {
    /// Load model weights and tokenizer from the given directory.
    ///
    /// Expects `model.safetensors`, `config.json`, and `tokenizer.json` in `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let device = Self::select_device();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config: BertConfig =
            serde_json::from_str(&config_str).context("failed to parse model config.json")?;

        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("failed to load model.safetensors")?
        };

        let model =
            BertModel::load(vb, &config).context("failed to build BertModel from weights")?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Select the best available device (CUDA if available, else CPU).
    fn select_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    }

    /// Embed a single text, returning a 384-dim L2-normalized vector.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed multiple texts in batches, returning 384-dim L2-normalized vectors.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(DEFAULT_BATCH_SIZE) {
            let batch = self.embed_batch_inner(chunk)?;
            all_embeddings.extend(batch);
        }

        Ok(all_embeddings)
    }

    /// Internal: embed a single batch (up to DEFAULT_BATCH_SIZE texts).
    fn embed_batch_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                let mut encoding = self
                    .tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
                encoding.truncate(MAX_SEQ_LEN, 0, tokenizers::TruncationDirection::Right);
                Ok(encoding)
            })
            .collect::<Result<Vec<_>>>()?;

        // Pad to longest sequence in batch
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids_vec = Vec::with_capacity(texts.len() * max_len);
        let mut attention_mask_vec = Vec::with_capacity(texts.len() * max_len);
        let mut token_type_ids_vec = Vec::with_capacity(texts.len() * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let pad_len = max_len - ids.len();

            input_ids_vec.extend_from_slice(ids);
            input_ids_vec.extend(std::iter::repeat_n(0u32, pad_len));

            attention_mask_vec.extend(mask.iter().map(|&m| m as f32));
            attention_mask_vec.extend(std::iter::repeat_n(0.0f32, pad_len));

            token_type_ids_vec.extend_from_slice(type_ids);
            token_type_ids_vec.extend(std::iter::repeat_n(0u32, pad_len));
        }

        let batch_size = texts.len();
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)?;

        // Forward pass
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pool + L2 normalize
        let pooled = pool::mean_pool(&output, &attention_mask)?;
        let normalized = pool::l2_normalize(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let normalized = normalized.to_device(&Device::Cpu)?;
        let result: Vec<Vec<f32>> = normalized.to_vec2()?;
        Ok(result)
    }
}
