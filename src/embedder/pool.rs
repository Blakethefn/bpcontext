use anyhow::Result;
use candle_core::Tensor;

/// Mean pooling over token embeddings, respecting attention mask.
///
/// `embeddings` shape: (batch, seq_len, hidden_dim)
/// `attention_mask` shape: (batch, seq_len)
///
/// Returns: (batch, hidden_dim) tensor of mean-pooled vectors.
pub fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Expand mask to (batch, seq_len, 1) for broadcasting
    let mask = attention_mask.unsqueeze(2)?.to_dtype(embeddings.dtype())?;

    // Zero out padding positions, then sum across seq_len
    let masked = embeddings.broadcast_mul(&mask)?;
    let summed = masked.sum(1)?; // (batch, hidden_dim)

    // Count non-padding tokens per sequence
    let counts = mask.sum(1)?; // (batch, 1)
    let counts = counts.clamp(1.0, f64::MAX)?; // Avoid division by zero

    let pooled = summed.broadcast_div(&counts)?;
    Ok(pooled)
}

/// L2-normalize each vector so cosine similarity = dot product.
///
/// `vectors` shape: (batch, dim)
/// Returns: (batch, dim) with each row having unit norm.
pub fn l2_normalize(vectors: &Tensor) -> Result<Tensor> {
    let norms = vectors
        .sqr()?
        .sum_keepdim(1)? // (batch, 1)
        .sqrt()?
        .clamp(1e-12, f64::MAX)?; // Avoid division by zero

    let normalized = vectors.broadcast_div(&norms)?;
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn mean_pool_simple() {
        let device = Device::Cpu;
        // (1, 3, 2) — 1 batch, 3 tokens, 2 dims
        let embeddings = Tensor::new(
            &[[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            &device,
        )
        .unwrap();
        // All tokens valid
        let mask = Tensor::new(&[[1.0f32, 1.0, 1.0]], &device).unwrap();

        let result = mean_pool(&embeddings, &mask).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn mean_pool_with_padding() {
        let device = Device::Cpu;
        // (1, 3, 2) — last token is padding
        let embeddings = Tensor::new(
            &[[[1.0f32, 2.0], [3.0, 4.0], [999.0, 999.0]]],
            &device,
        )
        .unwrap();
        let mask = Tensor::new(&[[1.0f32, 1.0, 0.0]], &device).unwrap();

        let result = mean_pool(&embeddings, &mask).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Mean of [1,2] and [3,4] = [2.0, 3.0]
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_produces_unit_vectors() {
        let device = Device::Cpu;
        let vectors = Tensor::new(&[[3.0f32, 4.0], [0.0, 5.0]], &device).unwrap();

        let normed = l2_normalize(&vectors).unwrap();
        let data: Vec<Vec<f32>> = normed.to_vec2().unwrap();

        // [3,4] -> [0.6, 0.8]
        assert!((data[0][0] - 0.6).abs() < 1e-5);
        assert!((data[0][1] - 0.8).abs() < 1e-5);

        // [0,5] -> [0.0, 1.0]
        assert!(data[1][0].abs() < 1e-5);
        assert!((data[1][1] - 1.0).abs() < 1e-5);
    }
}
