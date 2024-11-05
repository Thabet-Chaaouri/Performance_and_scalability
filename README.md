# Performance_and_scalability

### Optimizing inference

Use a GPU, and optimize inference with:

- Flash-attention :

FlashAttention-2 is a faster and more efficient implementation of the standard attention mechanism that can significantly speedup inference and transform its complexity from quadratic to linear.
On Nvidia, we have to install flash attention:

```pip install flash-attn --no-build-isolation```

- BetterTransformer :
- Quantization with bitsandbytes :
