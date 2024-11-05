# Performance_and_scalability

### Optimizing inference

Use a GPU, and optimize inference with:

- Flash-attention :

FlashAttention-2 is a faster and more efficient implementation of the standard attention mechanism that can significantly speedup inference and transform its complexity from quadratic to linear.

On Nvidia, we have to install flash attention:
```pip install flash-attn --no-build-isolation```

Then to use the flash attention algo, we have to pass the argument ```attn_implementation="flash_attention_2"``` to from_pretrained():
```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

FlashAttention-2 can only be used when the model’s dtype is fp16 or bf16.

FlashAttention-2 can be combined with other optimization techniques like quantization (with 8-bit or 4-bit) to further speedup inference.

FlashAttention-2 does not support computing attention scores with padding tokens. To overcome this, you should use FlashAttention-2 without padding tokens in the sequence during training (by packing a dataset or concatenating sequences until reaching the maximum sequence length).

The larger sequence lengths you have , the more speedups you get.

For example, with a relatively small sequence length (prompt_max_kength=256),  and a padding ratio of 0.3 (30% of the input is filled with padding tokens) a single forward pass creates overhead leading to a small speedup
![Screenshot](smal_seq_length_with_padd.PNG)

But for larger sequence lengths (prompt_max_kength=2048 and keeping a padding ratio of 0.3 ), you can expect even more speedup benefits. Besides, FlashAttention is more memory efficient, meaning you can train on much larger sequence lengths without running into out-of-memory issues.
![Screenshot](larg_seq_length.PNG)

And if Flash attention is well  used with large sequences and without padding, we can have more speedup benefits (up to *2). check-out the speeedups here on the same model for larger sequence lengths (prompt_max_kength=4096) and  a padding ratio of 0.
![Screenshot](ideal_situation.PNG)



- BetterTransformer :
- Quantization with bitsandbytes :
