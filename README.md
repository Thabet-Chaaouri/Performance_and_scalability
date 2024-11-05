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



- PyTorch scaled dot product attention :

 SDPA support is currently being added natively in Transformers and is used by default for torch>=2.1.1 when an implementation is available. You may also set ```attn_implementation="sdpa"``` in from_pretrained() to explicitly request SDPA to be used

 
- BetterTransformer :

Some BetterTransformer features are being upstreamed to Transformers with default support for native SDPA; BetterTransformer still has a wider coverage than the Transformers SDPA integration, but you can expect more and more architectures to natively support SDPA in Transformers.

BetterTransformer accelerates inference with its fastpath (native PyTorch specialized implementation of Transformer functions) execution.

BetterTransformer two optimizations in the fastpath execution (fusion and skipping the inherent sparsity of padding tokens). It also converts all attention operations to use the more memory-efficient scaled dot product attention (SDPA), and it calls optimized kernels like FlashAttention under the hood.

Make sure you have 🤗 Optimum installed. Then you can enable BetterTransformer with :
```model = model.to_bettertransformer()```


- Quantization with bitsandbytes :

Quantization reduces your model size compared to its native full precision version, making it easier to fit large models onto GPUs with limited memory.  It supports 4-bit and 8-bit quantization.

If you’re loading a model in 8-bit for text generation, you should use the generate() method instead of the Pipeline function which is not optimized for 8-bit models and will be slower. You should also place all inputs on the same device as the model.

- Combine optimizations :

It is often possible to combine several of the optimization techniques.  For example, you can load a model in 4-bit, and then enable BetterTransformer with FlashAttention:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

# enable BetterTransformer
model = model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# enable FlashAttention
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

