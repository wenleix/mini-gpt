
import mlx.core as mx
import mlx.nn as nn
import math

nn.MultiHeadAttention
class MultiheadAttention(nn.Module):
    """
    References:
    * https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L29-L34
    * https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Multi-Head-Attention
    * https://github.com/ml-explore/mlx-examples/blob/1e05aef344907d2697f82b3a5b5c00cdf21c298c/llms/llama/llama.py#L31-L49
    * https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html#attention-layer
    """
    def __init__(self, embed_dims: int, num_heads: int):
        super.__init__()

        # embed_dims = num_heads * head_dims
        assert embed_dims % num_heads == 0, 'embed_dims must be divisible by num_heads'

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        # Seperate the projection for Q, K, V for readability
        # In many implementations, these are combined into a single projection as nn.Linear(embed_dims, 3*embed_dims)
        self.query_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.key_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.value_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=False)

    def forward(self, x: mx.array, mask: mx.array | None = None):
        # queries == keys == values == x
        batch_size, seq_len, embed_dims = x.shape
        assert embed_dims == self.embed_dims, 'Unexpected input embedding dimensions'

        queries, keys, values = self.query_proj(x), self.key_proj(x), self.value_proj(x)
        # [Batch, Head, SeqLen, HeadDims]
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)

        # self-attention:
        #   [Batch, Head, SeqLen, HeadDims] X [Batch, Head, HeadDims, SeqLen] -> [Batch, Head, SeqLen, SeqLen]
        scores = queries @ keys.transpose(0, 1, 3, 2) * math.sqrt(1.0 / self.head_dims)
        if mask is not None:
            scores = scores + mx.logical_not(mask) * 1e-9
        scores = mx.softmax(scores, axis=-1)

        # [Batch, Head, SeqLen, SeqLen] X [Batch, Head, SeqLen, HeadDims] -> [Batch, Head, SeqLen, HeadDims
        values = scores @ values
        # re-assemble all head outputs side by side
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dims)

        # output projection
        return self.out_proj(values)




