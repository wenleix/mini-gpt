
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
import numpy as np

nn.TransformerEncoder

class MultiheadAttention(nn.Module):
    """
    References:
    * https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L29-L34
    * https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Multi-Head-Attention
    * https://github.com/ml-explore/mlx-examples/blob/1e05aef344907d2697f82b3a5b5c00cdf21c298c/llms/llama/llama.py#L31-L49
    * https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html#attention-layer
    """
    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()

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

    def __call__(self, x: mx.array, mask: mx.array | None = None):
        # queries = keys = values = x
        batch_size, seq_len, embed_dims = x.shape
        assert embed_dims == self.embed_dims, 'Unexpected input embedding dimensions'

        queries, keys, values = self.query_proj(x), self.key_proj(x), self.value_proj(x)
        # Reshape & Transpose -> [Batch, Head, SeqLen, HeadDims]
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dims).transpose(0, 2, 1, 3)

        # self-attention:
        #   [Batch, Head, SeqLen, HeadDims] X [Batch, Head, HeadDims, SeqLen] -> [Batch, Head, SeqLen, SeqLen]
        scores = queries @ keys.transpose(0, 1, 3, 2) * math.sqrt(1.0 / self.head_dims)
        if mask is not None:
            # hmm, this follows MLX pattern rather than the original mask_fill pattern
            scores += mask
        scores = mx.softmax(scores, axis=-1)

        # [Batch, Head, SeqLen, SeqLen] X [Batch, Head, SeqLen, HeadDims] -> [Batch, Head, SeqLen, HeadDims
        values = scores @ values
        # re-assemble all head outputs side by side
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dims)

        # output projection
        return self.out_proj(values)


class EncoderBlock(nn.Module):
    """
    References:
    * https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L73-L93
    * https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Transformer-Encoder
    * https://github.com/ml-explore/mlx-examples/blob/1e05aef344907d2697f82b3a5b5c00cdf21c298c/llms/llama/llama.py#L102-L111
    """
    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()

        # Attention layer
        self.attention = MultiheadAttention(embed_dims, num_heads)

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, 4 * embed_dims),
            nn.GELU(),
            nn.Linear(4 * embed_dims, embed_dims)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def __call__(self, x, mask: mx.array | None = None):
        # Attention part
        x = x + self.attention(self.norm1(x), mask)

        # MLP part
        x = x + self.mlp(self.norm2(x))

        return x


class GPT(nn.Module):
    """
    References:
    * https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L95-L96
    * https://github.com/ml-explore/mlx-examples/blob/1e05aef344907d2697f82b3a5b5c00cdf21c298c/llms/llama/llama.py#L126-L134
    * https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html#full-model
    """
    def __init__(self, num_layers: int, vocab_size: int, embed_dims: int, num_heads: int, block_size: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dims)
        self.block_size = block_size
        self.positional_embedding = nn.Embedding(block_size, embed_dims)
        self.encoder_blocks = [
            EncoderBlock(embed_dims, num_heads)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(embed_dims)
        self.out_proj = nn.Linear(embed_dims, vocab_size, bias=False)

    def __call__(self, x: mx.array):
        batch_size, seq_len = x.shape
        assert seq_len <= self.block_size, 'Sequence length exceeds block size'

        positions = mx.arange(seq_len).reshape(1, seq_len)
        # [Batch, SeqLen, EmbedDims] + [1, SeqLen, EmbedDims]
        x = self.token_embedding(x) + self.positional_embedding(positions)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=x.dtype)
        for block in self.encoder_blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.out_proj(x)


def loss_fn(model, X: mx.array, Y: mx.array):
    logits = model(X)
    losses = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1), reduction='mean')
    return losses


def batch_iterate(batch_size, X, Y):
    assert X.shape == Y.shape
    n = X.shape[0]
    perm = mx.array(np.random.permutation(n))
    for i in range(0, n, batch_size):
        ids = perm[i:i + batch_size]
        yield X[ids], Y[ids]


BLOCK_SIZE = 64
BATCH_SIZE = 16

def main():
    model = GPT(num_layers=4, vocab_size=256, embed_dims=128, num_heads=4, block_size=BLOCK_SIZE)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=0.001, betas=[0.9, 0.99])
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    with open('everglow.txt', 'r') as f:
        text = f.read()
    data = mx.array([ord(c) for c in text])
    print(data.shape)
    X = []
    Y = []
    for i in range(0, len(data) - BLOCK_SIZE):
        chunk = data[i:i+BLOCK_SIZE+1]
        X.append(chunk[:-1])
        Y.append(chunk[1:])
    X = mx.array(X)
    Y = mx.array(Y)
    print(X.shape, Y.shape)

    for epoch in range(5):
        for X_batch, Y_batch in batch_iterate(BATCH_SIZE, X, Y):
            loss, grads = loss_and_grad_fn(model, X_batch, Y_batch)
            optimizer.update(model, grads)
            mx.eval(model.state)
            print(f'loss: {loss.item()}')

if __name__ == '__main__':
    main()
