import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from xminigrid.core.constants import NUM_TILES, NUM_COLORS


def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


# class AliBiCausalSelfAttention(nn.Module):
#     def __init__(self, hidden_dim, num_heads, dropout=0.0, normalize_qk=False):
#         super().__init__()
#         self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.register_buffer(
#             "alibi_slopes", torch.as_tensor(get_alibi_slopes(num_heads)), persistent=False
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.num_heads = num_heads
#         self.normalize_qk = normalize_qk
#         self.scale = (hidden_dim // num_heads) ** -0.5

#         if normalize_qk:
#             self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
#             self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

    # def forward(self, x, **kwargs):
    #     B, L, D = x.shape
    #     qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)
    #     q, k, v = qkv.unbind(2)
        
    #     if self.normalize_qk:
    #         q, k = self.q_norm(q), self.k_norm(k)
        
    #     q = q * self.scale
    #     attn_weights = (q @ k.transpose(-2, -1))
    #     attn_weights += self.alibi_slopes.view(1, 1, self.num_heads, 1).to(x.device)
    #     attn_weights = attn_weights.masked_fill(
    #         torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1), float('-inf')
    #     )
    #     attn_weights = F.softmax(attn_weights, dim=-1)
    #     attn_weights = self.dropout(attn_weights)
    #     out = (attn_weights @ v).reshape(B, L, D)
    #     return self.out_proj(out)

# class AliBiCausalSelfAttention(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_heads: int,
#         attention_dropout: float,
#         normalize_qk: bool = False,
#         with_alibi: bool = True,
#     ):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
#         self.normalize_qk = normalize_qk
#         self.with_alibi = with_alibi
        
#         # When not normalizing, scale queries by the inverse square root of head_dim.
#         self.scale = self.head_dim ** -0.5
        
#         # Combined QKV projection; output shape will be [batch, seq_len, 3 * hidden_dim]
#         self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_dropout = nn.Dropout(attention_dropout)
        
#         # Precompute ALiBi slopes for each head if requested.
#         if self.with_alibi:
#             self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads), persistent=False)

#     def forward(self, x: torch.Tensor, k_cache: torch.Tensor = None,
#                 v_cache: torch.Tensor = None, cache_seqlens=None) -> torch.Tensor:
#         """
#         Args:
#             x: Input tensor of shape [batch_size, seq_len, hidden_dim].
#             k_cache, v_cache: Optional tensors for cached keys and values
#                               (shape [batch, num_heads, cache_length, head_dim]).
#             cache_seqlens: (Optional) tensor indicating cache sequence lengths.
        
#         Returns:
#             Tensor of shape [batch_size, seq_len, hidden_dim] representing the output.
#         """
#         batch_size, seq_len, _ = x.size()
        
#         # Compute queries, keys, and values from input.
#         qkv = self.qkv_proj(x)  # shape: [batch, seq_len, 3 * hidden_dim]
#         q, k, v = torch.chunk(qkv, 3, dim=-1)  # each is [batch, seq_len, hidden_dim]
        
#         # Reshape and transpose to get separate heads.
#         # New shape: [batch, num_heads, seq_len, head_dim]
#         q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Optionally normalize q and k.
#         if self.normalize_qk:
#             q = F.normalize(q, dim=-1)
#             k = F.normalize(k, dim=-1)
#             scale = 1.0  # When vectors are normalized, scaling isn’t required.
#         else:
#             scale = self.scale

#         # Update caches if provided (for autoregressive decoding).
#         if k_cache is not None and v_cache is not None:
#             # Here we simply concatenate along the sequence length.
#             k = torch.cat([k_cache, k], dim=2)
#             v = torch.cat([v_cache, v], dim=2)
        
#         # Compute scaled dot-product attention scores.
#         # Shape: [batch, num_heads, query_len, key_len]
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
#         # Optionally add ALiBi relative position biases.
#         if self.with_alibi:
#             q_len = q.size(2)
#             k_len = k.size(2)
#             # Compute relative positions [q_len, k_len]: (j - i)
#             rel_positions = torch.arange(k_len, device=x.device).unsqueeze(0) - \
#                             torch.arange(q_len, device=x.device).unsqueeze(1)
#             # Expand dims to [1, 1, q_len, k_len] for broadcasting.
#             rel_positions = rel_positions.unsqueeze(0).unsqueeze(0).float()
#             # Expand slopes from [num_heads, 1, 1] to [1, num_heads, 1, 1] and multiply.
#             alibi_bias = self.alibi_slopes.unsqueeze(0) * rel_positions
#             attn_scores = attn_scores + alibi_bias

#         # Apply a causal mask so that position i cannot attend to j > i.
#         # Create a lower-triangular mask of shape [query_len, key_len].
#         causal_mask = torch.tril(torch.ones(seq_len, k.size(2), device=x.device)).unsqueeze(0).unsqueeze(0)
#         attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        
#         # Compute attention weights.
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attn_weights = self.attn_dropout(attn_weights)
        
#         # Compute the weighted sum of the values.
#         attn_output = torch.matmul(attn_weights, v)  # shape: [batch, num_heads, seq_len, head_dim]
        
#         # Recombine the heads: reshape back to [batch, seq_len, hidden_dim].
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
#         # Final output projection.
#         output = self.out_proj(attn_output)
#         return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        normalize_qk: bool = False,
        pre_norm: bool = True,
        with_alibi: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = FlashAliBiCausalSelfAttention(
            hidden_dim, num_heads, attention_dropout, normalize_qk=normalize_qk
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.pre_norm = pre_norm

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x, k_cache=None, v_cache=None, cache_seqlens=None):
        if self.pre_norm:
            attention_out = self.attention(
                self.norm1(x),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
            )
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out = self.attention(
                x, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens
            )
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        return x


# WARN: these modules are just an examples of attention implementation from scratch
# they are only for educational purposes here!
def get_alibi_relative_positions(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return (x - y).to(torch.float)


class EmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 2) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.entity_emb = nn.Embedding(NUM_TILES + 1, embedding_dim)
        self.color_emb = nn.Embedding(NUM_COLORS, embedding_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_emb = torch.cat(
            [self.entity_emb(img[..., 0]), self.color_emb(img[..., 1])], dim=-1
        )
        img_emb.swapaxes_(2, 3).swapaxes_(1, 2)
        # we want to have [bs * seq_len, emb_size * 2, 5, 5]

        return img_emb


class ObservationEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 16, features_dim: int = 64) -> None:
        super().__init__()

        self.embeding_dim = embedding_dim
        self.features_dim = features_dim
        self.transform = EmbeddingEncoder(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(2 * embedding_dim, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, features_dim),
        )

    def forward(self, img: torch.Tensor, cast_to=torch.float32) -> torch.Tensor:
        # img: shape [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
        batch_size, seq_len = img.shape[0], img.shape[1]

        if img.shape != (batch_size, seq_len, 5, 5, 2):
            img.swapaxes_(2, 3).swapaxes_(3, 4)

        assert img.shape == (batch_size, seq_len, 5, 5, 2)

        img_transformed = self.transform(img.flatten(0, 1))
        out = self.encoder(img_transformed.to(cast_to))
        return out.reshape(batch_size, seq_len, -1)