
"""Dense decoder-only Transformer used for the addition experiment."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

# ---------- Model configuration ----------

@dataclass(frozen=True)
class ModelConfig:
  vocab_size: int
  context_length: int

  d_model: int = 384
  num_layers: int = 6
  num_heads: int = 6
  rope_theta: float = 10_000.0
  d_ff: int = 1_024 # An empirical choice (for SwiGLU) is to take d_ff = (8/3)* d_model

  @property
  def head_dim(self) -> int:
    return self.d_model // self.num_heads


# ---------- Parameter initialization ----------

def truncated_xavier_init(key, shape, dtype = jnp.float32):
  fan_in , fan_out = shape

  std = jnp.asarray(math.sqrt(2.0/(fan_in+fan_out)),
                    dtype = dtype)

  samples = jax.random.truncated_normal(
      key,
      lower=-3.0,
      upper= 3.0,
      shape = shape,
      dtype = dtype
  )

  return samples*std

token_embedding_init = nn.initializers.truncated_normal(
    stddev = 1.0,
    lower = -3.0,
    upper = 3.0,
    dtype = jnp.float32
)


# ---------- RMSNorm ----------

class RMSNorm(nn.Module):
  d_model : int
  eps : float = 1e-5

  @nn.compact
  def __call__(self, x):
    # x has the shape of [B,T,D]
    mean_squared = jnp.mean(
        jnp.square(x),
        axis = -1,
        keepdims = True
    ) # this gives [B,T,1] and not [B,T]

    normalized = x * jax.lax.rsqrt(
        mean_squared+self.eps
    )

    scale = self.param(
        "scale",
        nn.initializers.ones,
        (self.d_model,)
    )

    # scale: [D], normalized: [B,T,D]. So we broadcast scale to [B,T,D] and then do the component wise multiplication
    return scale * normalized # [B,T,D]


# ---------- Rotary position embeddings ----------

def apply_rope(x ,positions, theta):
  # input x : [B , T, H, D]
  head_dim = x.shape[-1]

  # Convert positions to float32
  positions = positions.astype(jnp.float32) # [B,T] or [T]

  pair_indices = jnp.arange(
      head_dim//2,
      dtype = jnp.float32
  ) #[D/2]

  frequencies = theta**(-2*pair_indices/head_dim) # [D/2]

  angles = positions[..., None] * frequencies # [... B,T,D/2]
  # positions[..., None] # [... B, T, 1], broadcasted to [... B,T,D/2]
  # frequencies # [D/2], broadcasted to [... 1,1,D/2]

  cos = jnp.cos(angles)[...,None,:]   # [... B,T,1,D/2]
  sin = jnp.sin(angles)[...,None,:]   # [... B,T,1,D/2]

  x_even = x[...,0::2] # [... B,T,H,D/2]
  x_odd = x[...,1::2] # [... B,T,H,D/2]

  rotated_even = x_even * cos - x_odd * sin # [... B,T,H,D/2]
  rotated_odd = x_even * sin + x_odd * cos # [... B,T,H,D/2]

  rotated_pairs = jnp.stack(
      (rotated_even, rotated_odd),
      axis = -1
  ) # [... B, T, H, D/2, 2]

  return rotated_pairs.reshape(x.shape) # [... B, T, H, D]


# ---------- Causal self-attention calculation ----------

def causal_self_attention(q, k, v, query_start_index=0):
  """
  Explaining `query_start_index`
  For ordinary full-sequence attention, the first query corresponds to position zero.
  During cached generation, it tells the function where the current query begins inside
  the cache. For example, after an eight-character prompt, the next input token begins
  at cache position eight.
  """
  query_length = q.shape[-3]
  key_length = k.shape[-3]
  head_dim = q.shape[-1]

  scores = jnp.einsum("...thd,...shd->...hts",q,k)/ math.sqrt(head_dim) # [..., H, Tq, Tk]

  query_indices = query_start_index+ jnp.arange(query_length,dtype=jnp.int32) # [Tq]
  key_indices = jnp.arange(key_length,dtype=jnp.int32) # [Tk]

  mask = key_indices[None, :] <= query_indices[:, None] # [Tq, Tk]
  scores = jnp.where(mask,scores,-jnp.inf)
  weights = jax.nn.softmax(scores,axis=-1) # [..., H, Tq, Tk]
  context = jnp.einsum("...hts,...shd->...thd",weights,v) # [..., Tq, H, Dh]

  return context


# ---------- Multi-head causal self-attention ----------

class CausalSelfAttention(nn.Module):
  cfg: ModelConfig

  @nn.compact
  def __call__(self, x, positions, decode=False):
    """
    decode=False -> ordinary training and full-prefix inference
    decode=True  -> cached autoregressive inference
    """
    q = nn.Dense(
        self.cfg.d_model,
        use_bias=False,
        kernel_init=truncated_xavier_init,
        name="q_proj")(x) # [..., T, D]

    k = nn.Dense(
        self.cfg.d_model,
        use_bias=False,
        kernel_init=truncated_xavier_init,
        name="k_proj")(x) # [..., T, D]

    v = nn.Dense(
        self.cfg.d_model,
        use_bias=False,
        kernel_init=truncated_xavier_init,
        name="v_proj")(x) # [..., T, D]

    head_shape = x.shape[:-1] + ( self.cfg.num_heads,self.cfg.head_dim)

    q = q.reshape(head_shape) # [..., T, H, Dh]
    k = k.reshape(head_shape) # [..., T, H, Dh]
    v = v.reshape(head_shape) # [..., T, H, Dh]

    q = apply_rope(
        q,
        positions,
        self.cfg.rope_theta)

    k = apply_rope(
        k,
        positions,
        self.cfg.rope_theta)

    if decode:
      cache_shape = x.shape[:-2] + (self.cfg.context_length,
                                    self.cfg.num_heads,
                                    self.cfg.head_dim)

      cached_key = self.variable(
          "cache",
          "cached_key",
          lambda: jnp.zeros(
              cache_shape,
              dtype=k.dtype))

      cached_value = self.variable(
          "cache",
          "cached_value",
          lambda: jnp.zeros(
              cache_shape,
              dtype=v.dtype))

      cache_index = self.variable(
          "cache",
          "cache_index",
          lambda: jnp.array(
              0,
              dtype=jnp.int32))

      query_start_index = cache_index.value

      start_indices = (0,) * (k.ndim - 3)+ (query_start_index, 0, 0)

      new_cached_key = jax.lax.dynamic_update_slice(
          cached_key.value,
          k,
          start_indices)

      new_cached_value = jax.lax.dynamic_update_slice(
          cached_value.value,
          v,
          start_indices)

      cached_key.value = new_cached_key
      cached_value.value = new_cached_value

      cache_index.value = query_start_index + k.shape[-3]

      context = causal_self_attention(
          q,
          new_cached_key,
          new_cached_value,
          query_start_index=query_start_index)

    else:
      context = causal_self_attention(q,k,v)

    context = context.reshape(x.shape) # [..., T, D]

    output = nn.Dense(
        self.cfg.d_model,
        use_bias=False,
        kernel_init=truncated_xavier_init,
        name="out_proj")(context) # [..., T, D]

    return output


# ---------- SwiGLU feed-forward network ----------

class SwiGLU(nn.Module):
  cfg : ModelConfig

  @nn.compact
  def __call__(self, x):
    # x : [B,T,d_model]

    gate = nn.Dense(
        features = self.cfg.d_ff,
        use_bias = False,
        kernel_init=truncated_xavier_init,
        name = "gate_proj"
    )(x) # [B,T,d_model] -> [B,T, d_ff]

    up = nn.Dense(
        features = self.cfg.d_ff,
        use_bias = False,
        kernel_init=truncated_xavier_init,
        name = "up_proj"
    )(x) # [B,T,d_model] -> [B,T, d_ff]

    hidden = nn.silu(gate) * up # [B, T, d_ff]

    output = nn.Dense(
        features = self.cfg.d_model,
        use_bias = False,
        kernel_init=truncated_xavier_init,
        name = "down_proj"
    )(hidden) # [B,T,d_ff] -> [B,T, d_model]

    return output


# ---------- Pre-normalized Transformer block ----------

class TransformerBlock(nn.Module):
  cfg : ModelConfig

  @nn.compact
  def __call__(self, x, positions, decode=False):
    attention_input = RMSNorm(
        d_model = self.cfg.d_model,
        name = "attention_norm")(x) # [B, T, d_model]

    attention_update = CausalSelfAttention(
        cfg = self.cfg,
        name = "attention")(attention_input, positions, decode=decode) # [B, T, d_model]

    x = x + attention_update # First residual update

    mlp_input = RMSNorm(
        d_model = self.cfg.d_model,
        name = "mlp_norm")(x) # [B, T, d_model]

    mlp_update = SwiGLU(
        cfg = self.cfg,
        name = "mlp")(mlp_input) # [B, T, d_model]

    x = x + mlp_update # Second residual update

    return x


# ---------- Complete Decoder Only Transformer ----------

class DecoderOnlyTransformer(nn.Module):
  cfg : ModelConfig

  @nn.compact
  def __call__(self, token_ids, positions, decode=False):
    x = nn.Embed(
        num_embeddings = self.cfg.vocab_size,
        features = self.cfg.d_model,
        embedding_init = token_embedding_init,
        name = "token_embedding")(token_ids) # [B, T] -> [B, T, d_model]

    for layer_index in range(self.cfg.num_layers):
      x = TransformerBlock(
          cfg = self.cfg,
          name = f"block_{layer_index}")(x, positions, decode=decode) # [B, T, d_model]

    x = RMSNorm(
        d_model = self.cfg.d_model,
        name = "final_norm")(x) # [B, T, d_model]

    logits = nn.Dense(
        features = self.cfg.vocab_size,
        use_bias = False,
        kernel_init = truncated_xavier_init,
        name = "lm_head")(x) # [B ,T, d_model] -> [B ,T, vocab_size]

    return logits
