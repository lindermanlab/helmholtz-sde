"""
Encoders of the observations into a context sequence, and functions combining the context across time, shared by all
posterior parameterizations
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Optional

# --------------------- Encoders for observations ---------------------
class NullEncoder(nn.Module):
    """
    A null encoder, returns a pre-specified value for all observation times
    To be used when performing non-amortized inference
    """
    value: float = 0.0

    @nn.compact
    def __call__(self, ys: jnp.array, mask: Optional[jnp.array] = None) -> jnp.array:
        del mask
        T = ys.shape[0]
        ctx_seq = self.value * jnp.ones((T+1, 1), dtype=ys.dtype)
        return ctx_seq


class MaskedGRUCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry: jnp.array, inputs):
        y_t, m_t = inputs
        gru_cell = nn.GRUCell(features=self.hidden_size)
        new_carry, h_t = gru_cell(carry, y_t)
        m_t = m_t.astype(carry.dtype)

        carry_new = carry + (new_carry - carry) * m_t
        h_new = carry + (h_t - carry) * m_t
        return carry_new, h_new


class GRUEncoder(nn.Module):
    """
    A GRU that encodes a sequence of observations (y_1, ..., y_N) backward in time

    NOTE: this differs from the implementation in Bartosh et al., 2025 which decodes observations forward in time
    """
    hidden_size: int

    @nn.compact
    def __call__(self, ys: jnp.array, mask: Optional[jnp.array] = None) -> jnp.array:
        T, D = ys.shape

        ys_rev = ys[::-1, :]
        if mask is None:
            mask_rev = jnp.ones((T,), dtype=ys.dtype)
        else:
            mask_rev = mask[::-1].astype(ys.dtype)

        carry0 = jnp.zeros((self.hidden_size,), dtype=ys.dtype)

        masked_gru_scan = nn.scan(
            MaskedGRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(hidden_size=self.hidden_size)

        final, hs_rev = masked_gru_scan(carry0, (ys_rev, mask_rev))
        hs = hs_rev[::-1]
        ctx_seq = jnp.concatenate([final[None, :], hs], axis=0)
        return ctx_seq


class ForwardGRUEncoder(nn.Module):
    """
    A GRU that encodes a sequence of observations (y_1, ..., y_N) forward in time
    """
    hidden_size: int

    @nn.compact
    def __call__(self, ys: jnp.array, mask: Optional[jnp.array] = None) -> jnp.array:
        T, D = ys.shape

        if mask is None:
            mask = jnp.ones((T,), dtype=ys.dtype)
        else:
            mask = mask.astype(ys.dtype)

        carry0 = jnp.zeros((self.hidden_size,), dtype=ys.dtype)

        masked_gru_scan = nn.scan(
            MaskedGRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(hidden_size=self.hidden_size)

        final, hs = masked_gru_scan(carry0, (ys, mask))
        ctx_seq = jnp.concatenate([final[None, :], hs], axis=0)
        return ctx_seq


# --------------------- Functions for combining embeddings across time ---------------------
def constant_ctx(obs_times: jnp.array, ctx_seq: jnp.array, t: jnp.array, mask: Optional[jnp.array] = None) -> jnp.array:
    """
    A constant context
    To be used when performing non-amortized inference
    """
    return ctx_seq[0]


def weighted_ctx(obs_times: jnp.array, ctx_seq: jnp.array, t: jnp.array, beta=1., mask: Optional[jnp.array] = None) -> jnp.array:
    """
    Computes a weighted average of embeddings outputted from a GRU
    """
    out = ctx_seq[1:] # (T, H)
    h = ctx_seq[0] # (H)

    scores = -beta * (obs_times - t) ** 2 # (T)
    if mask is not None:
        scores = jnp.where(mask > 0, scores, -1e30)
    w = jax.nn.softmax(scores, axis=0) # (T)
    ctx = (w[:, None] * out).sum(axis=0) # (H)
    return ctx + h
