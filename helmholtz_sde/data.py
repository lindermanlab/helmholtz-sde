"""
Classes for processing data in the training loop
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from typing import Callable, NamedTuple, Optional

class ProcessedData(NamedTuple):
    """
    Tuple representing processed observations
    """
    ys: jnp.array # (B, T, D) observations
    obs_times: jnp.array # (B, T) observation times
    mask: jnp.array # (B, T) binary mask, indicating whether or not each observation is active; NOTE: the mask must be contiguous
    mask_kl0: Optional[jnp.array] = None # (B, T_full) optional mask for the KL0 encoder pass over the original unprocessed data of length T_full; if None, defaults to all-ones
    kl_t0: Optional[jnp.array] = None # (B) start times for KL integral; if None, inferred from mask
    kl_t1: Optional[jnp.array] = None # (B) end times for KL integral; if None, inferred from mask


DataProcessor = Callable[[jnp.array, jnp.array, jr.PRNGKey], ProcessedData]


def process_data_identity(ys: jnp.array, obs_times: jnp.array, key: jr.PRNGKey) -> ProcessedData:
    """
    Default data processor: return the original dataset and an all-ones mask
    """
    mask = jnp.ones(ys.shape[:2], dtype=ys.dtype)
    return ProcessedData(ys=ys, obs_times=obs_times, mask=mask)


def process_data_full_kl(ys: jnp.array, obs_times: jnp.array, key: jr.PRNGKey, t_max: float = 1.0) -> ProcessedData:
    """
    Like process_data_identity, but integrates KL over [0, t_max] instead of [first_obs, last_obs]

    Useful when first observation does not occur at time zero
    """
    B = ys.shape[0]
    mask = jnp.ones(ys.shape[:2], dtype=ys.dtype)
    kl_t0 = jnp.zeros((B,), dtype=ys.dtype)
    kl_t1 = jnp.full((B,), t_max, dtype=ys.dtype)
    return ProcessedData(ys=ys, obs_times=obs_times, mask=mask, kl_t0=kl_t0, kl_t1=kl_t1)


def process_data_sample_independent_windows(ys_full: jnp.array, obs_times_full: jnp.array, key: jr.PRNGKey, min_len: int, max_len: int, batch_size: Optional[int] = None) -> ProcessedData:
    """
    Sample per-item independent windows with replacement over sequences

    Returns a fixed-length padded window of length Lw = min(T, max_len) together with a binary mask marking the active prefix of each sampled window
    """
    B_full, T, D = ys_full.shape
    Lw = int(min(T, max_len))

    if Lw < 1:
        Bb = B_full if batch_size is None else int(batch_size)
        ys0 = jnp.broadcast_to(ys_full[0:1, :1, :], (Bb, 1, D))
        ts0 = jnp.broadcast_to(obs_times_full[0:1, :1], (Bb, 1))
        mask0 = jnp.ones((Bb, 1), dtype=ys_full.dtype)
        return ProcessedData(ys=ys0, obs_times=ts0, mask=mask0)

    Bb = B_full if batch_size is None else int(batch_size)
    l_min = min(max(1, int(min_len)), Lw)

    k_idx, k_len, k_u = jr.split(key, 3)
    idx = jr.randint(k_idx, shape=(Bb,), minval=0, maxval=B_full)
    win_len = jr.randint(k_len, shape=(Bb,), minval=l_min, maxval=Lw + 1)

    # Start is valid for a true window of length win_len
    start_max = T - win_len
    u01 = jr.uniform(k_u, shape=(Bb,))
    start = jnp.floor(u01 * (start_max.astype(ys_full.dtype) + 1.0)).astype(jnp.int32)

    # Pad on the right so we can always extract a fixed-length Lw window
    ys_pad_full = jnp.pad(ys_full, ((0, 0), (0, Lw), (0, 0)))
    ts_pad_full = jnp.pad(obs_times_full, ((0, 0), (0, Lw)))

    def make_one(idx_i, start_i, len_i):
        y_seq = ys_pad_full[idx_i]
        t_seq = ts_pad_full[idx_i]

        y_win = jax.lax.dynamic_slice_in_dim(y_seq, start_i, slice_size=Lw, axis=0)
        t_win = jax.lax.dynamic_slice_in_dim(t_seq, start_i, slice_size=Lw, axis=0)

        mask = (jnp.arange(Lw, dtype=jnp.int32) < len_i).astype(ys_full.dtype)
        return y_win, t_win, mask

    ys_pad, obs_times_pad, mask = vmap(make_one)(idx, start, win_len)
    return ProcessedData(ys=ys_pad, obs_times=obs_times_pad, mask=mask)


def process_data_sample_random_subsequence(ys_full: jnp.array, obs_times_full: jnp.array, key: jr.PRNGKey, min_len: int = 3, max_len: int = 300) -> ProcessedData:
    """
    Sample one common subsequence window for the whole batch and pad on the right
    """
    ys_full = jnp.asarray(ys_full)
    obs_times_full = jnp.asarray(obs_times_full)

    B, T, D = ys_full.shape
    Lw = int(min(T, max_len))

    if Lw < 1:
        ys0 = ys_full[:, :1, :]
        ts0 = obs_times_full[:, :1]
        mask0 = jnp.ones((B, 1), dtype=ys_full.dtype)
        return ProcessedData(ys=ys0, obs_times=ts0, mask=mask0)

    l_min = min(max(1, int(min_len)), Lw)

    key_len, key_start = jr.split(key, 2)

    # Shared active length
    win_len = jr.randint(key_len, shape=(), minval=l_min, maxval=Lw + 1)
    win_len = jnp.asarray(win_len, dtype=jnp.int32)

    start_max = T - win_len
    s = jr.randint(key_start, shape=(), minval=0, maxval=start_max + 1)
    s = jnp.asarray(s, dtype=jnp.int32)

    # Pad with zeros to ensure we can always extract a chunk of size Lw
    ys_pad_full = jnp.pad(ys_full, ((0, 0), (0, Lw), (0, 0)))
    ts_pad_full = jnp.pad(obs_times_full, ((0, 0), (0, Lw)))

    ys_slice = jax.lax.dynamic_slice_in_dim(ys_pad_full, s, slice_size=Lw, axis=1)
    ts_slice = jax.lax.dynamic_slice_in_dim(ts_pad_full, s, slice_size=Lw, axis=1)

    valid_mask = (jnp.arange(Lw, dtype=jnp.int32) < win_len).astype(ys_full.dtype)
    mask = jnp.broadcast_to(valid_mask, (B, Lw))

    ys_pad = ys_slice * mask[..., None]
    ts_pad = ts_slice * mask
    return ProcessedData(ys=ys_pad, obs_times=ts_pad, mask=mask)
