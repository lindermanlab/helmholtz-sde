import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from functools import partial

from sing.utils.params import *
from sing.sde import SDE, GPPost

from typing import Any, Dict

class InputSignals:
    """
    Represent inputs to latent dynamics
    """
    def __init__(self, v: jnp.array) -> None:
        """
        Params:
        --------------
        v: shape (n_trials, T, n_inputs) array, known input signals v(t)
        """
        self.v = v

    def update_input_effect(self, key: jr.PRNGKey, fn: SDE, t_grid: jnp.array, marginal_params: MarginalParams, trial_mask: jnp.array, inputs: jnp.array, gp_post: GPPost, drift_params: Dict[str, Any], jitter: float = 1e-8) -> jnp.array:
        """
        For the model dx(t) = {f(x(t)) + B v(t)} dt + dW(t), computes closed-form update for input effect matrix B

        Params:
        --------------
        fn: the prior SDE
        key: jr.PRNGKey
        t_grid: a shape (T) array, the time grid \tau
        marginal_params: sufficient statistics of the variational posterior over latents x
        trial_mask: a shape (batch_size, T) array, a binary mask indicating whether a trial is active
        inputs: a shape (batch_size, T, n_inputs) array, the model inputs
        gp_post: for SING-GP, the variational posterior over the drift f; else, None
        drift_params: a dictionary containing the parameters of the prior drift
        jitter: jitter when updating the input effect matrix  

        Returns:
        --------------
        B: the matrix mapping inputs into the latent space   
        """
        
        def _int_outer_dynamics_inputs(t_grid, fn, key, marginal_params, inputs, trial_mask, drift_params, gp_post): 
            """Computes sum((m_{t+1} - m{t}) - Delta_t * E[f(x_t)]) v(t)^T)""" 
            ms, Ss = marginal_params['m'], marginal_params['S'] 
            T, D = ms.shape
            del_t = t_grid[1:] - t_grid[:-1] # (T-1) 
            trans_mask = trial_mask[:-1] & trial_mask[1:] 

            def _f(mask, key_i, t_i, m_i, S_i):
                def do_eval(_): # ensures evaluation of E[f] does not nan after trial
                    val = fn.f(drift_params, key_i, t_i, m_i, S_i, gp_post)
                    return jnp.where(jnp.isfinite(val), val, jnp.zeros_like(val))
                def skip(_):
                    return jnp.zeros((D))
                return jax.lax.cond(mask, do_eval, skip, operand=None)
            E_f = jax.vmap(_f)(trans_mask, jr.split(key, T-1), t_grid[:-1], ms[:-1], Ss[:-1]) 

            ms_diffs = ms[1:] - ms[:-1] # (T-1, D)
            outer = jax.vmap(jnp.outer)(ms_diffs - del_t[..., None] * E_f, inputs[:-1]) # (T-1, D, I)
            outer = (trans_mask[:, None, None] * outer).sum(axis=0)   # (D, I)
            return outer

        def _int_outer_inputs(t_grid, inputs, trial_mask):
            """Computes Delta_t * sum(v(t) v(t)^T) from t=0 to t=T-1 (last timestep excluded)"""
            del_t = t_grid[1:] - t_grid[:-1] # (T-1)
            trans_mask = trial_mask[:-1] & trial_mask[1:] # (T-1)
    
            outer_prod = vmap(jnp.outer)(inputs[:-1], inputs[:-1])
            return (del_t[:,None,None] * trans_mask[:, None, None] * outer_prod).sum(0) # (I, I)
        
        batch_size, T, n_inputs = inputs.shape
        outer_inputs_term = vmap(partial(_int_outer_inputs, t_grid))(inputs, trial_mask).sum(0) # (I, I)
        outer_dynamics_inputs_term = vmap(partial(_int_outer_dynamics_inputs, t_grid, fn, drift_params=drift_params, gp_post=gp_post))(jr.split(key, batch_size), marginal_params, inputs, trial_mask).sum(axis=0) # (D, I)
        B = jnp.linalg.solve(outer_inputs_term + jitter * jnp.eye(n_inputs), outer_dynamics_inputs_term.T).T
        return B