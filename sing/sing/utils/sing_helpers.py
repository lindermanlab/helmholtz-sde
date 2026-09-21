"""
Contains helper functions specific to the SING variational inference algorithm
See sing/sing.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, value_and_grad
from jax.scipy.linalg import solve_triangular

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from sing.sde import SDE, GPPost
from sing.utils.params import *

from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

# --------------------- Functions for natural -> mean parameter conversion (parallelized) -------------------
class GaussianPotential(NamedTuple):
  """
  Represents a Gaussian potential phi(x_i, x_j).
  """
  J_diag_1: jnp.array
  J_diag_2: jnp.array
  h_1: jnp.array
  h_2: jnp.array
  J_lower_diag: jnp.array
  log_Z: jnp.array
  active_pair: jnp.bool_


def combine(a: GaussianPotential, b: GaussianPotential) -> GaussianPotential:
  return vmap(combine_elt)(a, b)

def combine_elt(a: GaussianPotential, b: GaussianPotential) -> GaussianPotential:
    J_a_1, J_a_2, h_a_1, h_a_2, J_a_lower, log_Z_a, act_a = a
    J_b_1, J_b_2, h_b_1, h_b_2, J_b_lower, log_Z_b, act_b = b
    act = act_a & act_b
    D = J_a_1.shape[0]

    def do_combine(_):
        # Condition on the shared middle node: a.right with b.left
        J_c = J_a_2 + J_b_1
        h_c = h_a_2 + h_b_1

        sqrt_Jc = jnp.linalg.cholesky(J_c)
        trm1 = solve_triangular(sqrt_Jc, h_c, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_a_lower, lower=True)
        trm3 = solve_triangular(sqrt_Jc, J_b_lower.T, lower=True)

        local_logZ  = 0.5 * D * jnp.log(2.0 * jnp.pi)
        local_logZ += -jnp.sum(jnp.log(jnp.diag(sqrt_Jc)))
        local_logZ += 0.5 * jnp.dot(trm1.T, trm1)

        J_p_1   = J_a_1 - trm2.T @ trm2
        J_p_2   = J_b_2 - trm3.T @ trm3
        J_p_lwr = -(trm3.T @ trm2)
        h_p_1   = h_a_1 - trm2.T @ trm1
        h_p_2   = h_b_2 - trm3.T @ trm1
        logZ_p  = log_Z_a + log_Z_b + local_logZ
        return GaussianPotential(J_p_1, J_p_2, h_p_1, h_p_2, J_p_lwr, logZ_p, act_b)

    def pass_through(_):
        # If the next segment is inactive, just carry a forward unchanged
        return a

    return lax.cond(act, do_combine, pass_through, operand=None)

def compute_log_normalizer_parallel(precision_diag_blocks: jnp.array, precision_lower_diag_blocks: jnp.array, linear_potential: jnp.array, trial_mask: jnp.array, jitter=1e-8) -> float:
    """
    Implements the parallelized log normalizer computation from Hu et al., 2025 for a linear, Gaussian Markov chain

    Returns:
    ------------
    log_normalizer: the log normalizer of the linear, Gaussian Markov chain
    """
    dim = precision_diag_blocks.shape[1]
    precision_diag_blocks = precision_diag_blocks + (trial_mask[:, None, None] * jitter) * jnp.eye(dim)[None, :, :] # only add jitter to active diagonals

    # Pad lower diag blocks with zero for the initial potential
    precision_lower_diag_blocks_pad = jnp.concatenate((jnp.zeros((1, dim, dim)), precision_lower_diag_blocks), axis=0)

    # Pad everything with zeros at the end to integrate out the last variable
    precision_diag_blocks_pad = jnp.concatenate([precision_diag_blocks, jnp.zeros((1, dim, dim))], axis=0)
    precision_lower_diag_blocks_pad = jnp.concatenate([precision_lower_diag_blocks_pad, jnp.zeros((1, dim, dim))], axis=0)
    linear_potential_pad = jnp.concatenate([linear_potential, jnp.zeros((1, dim))], axis=0)

    # Real edges in graph
    trans_mask = trial_mask[:-1] & trial_mask[1:] # binary mask indicating whether a transition occurs between x(i)->x(i+1)
    real_edge_pad = jnp.concatenate([trans_mask, jnp.array([False])], axis=0)  # (T)

    # Segment ends: alive at i but dead at i+1
    next_alive = jnp.concatenate([trial_mask[1:], jnp.array([False])], axis=0)  # (T)
    seg_end = trial_mask & (~next_alive)  # (T)

    # Node should be eliminated if it's a real edge OR a segment end
    active_pair = real_edge_pad | seg_end  # (T)
    active_pair = jnp.concatenate([trial_mask[0:1], active_pair], axis=0) # since first potential represents p(x(0))

    # Construct elements
    def construct_potential(p_diag, p_lower_diag, linear_potential, active_pair):
        return GaussianPotential(jnp.zeros_like(p_diag), p_diag, jnp.zeros_like(linear_potential), linear_potential, p_lower_diag, 0., active_pair)

    elems = vmap(construct_potential)(precision_diag_blocks_pad, precision_lower_diag_blocks_pad, linear_potential_pad, active_pair)

    # Perform the parallel associative scan
    scanned = lax.associative_scan(combine, elems)
    log_normalizer = scanned.log_Z[-1]
    return log_normalizer

def natural_to_mean_params(natural_params: NaturalParams, trial_mask: jnp.array, jitter=1e-8) -> Tuple[MeanParams, float]:
    """
    Parallel conversion between natural and mean parameters in a linear, Gaussian Markov chain
    Implemented by differentiating the log-normalizer 
    """
    precision_diag_blocks, precision_lower_diag_blocks, linear_potential = (-2)*natural_params['J'], (-1)*natural_params['L'], natural_params['h']

    # Take gradients of compute_log_normalizer_parallel to get mean parameters.
    f = value_and_grad(compute_log_normalizer_parallel, argnums=(0, 1, 2))
    log_normalizer, grads = f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential, trial_mask, jitter)

    # Correct for the -1/2 J -> J implementation
    ExxT = -2 * grads[0]
    ExxnT = -grads[1]
    Ex = grads[2]
    return {'Ex': Ex, 'ExxT': ExxT, 'ExxnT': ExxnT}, log_normalizer

def natural_to_marginal_params(natural_params: NaturalParams, trial_mask: jnp.array, jitter=1e-8) -> Tuple[MarginalParams, float]:
    """
    Parallel conversion between natural and marginal parameters in a linear, Gaussian Markov chain
    """
    mean_params, log_normalizer = natural_to_mean_params(natural_params, trial_mask, jitter)
    marginal_params = mean_to_marginal_params(mean_params)
    return marginal_params, log_normalizer

# --------------------- Other functions for parameter conversion -------------------
def mean_to_marginal_params(mean_params: MeanParams) -> MarginalParams:
    """
    Conversion between mean and marginal parameters
    """
    Ex, ExxT, ExxnT = mean_params['Ex'], mean_params['ExxT'], mean_params['ExxnT']
    T = Ex.shape[0]
    outer = vmap(lambda x1, x2: jnp.outer(x1, x2))
    return {'m': Ex, 'S': ExxT - outer(Ex, Ex), 'SS': ExxnT - outer(Ex[1:], Ex[:(T-1)])}

def marginal_to_mean_params(marginal_params: MarginalParams) -> MeanParams:
    """
    Inverse of mean_to_marginal_params
    """
    m, S, SS = marginal_params['m'], marginal_params['S'], marginal_params['SS']
    T = m.shape[0]
    outer = vmap(lambda x1, x2: jnp.outer(x1, x2))
    return {'Ex': m, 'ExxT': S + outer(m, m), 'ExxnT': SS + outer(m[1:], m[:(T-1)])}

def dynamics_to_natural_params(A: jnp.array, b: jnp.array, Q: jnp.array, trial_mask: jnp.array) -> NaturalParams:
    """
    Function for computing the natural parameters from parameters A, b, Q of a linear, Gaussian Markov chain
    
    Params:
    -------------
    A: a shape (T-1, D, D) array of transition matrices
    b: a shape (T, D) array of offsets, with b[0] being the initial mean
    Q: a shape (T, D, D) array of noise covariance matrices, with Q[0] being the initial covariance
    """

    T, D = b.shape
    trans_mask =  trial_mask[:-1] & trial_mask[1:] 

    # Diagonal blocks of precision matrix
    invQ = jnp.linalg.inv(Q)
    invQ_masked = jnp.where(trial_mask[:, None, None], invQ, jnp.zeros((T, D, D)))

    cross = vmap(lambda A_t, invQ_tp1: A_t.T @ invQ_tp1 @ A_t)(A, invQ[1:])     
    cross = jnp.where(trans_mask[:, None, None], cross, jnp.zeros((T-1, D, D)))                           
    J_diag = invQ_masked.at[:-1].add(cross)

    # Lower diagonal blocks of precision matrix
    J_lower_diag = -vmap(lambda invQ_tp1, A_t: invQ_tp1 @ A_t)(invQ[1:], A)
    J_lower_diag = jnp.where(trans_mask[:, None, None], J_lower_diag, jnp.zeros((T-1, D, D)))

    # Linear potential
    h_base = vmap(lambda invQ_t, b_t: invQ_t @ b_t)(invQ, b)
    h_base = jnp.where(trial_mask[:, None], h_base, jnp.zeros((T, D))) 
    h_corr = -vmap(lambda A_t, invQ_tp1, b_tp1: A_t.T @ (invQ_tp1 @ b_tp1))(A, invQ[1:], b[1:])
    h_corr = jnp.where(trans_mask[:, None], h_corr, jnp.zeros((T-1, D)))
    h = h_base.at[:-1].add(h_corr)

    return {'J': (-1/2)*J_diag, 'L': (-1)*J_lower_diag, 'h': h}

def mean_to_natural_params(mean_params: MeanParams, trial_mask: jnp.array, jitter=1e-8) -> NaturalParams: 
    """
    Conversion between mean and natural parameters in a linear, Gaussian Markov chain
    """

    marginal_params = mean_to_marginal_params(mean_params) # first convert to marginal params
    ms, Ss, SSs = marginal_params['m'], marginal_params['S'], marginal_params['SS']
    _, D = ms.shape

    # First construct parameters of the linear, Gaussian Markov chain
    # Compute transition matrices
    I = jnp.eye(D)
    trans_mask = (trial_mask[:-1] & trial_mask[1:])  # (T-1)
    def safe_inv(S, mask): # avoids inverting an ill-conditioned matrix (if the trial ends early)
        return jnp.where(mask, jnp.linalg.inv(S + jitter * I), jnp.zeros_like(S))
    invSs = jax.vmap(safe_inv)(Ss[:-1], trans_mask)  # (T-1, D, D)
    A = jax.vmap(lambda SS_t, invS_t: SS_t.T @ invS_t)(SSs, invSs)

    b0 = ms[0]
    b_rest = vmap(lambda A_t, m_t, m_tp1: m_tp1 - A_t @ m_t)(A, ms[:-1], ms[1:])
    b = jnp.vstack([b0[None, :], b_rest])  # (T, D)

    Q0 = Ss[0]
    Q_rest = vmap(lambda A_t, S_t, S_tp1: S_tp1 - A_t @ S_t @ A_t.T)(A, Ss[:-1], Ss[1:])
    Q = jnp.vstack([Q0[None, :, :], Q_rest]) # (T, D, D)

    # Then compute corresponding natural parameters
    return dynamics_to_natural_params(A, b, Q, trial_mask)

# ------------------------------- Helper functions for variational EM ---------------------------
class InitParams(TypedDict):
    """
    The initial parameters of the prior SDE

    - mu0: the initial mean
    - V0: the inial covariance
    """
    mu0: jnp.array
    V0: jnp.array

def compute_gaussian_entropy(natural_params: NaturalParams, marginal_params: MarginalParams, log_normalizer: float) -> float:
    """
    Computes the entropy, -E_p[log p(x)], of a multivariate Gaussian with block tridiagonal precision matrix

    Returns:
    ------------
    ent: the entropy of the multivariate Gaussian distribution
    """
    mean_params = marginal_to_mean_params(marginal_params) # convert marginal -> mean parameters
    Ex, ExxT, ExxnT = mean_params['Ex'], mean_params['ExxT'], mean_params['ExxnT']
    inner = lambda x1, x2: jnp.trace(x1.T @ x2)
    tr_term = vmap(inner)(natural_params['J'], ExxT).sum()
    tr_term += vmap(inner)(natural_params['L'], ExxnT).sum()
    tr_term += (Ex * natural_params['h']).sum()
    return log_normalizer - tr_term

def compute_neg_CE_initial(m0: jnp.array, S0: jnp.array, mu0: jnp.array, V0: jnp.array) -> float:
    """
    Computes the negative cross-entropy
    E_{q(x0)}[log p(x0)], where q(x0) = N(x0 | m0, S0)
    """
    p = tfd.MultivariateNormalFullCovariance(loc = mu0, covariance_matrix = V0)
    q = tfd.MultivariateNormalFullCovariance(loc = m0, covariance_matrix = S0)
    return -q.cross_entropy(p)

def compute_neg_CE_single(fn: SDE, gp_post: GPPost, drift_params: Dict[str, Any], key: jr.PRNGKey, t: float, del_t: float, mt: jnp.array, mt_next: jnp.array, St: jnp.array, St_next: jnp.array, SS: jnp.array, input_t: jnp.array, input_effect: jnp.array, sigma=1.) -> float:
    """
    Computes the negative cross-entropy
    E_f[[E_q[log p(x_{i+1}|x_i)]] 
    for a single i between 0 and T-1;
    NOTE: the expectation over f can be ignored if the prior drift is modelled as deterministic

    Params:
    ------------
    fn: the prior SDE 
    gp_post: for SING-GP, the GP posterior 
    drift_params: a dictionary containing the parameters of the prior SDE drift
    key: random key for sampling
    t: \tau_i, the time which the cross-entropy is computed
    del_t:\tau_{i+1} - \tau_i, the difference between time \tau_i and the subsequent timestep
    mt: a shape (D) array, the mean at time \tau_i
    mt_next: a shape (D) array, the mean at time \tau_{i+1}
    St: a shape (D, D) array, the covariance at time \tau_i
    St_next: a shape (D, D) array, the covariance at time \tau_{i+1}
    SS: a shape (D, D) array, the covariance between the states at times \tau_{i+1} and \tau_{i}
    input_t: a shape (n_inputs) array, the input at time \tau_i
    input_effect: a shape (D, n_inputs) array, the linear mapping from inputs to the latent space
    sigma: the noise scale of the prior SDE

    Returns:
    ------------ 
    neg_CE: the negative cross-entropy
    """
    keys = jr.split(key, 3) # need to compute three expectations
    
    Ef = fn.f(drift_params, keys[0], t, mt, St, gp_post)
    Eff = fn.ff(drift_params, keys[1], t, mt, St, gp_post)
    Edfdx = fn.dfdx(drift_params, keys[2], t, mt, St, gp_post)
    
    const = -0.5 * len(mt) * jnp.log(2 * jnp.pi * del_t * (sigma**2))
    trm = jnp.trace(St_next + jnp.outer(mt_next, mt_next))
    trm += jnp.trace(St + jnp.outer(mt, mt))
    trm += -2 * jnp.trace(SS + jnp.outer(mt_next, mt))
    trm += (del_t)**2 * Eff
    trm += -2 * del_t * jnp.trace(jnp.outer(Ef, mt_next) + Edfdx @ SS.T)
    trm += 2 * del_t * jnp.trace(jnp.outer(Ef, mt) + Edfdx @ St)

    trm += (del_t)**2 * ((input_effect @ input_t)**2).sum() 
    trm += -2 * del_t * jnp.dot(input_effect @ input_t, mt_next - mt - del_t * Ef)

    trm *= -1.0 / (2 * del_t * sigma**2)
    return const + trm

def compute_neg_CE(t_grid: jnp.array, fn: SDE, gp_post: GPPost, drift_params: Dict[str, Any], init_params: InitParams, key: jr.PRNGKey, marginal_params: MarginalParams, inputs: jnp.array, trial_mask: jnp.array, input_effect: jnp.array, sigma=1.) -> float:
    """
    Computes total expected negative cross entropy, -E_f[E_q[log p(x_0,...,x_T)].
    """
    T = t_grid.shape[0]
    ms, Ss, SSs = marginal_params['m'], marginal_params['S'], marginal_params['SS']
    
    neg_CE_init = compute_neg_CE_initial(ms[0], Ss[0], init_params['mu0'], init_params['V0'])
    neg_CE_init = jnp.where(trial_mask[0], neg_CE_init, 0.0)

    def per_step(mask, key_i, t_i, dt_i, m_i, m_ip1, S_i, S_ip1, SS_i, u_i):
        def do():
            val = compute_neg_CE_single(fn, gp_post, drift_params, key_i, t_i, dt_i, m_i, m_ip1, S_i, S_ip1, SS_i, u_i, input_effect, sigma)
            return jnp.where(jnp.isfinite(val), val, 0.0)
        return lax.cond(mask, do, lambda: 0.0)
    trans_mask = trial_mask[:-1] & trial_mask[1:]
    neg_CE_rest = jax.vmap(per_step)(trans_mask, jr.split(key, T-1), t_grid[:-1], t_grid[1:] - t_grid[:-1], ms[:-1], ms[1:], Ss[:-1], Ss[1:], SSs, inputs[:-1])

    neg_CE_rest = (neg_CE_rest * trans_mask).sum()
    return neg_CE_init + neg_CE_rest

def update_init_params(m0: jnp.array, S0: jnp.array) -> InitParams:
    """
    Updates the initial mean mu0 and covariance V0 of the prior

    Params:
    ------------
    m0: a shape (D) array, the initial mean of the variational posterior
    S0: a shape (D, D) array, the initial covariance of the variational posterior
    """
    mu0, V0 = m0, S0
    init_params = {'mu0': mu0, 'V0': V0}
    return init_params

# -------------------- Mini-batching helpers -------------------------
def subset_batches(args: list, batch_inds: jnp.array) -> list:
    return [jax.tree_util.tree_map(lambda x: x[batch_inds], arg) for arg in args]

def fill_batches(args: list, batch_args: list, batch_inds: jnp.array) -> list:
    return [jax.tree_util.tree_map(lambda x, y: x.at[batch_inds].set(y), arg, batch_arg) for (arg, batch_arg) in zip(args, batch_args)]