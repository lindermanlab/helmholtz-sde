import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jacfwd
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from abc import ABC, abstractmethod

from sing.expectation import Expectation, MonteCarlo
from sing.kernels import Kernel
from sing.utils.general_helpers import make_gram
from sing.utils.params import MarginalParams

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypedDict, Union

class SDE(ABC):
    """
    Abstract base class representing an SDE of the form

    dx(t) = f(x(t), t) dt + L dW(t)
    """
    def __init__(self, expectation: Optional[Expectation] = None, latent_dim: int = 1) -> None:
        """
        Params:
        ------------
        expectation: object for computing expectations of the prior drift under the variational posterior
        latent_dim: the output dimension of the kernel
        """
        super().__init__()
        self.latent_dim = latent_dim

        if expectation is None:
            # Use Monte Carlo approximation with 1 sample by default
            self.expectation = MonteCarlo(latent_dim, N=1)
        else:
            self.expectation = expectation
    
    @abstractmethod
    def drift(self, drift_params: Dict[str, Any], x: jnp.array, t: jnp.array) -> jnp.array:
        """
        Drift function of the SDE f(x(t),t)
        
        Params:
        ------------
        drift_params: a dictionary containing the SDE parameters
        x: a shape (D) array, the state at time t
        t: a shape (1) array, the time t
        
        Returns:
        ------------
        drift_val: f(x(t), t)
        """
        raise NotImplementedError
    
    def prior_term(self, *args) -> float:
        return 0.
      
    def f(self, drift_params: Dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        """
        Computes the expected value of the drift f(x_i, tau_i) under q(x_i), E_q(x_i)[f(x_i, tau_i)]
        where q is multivariate Gaussian with mean m and covariance S
        
        Params:
        ------------
        drift_params: a dictionary containing the SDE parameters
        key: a random key used for computing the expectation
        t: a shape (1) array, the time t
        m: a shape (D) array, the mean of the state x_i under q
        S: a shape (D, D) array, the covariance of the state x_i under q
        
        Returns:
        ------------
        Ef: the expectation of the drift under q(x_i)
        """
        fx = partial(self.drift, drift_params, t=t) # f: R^D -> R^D
        return self.expectation.gaussian_int(key=key, fn=fx, m=m, S=S) # (D)
    
    def ff(self, drift_params: dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        """
        Computes the expected value of |f(x_i,tau_i)|^2 under q(x_i)
        """
        ffx = lambda x: jnp.sum(jnp.square(self.drift(drift_params, x, t))) # ffx: R^D -> R
        return self.expectation.gaussian_int(key=key, fn=ffx, m=m, S=S) # scalar
    
    def dfdx(self, drift_params: dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        """
        Computes the expected value of d/dx f(x, t) at (x, t) = (x_i, tau_i) under q(x_i)
        """
        fx = jacfwd(partial(self.drift, drift_params, t=t)) # Df: R^D -> R^{D x D}
        return self.expectation.gaussian_int(key=key, fn=fx, m=m, S=S)  # (D, D)
        
    def update_dynamics_params(self, *args):
        """
        If the drift is modeled with a Baysian prior, update the (variational) posterior on the drift
        """
        return None

class LinearSDE(SDE):
    """
    A class for linear SDEs with constant drift term
    i.e. dx(t) = {Ax(t) + b}dt + dW(t)

    eg., the Ornstein–Uhlenbeck (OU) process dx(t) = -theta x(t) dt + dW(t)
    """
    def __init__(self, latent_dim: int = 1) -> None:
        super().__init__(None, latent_dim) # can compute expectations of the prior drift analytically
    
    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        A = drift_params['A']
        b = drift_params['b']
        return A @ x + b

    def f(self, drift_params: dict[str, jnp.array], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        A, b = drift_params['A'], drift_params['b']
        return A @ m + b

    def ff(self, drift_params: dict[str, jnp.array], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        A, b = drift_params['A'], drift_params['b']
        return jnp.trace(A.T @ A @ (S + jnp.outer(m, m))) + 2 * jnp.dot(b, A @ m) + jnp.dot(b, b)

    def dfdx(self, drift_params: dict[str, jnp.array], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, *args, **kwargs) -> jnp.array:
        return drift_params['A'] # (D, D)

class BasisSDE(SDE):
    """
    A class for SDEs whose drift coefficients are a weighted sum of (potentially time dependent) basis functions 
    f(x, t) = w_1 b_1(x, t) + ... + w_k b_k(x, t)
    """
    def __init__(self, basis_set: Callable[[jnp.array, float], jnp.array], expectation: Optional[Expectation] = None, latent_dim: int = 1) -> None:
        """
        Params:
        ------------
        basis_set: a function that takes as input x, a shape (D) array, and t, a shape (1) array, and returns 
        a shape (n_basis) array representing n_basis basis functions evaluated at the pair (x, t)
        """
        super().__init__(expectation, latent_dim)
        self.basis_set=basis_set
    
    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        w = drift_params['w']
        return jnp.dot(w, self.basis_set(x, t))

class VanDerPol(SDE):
    """
    A class implementing the Van der Pol oscillator, governed by the equations
        dx/dt = tau * mu * (x - x^3/3 - y)
        dy/dt = tau * x / mu
    plus the Brownian increment dW(t)
    """
    def __init__(self, expectation: Optional[Expectation] = None):
        super().__init__(expectation, latent_dim=2)

    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        f1 = drift_params['tau'] * drift_params['mu'] * (x[0] - x[0]**3 / 3. - x[1])
        f2 = drift_params['tau'] * x[0] / drift_params['mu']
        return jnp.array([f1, f2])

class DuffingOscillator(SDE):
    """
    A class implementing the Duffing oscillator, governed by the equations
        dx/dt = y
        dy/dt = alpha * x - beta * x^3 - gamma * y
    plus the Brownian increment dW(t)
    """

    def __init__(self, expectation: Optional[Expectation] = None):
        super().__init__(expectation, latent_dim=2)

    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        drift_scale = drift_params.get('drift_scale', 1.)
        f1 = x[1]
        alpha, beta, gamma = drift_params['alpha'].item(), drift_params['beta'].item(), drift_params['gamma'].item()
        f2 = alpha * x[0] - beta * x[0]**3 - gamma * x[1]
        return drift_scale * jnp.array([f1, f2])

class LorenzAttractor(SDE):
    """
    A class implementing the Lorenz attractor, governed by the equations
        dx/dt =  a1 * (y - x)
        dy/dt =  a2 * x - y - x * z
        dz/dt = x * y - a3 * z
    plus the Brownian increment dW(t)
    """

    def __init__(self, expectation: Optional[Expectation] = None) -> None:
        super().__init__(expectation, latent_dim=3)
    
    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        a = drift_params['a']

        f1 = a[0] * (x[1] - x[0])
        f2 = a[1] * x[0] - x[1] - x[0] * x[2]
        f3 = x[0] * x[1] - a[2] * x[2]
        return jnp.array([f1, f2, f3])

class EmbeddedLorenzAttractor(SDE):
    """
    A class implementing the "embedded" Lorenz attractor, whose first three dimensions evolve according
    to the Lorenz attractor and whose remaining dimensions evolve according to a simple random walk
    """

    def __init__(self, expectation: Optional[Expectation] = None, latent_dim: int = 3) -> None:
        assert latent_dim >= 3, "require latent_dim greater than or equal to 3"
        super().__init__(expectation, latent_dim=latent_dim)
    
    def drift(self, drift_params: Dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        a = drift_params['a']

        f1 = a[0] * (x[1] - x[0])
        f2 = a[1] * x[0] - x[1] - x[0] * x[2]
        f3 = x[0] * x[1] - a[2] * x[2]

        rw_drifts = jnp.zeros(self.latent_dim - 3)
        return jnp.concatenate([jnp.array([f1, f2, f3]), rw_drifts])

class DoubleWell(SDE):
    """
    A class implementing the (one-dimensional) double-well SDE, governed by
        dx/dt = theta_0 x(theta_1 - x^2)
    plus the Brownian increment dW(t)
    """
    def __init__(self, expectation: Optional[Expectation] = None) -> None:
        super().__init__(expectation, latent_dim=1)

    def drift(self, drift_params: dict[str, jnp.array], x: jnp.array, t: jnp.array) -> jnp.array:
        return drift_params['theta0'] * x * (drift_params['theta1'] - jnp.square(x))

class NeuralSDE(SDE):
    """
    A class implementing a neural SDE, an SDE with drift parameterized by a neural network
    """
    def __init__(self, apply_fn: Callable[[Dict[str, Any], jnp.ndarray, jnp.ndarray], jnp.ndarray], expectation: Optional[Expectation] = None, latent_dim: int = 1) -> None:
        """
        Params:
        ------------
        apply_fn: a function with signature apply_fn(params, x, t) -> jnp.ndarray of shape (latent_dim)

        Example:
        ------------
        from sing.utils.general_helpers import MLP
        model = MLP(features=[64, 64], latent_dim=latent_dim) # instantiate NN object
        model_key = jr.PRNGKey(0)
        
        x0 = jnp.zeros((1, latent_dim)) # just for initialization
        t0 = jnp.zeros((1))
        network_params = model.init(model_key, x0, t0) # initialize NN parameters
        sde_params = {'network_params': network_params} 

        fn = NeuralSDE(quadrature=quadrature, apply_fn=model.apply, latent_dim=latent_dim)
        """
        
        super().__init__(expectation, latent_dim)
        self.apply_fn = apply_fn

    def drift(self, drift_params: Dict[str, Any], x: jnp.array, t: jnp.array) -> jnp.array:
        return self.apply_fn(drift_params['network_params'], x, t)

class GPPost(TypedDict):
    """
    Represents the GP variational posterior according to the sparse variational GP framework (see Titsias, 2009 and Duncker et al., 2019)
    
    - q_u_mu: a shape (n_inducing) array, the posterior mean over inducing points
    - q_u_sigma: a shape (n_inducing, n_inducing) array, the posterior variance over inducing points
    """

    q_u_mu: jnp.array
    q_u_sigma: jnp.array

class SparseGP(SDE):
    """
    A class implementing a GP-SDE, an SDE with a Gaussian process (GP) prior on the drift

    Approximate inference is performed on the GP prior drift using the sparse variational GP framework (see Titsias, 2009 and Duncker et al., 2019)
    """
    def __init__(self, zs: jnp.array, kernel: Kernel, expectation: Optional[Expectation] = None, jitter: float = 1e-8) -> None:
        """
        Params:
        ------------
        expectation: object for computing expectations of the prior drift under the variational posterior, 
        either a GaussHermiteQuadrature or GaussianMonteCarlo object
        zs: a shape (n_inducing, D) array, represents the grid of inducing points on R^D that determine the variational posterior
        kernel: the positive semi-definite kernel function K: R^D x R^D -> R_+ that defines the Gaussian process prior
        jitter
        """
        self.zs = zs
        self.kernel = kernel
        self.jitter = jitter
        super().__init__(expectation, latent_dim=self.zs.shape[-1])
    
    def drift(self, drift_params: Dict[str, Any], x: jnp.array, t: jnp.array) -> None:
        return None
        
    def prior_term(self, drift_params: Dict[str, Any], gp_post: GPPost) -> float:
        """
        Computes sum of KL[q(u_d)||p(u_d)]] across D dimensions d = 1, ..., D
        """
        Kzz = vmap(vmap(partial(self.kernel.K, kernel_params=drift_params), (None, 0)), (0, None))(self.zs, self.zs) + self.jitter * jnp.eye(len(self.zs)) # (n_inducing, n_inducing)
        q_dist = tfd.MultivariateNormalFullCovariance(gp_post['q_u_mu'], gp_post['q_u_sigma'])
        p_dist = tfd.MultivariateNormalFullCovariance(0, Kzz) # one sample is shape (n_inducing) (prior dist is same across dimensions)
        kl = tfd.kl_divergence(q_dist, p_dist).sum()
        return -kl

    def get_posterior_f_mean(self, gp_post: GPPost, drift_params: Dict[str, Any], xs: jnp.array) -> jnp.array:
        """
        Computes posterior mean q(f) on a grid of points xs
        """
        Kxz = make_gram(self.kernel.K, drift_params, xs, self.zs, jitter=None)
        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)
        f_mean = (Kxz @ jnp.linalg.solve(Kzz, gp_post['q_u_mu'].T))
        return f_mean
    
    def get_posterior_f_var(self, gp_post: GPPost, drift_params: Dict[str, Any], xs: jnp.array) -> jnp.array:
        """
        Computes posterior variance under q(f) at a grid of points xs.

        Returns: 
        -----------
        f_var: a shape (N_pts, N_pts) array, the posterior variance of f on the specified grid of xs 
        """      
        Kxx = make_gram(self.kernel.K, drift_params, xs, xs, jitter=self.jitter)
        Kxz = make_gram(self.kernel.K, drift_params, xs, self.zs, jitter=None)
        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)

        f_var = jnp.diagonal(Kxx - Kxz @ jnp.linalg.solve(Kzz, Kxz.T) + Kxz @ jnp.linalg.solve(Kzz, gp_post['q_u_sigma']) @ jnp.linalg.solve(Kzz, Kxz.T), axis1=-2, axis2=-1)
        return f_var

    # --------- Closed-form expectations wrt q(f) and q(x) ----------
    def f(self, drift_params: Dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, gp_post: GPPost) -> jnp.array:
        M, D = self.zs.shape
        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)
        Kzz_inv = jnp.linalg.inv(Kzz)
        E_Kxz = vmap(partial(self.kernel.E_Kxz, self.expectation, key, m=m, S=S, kernel_params=drift_params))(self.zs)[None] # (1, n_inducing)
        E_f = E_Kxz @ Kzz_inv @ gp_post['q_u_mu'].T
        return E_f[0]

    def ff(self, drift_params: Dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, gp_post: GPPost) -> jnp.array:
        keys = jr.split(key, 2)
        M, D = self.zs.shape
        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)
        Kzz_inv = jnp.linalg.inv(Kzz)
        E_KzxKxz = vmap(vmap(partial(self.kernel.E_KzxKxz, self.expectation, keys[0], m=m, S=S, kernel_params=drift_params), (None, 0)), (0, None))(self.zs, self.zs) # (n_inducing, n_inducing)

        term1 = D * (self.kernel.E_Kxx(self.expectation, keys[1], m, S, drift_params) - jnp.trace(Kzz_inv @ E_KzxKxz))
        term2 = jnp.trace(Kzz_inv @ gp_post['q_u_sigma'].sum(0) @ Kzz_inv @ E_KzxKxz)
        term3 = jnp.trace(E_KzxKxz @ Kzz_inv @ gp_post['q_u_mu'].T @ gp_post['q_u_mu'] @ Kzz_inv)
        return term1 + term2 + term3

    def dfdx(self, drift_params: Dict[str, Any], key: jr.PRNGKey, t: jnp.array, m: jnp.array, S: jnp.array, gp_post: GPPost) -> jnp.array:
        M, D = self.zs.shape
        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)
        Kzz_inv = jnp.linalg.inv(Kzz)
        E_dKzxdx = vmap(partial(self.kernel.E_dKzxdx, self.expectation, key, m=m, S=S, kernel_params=drift_params))(self.zs)
        return gp_post['q_u_mu'] @ Kzz_inv @ E_dKzxdx # (D, D)

    def update_dynamics_params(self, key: jr.PRNGKey, t_grid: jnp.array, marginal_params: MarginalParams, trial_mask: jnp.array, drift_params: dict[str, Any], inputs: jnp.array, input_effect: jnp.array, sigma: int = 1.) -> GPPost:
        del_t = t_grid[1:] - t_grid[:-1] # (T-1)
        ms, Ss, SSs = marginal_params['m'], marginal_params['S'], marginal_params['SS']
        batch_size, T, D = ms.shape
        M = self.zs.shape[0]
        trans_mask = trial_mask[..., :-1] & trial_mask[..., 1:]

        keys = jr.split(key, 3)
        def _q_u_sigma_helper(ms, Ss, mask, del_t, kernel_params):
            def step(mask_t, m_t, S_t):
                def compute(_):
                    return vmap(vmap(partial(self.kernel.E_KzxKxz, self.expectation, keys[0], kernel_params=kernel_params), (None, 0, None, None)), (0, None, None, None))(self.zs, self.zs, m_t, S_t)
                def skip(_):
                    return jnp.zeros((M, M))
                return jax.lax.cond(mask_t, compute, skip, operand=None)

            E_KzxKxz_on_grid = jax.vmap(step)(mask, ms[:-1], Ss[:-1])  # (T-1, M, M)
            int_E_KzxKxz = (del_t[:, None, None] * E_KzxKxz_on_grid).sum(0)
            return int_E_KzxKxz  # (M, M)

        def _q_u_mu_helper1(ms, Ss, inputs, mask, del_t, B, kernel_params):
            ms_diff = ms[1:] - ms[:-1]  # (T-1, D)
            def step(mask_t, m_t, S_t):
                def compute(_):
                    return vmap(partial(self.kernel.E_Kxz, self.expectation, keys[1], kernel_params=kernel_params), (0, None, None))(self.zs, m_t, S_t)
                def skip(_):
                    return jnp.zeros((M))
                return jax.lax.cond(mask_t, compute, skip, operand=None)

            E_Kxz_on_grid = jax.vmap(step)(mask, ms[:-1], Ss[:-1])  # (T-1, M)
            input_correction = (B[None] @ inputs[..., None]).squeeze(-1)  # (T, D)
            int_E_Kzx_ms_diff = (vmap(jnp.outer)(E_Kxz_on_grid, ms_diff - del_t[:, None] * input_correction[:-1])).sum(0)
            return int_E_Kzx_ms_diff

        def _q_u_mu_helper2(ms, Ss, SSs, mask, kernel_params):
            Ss_diff = (SSs - Ss[:-1])  # (T-1, D, D)
            def step(mask_t, m_t, S_t):
                def compute(_):
                    return vmap(partial(self.kernel.E_dKzxdx, self.expectation, keys[2], kernel_params=kernel_params), (0, None, None))(self.zs, m_t, S_t)
                def skip(_):
                    return jnp.zeros((M, D))
                return jax.lax.cond(mask_t, compute, skip, operand=None)

            E_dKzxdx_on_grid = jax.vmap(step)(mask, ms[:-1], Ss[:-1])  # (T-1, M, D)
            int_E_dKzxdx_Ss_diff = (E_dKzxdx_on_grid @ Ss_diff).sum(0)
            return int_E_dKzxdx_Ss_diff  # (M, D)

        Kzz = make_gram(self.kernel.K, drift_params, self.zs, self.zs, jitter=self.jitter)        
        int_E_KzxKxz = vmap(partial(_q_u_sigma_helper, del_t=del_t, kernel_params=drift_params))(ms, Ss, trans_mask).sum(0) # vmap over batches
        q_u_sigma = Kzz @ jnp.linalg.solve(Kzz + (1/(sigma**2)) * int_E_KzxKxz, Kzz)
        q_u_sigma = q_u_sigma[None].repeat(ms.shape[-1], 0) # (D, n_inducing, n_inducing)

        int1 = vmap(partial(_q_u_mu_helper1, del_t=del_t, B=input_effect, kernel_params=drift_params))(ms, Ss, inputs, trans_mask).sum(0) # vmap over batches
        int2 = vmap(partial(_q_u_mu_helper2, kernel_params=drift_params))(ms, Ss, SSs, trans_mask).sum(0) # vmap over batches
        q_u_mu = (1/(sigma)**2) * (Kzz @ jnp.linalg.solve(Kzz + (1/(sigma**2)) * int_E_KzxKxz, int1 + int2)).T

        gp_post = {'q_u_mu': q_u_mu, 'q_u_sigma': q_u_sigma}
        return gp_post