import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from functools import partial

from sing.expectation import GaussHermiteQuadrature
from sing.utils.params import *
from sing.utils.general_helpers import sgd

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

class Likelihood(ABC):
    """
    Represents the likelihood model
    """
    def __init__(self, ys_obs: jnp.array, t_mask: jnp.array) -> None:
        """
        Params:
        ------------
        ys_obs: a shape (n_trials, T, N) array, the observations of the model
        t_mask: a shape (n_trials, T) array, a binary mask indicating whether for trial i, there exists an observation at time \tau_j
        """
        self.ys_obs = ys_obs
        self.t_mask = t_mask

    @abstractmethod
    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params: Dict[str, jnp.array]) -> float:
        """
        Computes the expected log likelihood
        E_{q(x_i)}[log p(y_i | x_i)]
        under a Gaussian distribution q(x_i)

        Params:
        ------------
        y: a shape (N) array, an observation
        mean: a shape (D) array, the Gaussian mean
        var: a shape (D, D) array, the Gaussian mean
        output_params: a dictionary representing the parameters of the likelihood

        Returns:
        ------------
        exp_ll: the expected log likelihood
        """
        raise NotImplementedError

    def grad_ell(self, mean_params: MeanParams, y: jnp.array, output_params: Dict[str, jnp.array]) -> Tuple[jnp.array, jnp.array]:
        """
        Params:
        ---------
        mean_params: mean parameters of the variational posterior over x
        y: a shape (D) array, the observation y(tau_i) at a single dimension
        output_params: a dictionary representing the parameters of the likelihood, containing
            - C: a shape (D) array, a single row of the emissions matrix C
            - d: a shape (1) array, a single component of the emissions offset vector d
            where E[y(tau_i)|x_i] = inverse_link(Cx_i + d)

        Returns:
        ---------
        grad_mu1: (D, ) gradient wrt mu1
        grad_mu2: (D, D) gradient wrt mu2

        NOTE:
        -----
        This assumes the inverse link function is applied component-wise and output dimensions are conditionally independent given x
        """
        def _grad_ell_dim(mean_params, y, output_params):
            c, d = output_params["C"], output_params["d"]
            Ex, ExxT = mean_params["Ex"], mean_params["ExxT"]

            mean = jnp.dot(c, Ex) + d
            var = jnp.dot(c, (ExxT - jnp.outer(Ex, Ex)) @ c)

            alpha, beta = jax.grad(self.ell, argnums=(1, 2))(y, mean, var, output_params)
            grad_mu1 = (alpha - 2.0 * beta * jnp.dot(c, Ex)) * c
            grad_mu2 = beta * jnp.outer(c, c)
            return grad_mu1, grad_mu2
    
        grad_mu1s, grad_mu2s = vmap(lambda y_i, output_params_i: _grad_ell_dim(mean_params, y_i, output_params_i))(y, output_params)
        return grad_mu1s.sum(axis=0), grad_mu2s.sum(axis=0)
    
    def ell_over_obs_dims(self, y: jnp.array, marginal_params: MarginalParams, output_params: Dict[str, jnp.array]) -> float:
        """
        Compute the expected log-likelihood across all output dimensions
        """
        C, d = output_params['C'], output_params['d']
        m, S = marginal_params['m'], marginal_params['S']
        means = C @ m + d # (N)
        vars = vmap(lambda c: jnp.dot(c, S @ c))(C) # (N)
        return vmap(self.ell)(y, means, vars, output_params).sum()

    def ell_over_time(self, ys: jnp.array, marginal_params: MarginalParams, t_mask: jnp.array, output_params: Dict[str, jnp.array]) -> float:
        """
        Compute the expected log-likelihood across all output dimensions and across the time grid tau, taking into account masking observations
        sum_{i=0}^T E_{q(x_i)}[log p(y(tau_i) | x_i)] delta(tau_i)
        where delta(tau_i) =1 if there is an observation at time i and =0 if not
        """
        ell_on_grid = vmap(partial(self.ell_over_obs_dims, output_params=output_params))(ys, marginal_params)
        return (ell_on_grid * t_mask).sum()

    def update_output_params(self, marginal_params: MarginalParams, key: jr.PRNGKey, output_params: Dict[str, jnp.array], loss_fn: Callable[[jr.PRNGKey, jnp.array], float], n_iters_m: int = 200, learning_rate: int = .08, learning_rate_final: Optional[float] = None, max_grad_norm: float = 5.0, **kwargs) -> Dict[str, jnp.array]:
        """
        Updates the output parameters during the M-step of the variational expectation-maximization (vEM) algorithm
        By default, learn output parameters with SGD

        Params:
        ---------
        marginal_params
        output_params
        loss_fn: a function with signature loss_fun(output_params) -> scalar to be maximized
        for vEM, we choose the loss function to be the negative ELBO
        n_iters_m: number of M-steps to be performed
        learning_rate: learning rate
        learning_rate_final: optional final learning rate for exponential decay
        max_grad_norm: maximum global norm for gradient clipping

        Returns:
        ---------
        output_params_new: the output parameters that approximately maximize the ELBO at the current set of variational parameters
        """
        output_params_new, _ = sgd(key, loss_fn, output_params, n_iters=n_iters_m, learning_rate=learning_rate, learning_rate_final=learning_rate_final, max_grad_norm=max_grad_norm)
        return output_params_new

class Gaussian(Likelihood):
    """
    A Gaussian likelihood with covariance R, y ~ N(Cx + d, R)
    Supports:
      - R.shape == (N,)   : diagonal covariance
      - R.shape == (N,N)  : full covariance
    """
    def __init__(self, ys_obs: jnp.array, t_mask: jnp.array) -> None:
        super().__init__(ys_obs, t_mask)

    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params: Dict[str, jnp.array]) -> float:
        r = output_params["R"]

        if r.ndim == 0:
            # Diagonal scalar case: called through base-class factorized machinery
            ll = tfd.Normal(mean, jnp.sqrt(r)).log_prob(y)
            correction = -0.5 * var / r
            return ll + correction

        elif r.ndim == 2:
            # Full covariance case: y, mean are vectors and var is full obs covariance
            ll = tfd.MultivariateNormalFullCovariance(
                loc=mean,
                covariance_matrix=r,
            ).log_prob(y)
            correction = -0.5 * jnp.trace(jnp.linalg.solve(r, var))
            return ll + correction
        else:
            raise ValueError(f"R must have ndim 0 or 2 here, got shape {r.shape}")

    def grad_ell(self, mean_params: MeanParams, y: jnp.array, output_params: Dict[str, jnp.array]) -> Tuple[jnp.array, jnp.array]:
        """
        For diagonal R, use the factorized base-class implementation
        For full R, compute the gradient jointly across output dimensions
        """
        r = output_params["R"]

        if r.ndim == 1:
            return super().grad_ell(mean_params, y, output_params)
        elif r.ndim == 2:
            Ex, ExxT = mean_params["Ex"], mean_params["ExxT"]
            C, d = output_params["C"], output_params["d"]

            def ell_wrapped(mu1, mu2): # NOTE: manually computes the mean -> marginal conversion
                S = mu2 - jnp.outer(mu1, mu1)
                mean = C @ mu1 + d
                obs_cov = C @ S @ C.T
                return self.ell(y, mean, obs_cov, output_params)
            grad_mu1, grad_mu2 = jax.grad(ell_wrapped, argnums=(0, 1))(Ex, ExxT)
            return grad_mu1, grad_mu2
        else:
            raise ValueError(f"R must have ndim 1 or 2, got shape {r.shape}")

    def ell_over_obs_dims(self, y: jnp.array, marginal_params: MarginalParams, output_params: Dict[str, jnp.array]) -> float:
        """
        For diagonal R, use the factorized base-class implementation
        For full R, compute the joint Gaussian expected log-likelihood
        """
        r = output_params["R"]

        if r.ndim == 1:
            return super().ell_over_obs_dims(y, marginal_params, output_params)
        elif r.ndim == 2:
            C, d = output_params["C"], output_params["d"]
            m, S = marginal_params["m"], marginal_params["S"]
            mean = C @ m + d
            obs_cov = C @ S @ C.T
            return self.ell(y, mean, obs_cov, output_params)
        else:
            raise ValueError(f"R must have ndim 1 or 2, got shape {r.shape}")

    def update_output_params(self, marginal_params: MarginalParams, ys: jnp.array, t_mask: jnp.array, **kwargs,) -> Dict[str, jnp.array]:
        """
       Perform closed-form updates for output mapping parameters for Gaussian likelihood model according to Hu et al., 2025

        NOTE:
        -----
        This implementation is only for diagonal R
        """
        output_params_prev = kwargs.get("output_params", None)
        if output_params_prev is not None and output_params_prev["R"].ndim != 1:
            return output_params_prev

        ms, Ss = marginal_params["m"], marginal_params["S"]
        D = ms.shape[-1]  # latent dimension
        N = ys.shape[-1]  # output dimension
        n_total_obs = t_mask.sum()  # number of total observations

        # Stack ys across trials
        ys_obs_stacked = ys.reshape(-1, N)
        t_mask_stacked = t_mask.reshape(-1)
        ms_stacked, Ss_stacked = ms.reshape(-1, D), Ss.reshape(-1, D, D)

        # Compute closed-form update for C
        ybar = jnp.sum(t_mask_stacked[:, None] * ys_obs_stacked, axis=0) / n_total_obs
        mbar = jnp.sum(t_mask_stacked[:, None] * ms_stacked, axis=0) / n_total_obs

        C_term1_on_grid = vmap(jnp.outer)(
            ys_obs_stacked - ybar[None, :],
            ms_stacked - mbar[None, :],
        )  # (-1, N, D)
        C_term1 = (t_mask_stacked[:, None, None] * C_term1_on_grid).sum(axis=0)

        C_term2_on_grid = Ss_stacked + vmap(jnp.outer)(
            ms_stacked - mbar[None, :],
            ms_stacked - mbar[None, :],
        )  # (-1, D, D)
        C_term2 = (t_mask_stacked[:, None, None] * C_term2_on_grid).sum(axis=0)

        C = jnp.linalg.solve(C_term2, C_term1.T).T  # (N, D)

        # Update for d
        d_term1_on_grid = ys_obs_stacked - (C @ ms_stacked.T).T  # (-1, N)
        d_term1 = (t_mask_stacked[:, None] * d_term1_on_grid).sum(axis=0)
        d = (1.0 / n_total_obs) * d_term1

        # Update for diagonal R only
        all_mus = (C @ ms_stacked[..., None]).squeeze(-1) + d  # (-1, N)
        all_vars = vmap(jnp.diag)(C @ Ss_stacked @ C.T)  # (-1, N)
        R_term1 = (t_mask_stacked[:, None] * ys_obs_stacked**2).sum(axis=0)
        R_term2 = -2 * (t_mask_stacked[:, None] * ys_obs_stacked * all_mus).sum(axis=0)
        R_term3 = (t_mask_stacked[:, None] * (all_vars + all_mus**2)).sum(axis=0)
        R = (1.0 / n_total_obs) * (R_term1 + R_term2 + R_term3)

        output_params_new = {"C": C, "d": d, "R": R}
        return output_params_new

class NonlinearGaussian(Likelihood):
    """
    A Gaussian likelihood with diagonal covariance and non-identity inverse link
    i.e. p(y | x) = N(y| inv_link(Cx + d), R)
    where inv_link is applied element-wise
    """

    def __init__(self, ys_obs: jnp.array, t_mask: jnp.array, link: Callable[[float], float], n_quad: int = 100) -> None:
        """
        Params:
        --------------
        ys_obs
        t_mask
        link: inverse link function
        n_quad: number of 1D quadrature points used to compute the expected log likelihood along each dimension
        (since these cannot be computed in closed-form)
        """
        super().__init__(ys_obs, t_mask)
        self.link = link
        self.quadrature = GaussHermiteQuadrature(1, n_quad=n_quad)

    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params: Dict[str, jnp.array]) -> float:
        r = output_params['R']
        fn = lambda u: tfd.Normal(self.link(u), jnp.sqrt(r)).log_prob(y)
        return self.quadrature.gaussian_int(fn, mean.reshape(1), var.reshape((1, 1))).squeeze(-1)

class Poisson(Likelihood):
    """
    Poisson likelihood class, y ~ Pois(g(Cx + d)*dt), where g is a pre-specified inverse link function
    """
    def __init__(self, ys_obs: jnp.array, t_mask: jnp.array, dt: float, link: Optional[Callable[[float], float]] = None, include_dt: bool = False, n_quad: int = 20) -> None:
        """
        Params:
        --------------
        ys_obs
        t_mask
        dt: discretization time-step of data
        link: inverse link function
        - If None, defaults to 'exp' canonical link, which has a closed-form expectation
        - If specified, uses quadrature to approximate expectation
        include_dt: bool, whether to scale the Poisson rate by dt
        """
        super().__init__(ys_obs, t_mask)
        self.dt = dt
        self.link = link
        self.include_dt = include_dt
        self.quadrature = GaussHermiteQuadrature(1, n_quad=n_quad)
    
    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params = None) -> float:
        """
        Compute expectation E[log Pois(y|inv_link(u))] wrt q(u) = N(u|mean, var).
        Here, we can view u = c^Tx + d, so mean = c^Tm + d and var = c^TSc
        """
        if self.link is None: # Denotes 'exp' link
            cov_term = 0.5 * var
            log_rate = mean + cov_term
            if self.include_dt:
                log_rate += jnp.log(self.dt)
            ll = tfd.Poisson(log_rate=log_rate).log_prob(y) # scalar
            correction = -y * cov_term
            return ll + correction
        else: # Denotes non-canonical link
            # u = c^Tx + d
            # q(x) = N(m, S)
            # q(u) = N(c^Tm + d, c^TSc) = N(mean, var)
            if self.include_dt:
                fn = lambda u: tfd.Poisson(rate=self.dt * self.link(u)).log_prob(y)
            else:
                fn = lambda u: tfd.Poisson(rate=self.link(u)).log_prob(y)
            return self.quadrature.gaussian_int(fn, mean.reshape(1), var.reshape((1, 1))).squeeze(-1)

class PoissonProcess(Likelihood):
    """
    Poisson process likelihood class, t_i|x ~ PP(g(Cx+d))
    """
    def __init__(self, ys_obs: jnp.array, t_mask: jnp.array, dt: float, link: Optional[Callable[[float], float]] = None, n_quad: int = 20) -> None:
        super().__init__(ys_obs, t_mask)
        self.dt = dt
        self.link = link
        self.quadrature = GaussHermiteQuadrature(1, n_quad=n_quad)

    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params = None) -> float:
        # for Poisson Process likelihood, each time step gets a "continuous" part (from discretized integral) and "jump" part (if there is an observation at that time step)
        if self.link is None: # Denotes 'exp' link
            ell_cont = -self.dt * jnp.exp(mean + 0.5 * var)
            ell_jump = mean
        else: # Denotes non-canonical link
            fn_cont = lambda u: -self.link(u)
            ell_cont = self.dt * self.quadrature.gaussian_int(fn_cont, mean.reshape(1), var.reshape((1, 1))).squeeze(-1)
            fn_jump = lambda u: jnp.log(self.link(u))
            ell_jump = self.quadrature.gaussian_int(fn_jump, mean.reshape(1), var.reshape((1, 1))).squeeze(-1)

        return ell_cont + (y > 0) * ell_jump

class GeneralizedPoisson(Likelihood):
    """
    In this model, y_n ~ Pois(f_n(x)), n = 1, ..., N
    NOTE: this differs from above because each observation dimension has a different rate function f_n

    This is used for the synthetic place cell model in the demo notebooks and in the paper Hu et al., 2025
    """
    def __init__(self, latent_dim: int, ys_obs: jnp.array, t_mask: jnp.array, link: Callable[[float, int], float], n_quad: int = 5) -> None:
        """
        Params:
        --------------
        link: inverse link function with signature link(x, [obs_dim]) -> rate
        where obs_dim is an integer denoting the observation dimension
        quad: a D-dimensional Gaussian quadrature object 
        """
        super().__init__(ys_obs, t_mask)
        self.link = link
        # NOTE: Expectations approximated over D dimensions due to different rate function per obs dimension
        self.quadrature = GaussHermiteQuadrature(latent_dim, n_quad=n_quad) 
        
    def ell(self, y: jnp.array, mean: jnp.array, var: jnp.array, output_params: Dict[str, jnp.array]) -> float:
        idx = output_params['obs_idx'].astype(int)
        fn = lambda u: tfd.Poisson(rate=self.link(u, idx)).log_prob(y)
        return self.quadrature.gaussian_int(fn, mean, var)
        
    def grad_ell(self, mean_params: MeanParams, y: jnp.array, output_params: Dict[str, jnp.array]) -> Tuple[jnp.array, jnp.array]:
        Ex, ExxT = mean_params['Ex'], mean_params['ExxT']

        # Compute gradients directly i.e. without the chain rule
        ell_wrapped = lambda mu1, mu2: self.ell(y, mu1, mu2 - jnp.outer(mu1, mu1), output_params)
        grad_mu1, grad_mu2 = jax.grad(ell_wrapped, argnums=(0, 1))(Ex, ExxT)

        return grad_mu1, grad_mu2
    
    def ell_over_obs_dims(self, y: jnp.array, marginal_params: MarginalParams, output_params: Dict[str, jnp.array]) -> float:
        return vmap(lambda y, output_params: self.ell(y, marginal_params['m'], marginal_params['S'], output_params))(y, output_params).sum()