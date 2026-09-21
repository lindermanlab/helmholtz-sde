import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from functools import partial

from sing.expectation import Expectation

from abc import ABC, abstractmethod 
from typing import Any, Callable, Dict, Union

class Kernel(ABC):
    """
    Abstract base class representing the positive semi-definite kernel of a Gaussian process (GP)
    """

    def __init__(self, latent_dim: int = 1) -> None:
        self.latent_dim = latent_dim

    @abstractmethod
    def K(self, x1: jnp.array, x2: jnp.array, kernel_params: Dict[str, Any]) -> float:
        """
        Evaluates the kernel at inputs x1, x2
        NOTE: K(x1, x2) = K(x2, x1) so the ordering of x1 and x2 doesn't matter

        Params:
        -------------
        x1, x2: shape (D) arrays, the input points at which to evaluate the kernel
        kernel: a dictionary containing the parameters of the kernel
        """
        return NotImplementedError

    # --------- Expectations computed with quadrature or Monte Carlo ----------
    def E_Kxx(self, expectation: Expectation, key: jr.PRNGKey, m: jnp.array, S: jnp.array, kernel_params: Dict[str, Any]) -> float:
        """Computes E[k(x,x)] wrt q(x) = N(x|m,S)"""
        fn = lambda x: self.K(x, x, kernel_params)
        return expectation.gaussian_int(key=key, fn=fn, m=m, S=S) # scalar

    def E_Kxz(self, expectation: Expectation, key: jr.PRNGKey, z: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, Any]) -> float:
        """Computes E[k(x,z)] wrt q(x) = N(x|m,S)"""
        fn = lambda x: self.K(x, z, kernel_params)
        return expectation.gaussian_int(key=key, fn=fn, m=m, S=S) # scalar

    def E_KzxKxz(self, expectation: Expectation, key: jr.PRNGKey, z1: jnp.array, z2: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, Any]) -> float:
        """Computes E[k(z1,x)k(x,z2)] wrt q(x) = N(x|m,S)"""
        fn = lambda x: self.K(z1, x, kernel_params) * self.K(x, z2, kernel_params)
        return expectation.gaussian_int(key=key, fn=fn, m=m, S=S) # scalar

    def E_dKzxdx(self, expectation: Expectation, key: jr.PRNGKey, z: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, Any]) -> jnp.array:
        """Computes E[dk(z,x)/dx] wrt q(x) = N(x|m,S)."""
        fn = grad(partial(self.K, z, kernel_params=kernel_params))
        return expectation.gaussian_int(key=key, fn=fn, m=m, S=S) # (D)

class RBF(Kernel):
    """
    The RBF kernel
    """
    def __init__(self, latent_dim: int = 1) -> None:
        super().__init__(latent_dim)

    def K(self, x1: jnp.array, x2: jnp.array, kernel_params: Dict[str, jnp.array]) -> float:
        """
        Params:
        -------------
        x1, x2
        kernel_params: dictionary containing
        - length_scales: (D) length scale for each state dimension
        - output_scale: scalar
        """
        sq_diffs = (((x1 - x2) / kernel_params["length_scales"])**2).sum()
        return kernel_params["output_scale"]**2 * jnp.exp(-0.5 * sq_diffs) # scalar

    # --------- Closed-form expectations ----------
    def E_Kxx(self, expectation: Expectation, key: jr.PRNGKey, m: jnp.array, S: jnp.array, kernel_params: Dict[str, jnp.array]) -> float:
        return kernel_params["output_scale"]**2 # scalar

    def E_Kxz(self, expectation: Expectation, key: jr.PRNGKey, z: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, jnp.array], **kwargs) -> float:
        K = len(m)
        integral = tfd.MultivariateNormalFullCovariance(m, S + jnp.diag(kernel_params["length_scales"]**2)).prob(z)
        const = kernel_params["output_scale"]**2 * jnp.sqrt((2 * jnp.pi)**K) * kernel_params["length_scales"].prod()
        return const * integral

    def E_KzxKxz(self, expectation: Expectation, key: jr.PRNGKey, z1: jnp.array, z2: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, jnp.array], **kwargs) -> float:
        K = len(m)
        squared_length_scales = kernel_params["length_scales"]**2

        S_inv = jnp.linalg.solve(S, jnp.eye(K))
        L_inv = jnp.diag(1. / squared_length_scales)
        linear_term = jnp.linalg.solve(S, m) + (z2 / squared_length_scales)
        new_mean = jnp.linalg.solve(S_inv + L_inv, linear_term)
        new_cov = jnp.linalg.solve(S_inv + L_inv, jnp.eye(K)) + jnp.diag(squared_length_scales)

        const = kernel_params["output_scale"]**4 * (2 * jnp.pi)**K * (squared_length_scales).prod()
        prob1 = tfd.MultivariateNormalFullCovariance(m, S + jnp.diag(squared_length_scales)).prob(z2)
        prob2 = tfd.MultivariateNormalFullCovariance(new_mean, new_cov).prob(z1)

        return const * prob1 * prob2

    def E_dKzxdx(self, expectation: Expectation, key: jr.PRNGKey, z: jnp.array, m: jnp.array, S: jnp.array, kernel_params: Dict[str, jnp.array], **kwargs) -> jnp.array:
        Psi1 = self.E_Kxz(expectation, key, z, m, S, kernel_params)
        L = jnp.diag(kernel_params["length_scales"]**2)
        return Psi1 * jnp.linalg.solve(L + S, z - m) # (D, )

# --------------------- see Hu et al., 2024 for discussion of linear kernels -------------------
class SimpleLinear(Kernel):
    """
    The linear kernel
    
    Trajectories drawn from the GP with linear kernel will be linear functions
    """
    def __init__(self, noise_var: float =  1., latent_dim: int = 1) -> None:
        super().__init__(latent_dim)
        self.noise_var = noise_var

    def K(self, x1: jnp.array, x2: jnp.array, kernel_params = None) -> float:
        """Linear kernel with M = I, c = 0"""
        return (x1 * x2).sum() + self.noise_var

class Linear(Kernel):
    """
    The linear kernel with a fixed point parameter
   
    The fixed point defines the x values at which a zero of the linear function is most likely to occur
    """
    def __init__(self, noise_var: float = 1., latent_dim: int = 1) -> None:
        super().__init__(latent_dim)
        self.noise_var = noise_var

    def K(self, x1: jnp.array, x2: jnp.array, kernel_params: Dict[str, jnp.array]) -> float:
        """
        Params:
        -------------
        x1, x2
        kernel_params: a dictionary containing
            - fixed_point: a shape (D) array, the fixed point of the linear kernel
        """
        c = kernel_params["fixed_point"]
        M = self.noise_var * jnp.ones(len(x1))
        return (M * (x1 - c) * (x2 - c)).sum() + self.noise_var

class FullLinear(Kernel):
    """
    The linear kernel with a fixed point parameter and non-identity slope variance
   
    NOTE: assumes slope variance is diagonal
    """
    def __init__(self, latent_dim: int = 1) -> None:
        super().__init__(latent_dim)

    def K(self, x1: jnp.array, x2: jnp.array, kernel_params: Dict[str, jnp.array]) -> float:
        """
        Params:
        -------------
        x1, x2
        kernel_params: a dictionary containing
            - fixed_point: a shape (D) array, the fixed point of the linear kernel
            - log_M: a shape (D) array, the log diagonal of M
            - log_noise_var: a scalar, the log of noise_var
        """
        c = kernel_params["fixed_point"]
        M = jnp.exp(kernel_params["log_M"]) # (D, )
        noise_var = jnp.exp(kernel_params["log_noise_var"])
        return (M * (x1 - c) * (x2 - c)).sum() + noise_var

class SSL(Kernel):
    """
    The smoothly switching linear kernel introduced by Hu et al., 2024
    """
    def __init__(self, linear_kernel: Union[SimpleLinear, Linear, FullLinear], basis_set: Callable[[jnp.array], jnp.array], latent_dim: int = 1) -> None:
        """
        Params:
        -------------
        linear_kernel: the linear kernel from which the smoothly switching linear kernel is defined
        basis_set: a function with signature basis_set(x) -> R^{num_bases}, where num_bases is a set of basis functions that determines the boundary between states
        latent_dim
        """
        super().__init__(latent_dim)
        self.linear_kernel = linear_kernel
        self.basis_set = basis_set

    def construct_partition(self, x: jnp.array, W: jnp.array, log_tau: float) -> jnp.array:
        """
        Construct partition function pi at a given latent space location x
        """
        activations = W.T @ self.basis_set(x)
        pi = tfb.SoftmaxCentered().forward(activations / jnp.exp(log_tau))
        return pi

    def K(self, x1: jnp.array, x2: jnp.array, kernel_params: Dict[str, jnp.array]) -> float:
        """
        Compute smoothly switching linear (SSL) kernel

        Params:
        --------------
        x1, x2: shape (D) arrays, input locations
        kernel_params: dictionary containing
        - linear_params: list of length num_states, where each entry is a dict containing linear kernel params
        - W: a shape (num_bases, num_states-1) array, partition function basis weights
        - log_tau: scalar, partition function smoothing parameter
        """
        linear_params = kernel_params["linear_params"] # list of linear params, one dict per regime
        W = kernel_params["W"]
        log_tau = kernel_params["log_tau"]
        pi_x1 = self.construct_partition(x1, W, log_tau)
        pi_x2 = self.construct_partition(x2, W, log_tau)
        linear_kernels = jnp.array([self.linear_kernel.K(x1, x2, param) for param in linear_params]) # (num_states)
        return (pi_x1 * pi_x2 * linear_kernels).sum()