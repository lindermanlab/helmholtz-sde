"""
Tests for the Helmholtz corrections in helmholtz_sde.helmholtz
"""
import jax
jax.config.update("jax_enable_x64", True)

import dataclasses
import functools
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from helmholtz_sde.helmholtz.correction import HelmCorrection, compute_kl_helmholtz_approx, make_corrected_drift
from helmholtz_sde.utils.hermite import (HermiteCoeffs, hermite_products, hermite_to_monomial,
                                        index_tables, monomial_to_hermite, whiten_frame)
from helmholtz_sde.utils.quadrature import gauss_hermite_nodes
from helmholtz_sde.helmholtz.taylor import TaylorCorrection
from helmholtz_sde.helmholtz.subspace import LeastSquaresCorrection, DEFAULT_CORRECTION
from helmholtz_sde.utils.general_helpers import symmetrize_full, sym_sqrt_and_invsqrt


# --------------------- Helpers ---------------------
def _random_spd(key: jr.PRNGKey, K: int) -> jnp.array:
    A = jr.normal(key, (K, K), dtype=jnp.float64)
    return A @ A.T + jnp.eye(K, dtype=jnp.float64)


def _make_polynomial_field(key: jr.PRNGKey, K: int, degree: int):
    """
    Build fp, fq such that r = fp - fq is a random polynomial of total degree <= degree
    """
    keys = jr.split(key, degree + 2)
    coeffs = [0.5 * jr.normal(keys[d], (K,) * (d + 1), dtype=jnp.float64) for d in range(degree + 1)]

    def r_of_x(x):
        out = jnp.zeros((K,), dtype=x.dtype)
        for d, C in enumerate(coeffs):
            term = C
            for _ in range(d):
                term = term @ x
            out = out + term
        return out

    J0 = jr.normal(keys[-1], (K, K), dtype=jnp.float64)

    def fq(x):
        return J0 @ x

    def fp(x):
        return fq(x) + r_of_x(x)

    return fp, fq, r_of_x


def _make_smooth_field(key: jr.PRNGKey, K: int):
    """
    Build fp, fq such that r = fp - fq is a smooth non-polynomial field
    """
    k1, k2, k3, k4 = jr.split(key, 4)
    W1 = jr.normal(k1, (8, K), dtype=jnp.float64)
    b1 = jr.normal(k2, (8,), dtype=jnp.float64)
    W2 = jr.normal(k3, (K, 8), dtype=jnp.float64)
    J0 = jr.normal(k4, (K, K), dtype=jnp.float64)

    def fq(x):
        return J0 @ x

    def fp(x):
        return fq(x) + W2 @ jnp.tanh(W1 @ x + b1)

    return fp, fq


def _setup(seed: int, K: int):
    key = jr.PRNGKey(seed)
    kx, km, ks, kr = jr.split(key, 4)
    x = jr.normal(kx, (K,), dtype=jnp.float64)
    m = jr.normal(km, (K,), dtype=jnp.float64)
    S = _random_spd(ks, K)
    return x, m, S, kr


def _q_divergence(h, x, m, S):
    """
    div(q h) / q for q = N(m, S), which equals div h - h^T S^{-1} (x - m)
    """
    J = jax.jacfwd(h)(x)
    return jnp.trace(J) - h(x) @ jnp.linalg.solve(S, x - m)


def _eval_monomial(P, x):
    """
    Evaluate a polynomial field from its highest-degree-first monomial tensors
    """
    ell = len(P) - 1
    out = jnp.zeros_like(x)
    for i, C in enumerate(P):
        term = C
        for _ in range(ell - i):
            term = term @ x
        out = out + term
    return out


def _id(correction):
    """
    Compact test id for a correction, e.g. LeastSquaresCorrection(ell=1,kappa=1,n_nodes=3)
    """
    if correction is None:
        return "none"
    fields = {k: v for k, v in dataclasses.asdict(correction).items() if k != "jitter" and v is not None}
    if fields.get("n_nodes") is not None:
        fields.pop("n_mc", None) # quadrature ignores n_mc
    return type(correction).__name__ + "(" + ",".join(f"{k}={v}" for k, v in fields.items()) + ")"


TAYLOR = [TaylorCorrection(ell=ell) for ell in (1, 2, 3)]
LEAST_SQUARES = [LeastSquaresCorrection(ell=ell, kappa=kappa, n_nodes=3) for ell in (1, 2) for kappa in (0, 1)]


# --------------------- Divergence-free constraint ---------------------
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("correction", TAYLOR + LEAST_SQUARES, ids=_id)
def test_correction_is_q_divergence_free(correction, K):
    """
    Every correction satisfies div(q h) = 0 for the Gaussian marginal q, even for non-polynomial residuals
    """
    x, m, S, kr = _setup(1, K)
    fp, fq = _make_smooth_field(kr, K)
    coeffs = correction.fit(m, S, fp, fq)
    h = lambda x_: correction.evaluate(coeffs, x_)
    for i in range(3):
        xi = m + 0.7 * (i + 1) * (x - m)
        np.testing.assert_allclose(_q_divergence(h, xi, m, S), 0.0, atol=1e-6, err_msg=f"{_id(correction)} at point {i}")


# --------------------- Taylor vs least-squares on polynomial residuals ---------------------
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("kappa", [0, 1, 2])
@pytest.mark.parametrize("ell", [1, 2, 3])
def test_least_squares_quadrature_matches_taylor_on_polynomials(ell, kappa, K):
    """
    For a polynomial residual of degree <= ell the Taylor expansion is exact and the least-squares projection with exact
    quadrature recovers the same divergence-free part, so the two corrections coincide (kappa > ell is clipped to ell)
    """
    x, m, S, kr = _setup(2 + ell, K)
    fp, fq, _ = _make_polynomial_field(kr, K, degree=ell)
    taylor = TaylorCorrection(ell=ell)
    least_squares = LeastSquaresCorrection(ell=ell, kappa=kappa, n_nodes=ell + 2)
    h_t = taylor.evaluate(taylor.fit(m, S, fp, fq), x)
    h_g = least_squares.evaluate(least_squares.fit(m, S, fp, fq), x)
    np.testing.assert_allclose(h_g, h_t, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("correction", [TaylorCorrection(ell=2), LeastSquaresCorrection(ell=2, kappa=1, n_nodes=4)], ids=_id)
def test_residual_minus_correction_is_gradient(correction):
    """
    For a polynomial residual of degree <= ell, r - h is a gradient field, so its Jacobian is symmetric
    """
    K = 3
    x, m, S, kr = _setup(7, K)
    fp, fq, r = _make_polynomial_field(kr, K, degree=2)
    coeffs = correction.fit(m, S, fp, fq)
    g = lambda x_: r(x_) - correction.evaluate(coeffs, x_)
    J = jax.jacfwd(g)(x)
    np.testing.assert_allclose(J, J.T, atol=1e-6)


def test_least_squares_monte_carlo_converges_to_quadrature():
    """
    The root mean square error (over seeds) of the Monte Carlo estimate of the correction, relative to the estimate with
    a fine quadrature, decreases with n_mc and is small at n_mc = 2^14
    """
    K, n_seeds = 2, 8
    x, m, S, kr = _setup(23, K)
    fp, fq = _make_smooth_field(kr, K)
    quadrature = LeastSquaresCorrection(ell=2, kappa=1, n_nodes=40)
    h_quad = quadrature.evaluate(quadrature.fit(m, S, fp, fq), x)
    rms = []
    for n_mc in (2 ** 8, 2 ** 14):
        monte_carlo = LeastSquaresCorrection(ell=2, kappa=1, n_mc=n_mc)
        h_mc = [monte_carlo.evaluate(monte_carlo.fit(m, S, fp, fq, key=jr.PRNGKey(seed)), x) for seed in range(n_seeds)]
        rms.append(float(jnp.sqrt(jnp.mean(jnp.array([jnp.sum(jnp.square(h - h_quad)) for h in h_mc])))))
    assert rms[1] < 0.5 * rms[0]
    assert rms[1] < 5e-2 * float(jnp.linalg.norm(h_quad))


@pytest.mark.parametrize("correction", [TaylorCorrection(ell=0), LeastSquaresCorrection(ell=0, kappa=1, n_nodes=3), TaylorCorrection(ell=2), LeastSquaresCorrection(ell=2, kappa=1, n_nodes=4)], ids=_id)
def test_correction_vanishes_without_divergence_free_fields(correction):
    """
    No non-zero polynomial field of degree 0 is q-divergence-free, and none of any degree is in one dimension, so h = 0
    """
    for K in ((2, 3) if correction.ell == 0 else (1,)):
        x, m, S, kr = _setup(29 + K, K)
        fp, fq = _make_smooth_field(kr, K)
        h = correction.evaluate(correction.fit(m, S, fp, fq), x)
        np.testing.assert_allclose(h, 0.0, atol=1e-12)


# --------------------- Coefficients ---------------------
@pytest.mark.parametrize("correction", TAYLOR + LEAST_SQUARES, ids=_id)
def test_monomial_coeffs_match_evaluate(correction):
    K = 3
    x, m, S, kr = _setup(11, K)
    fp, fq = _make_smooth_field(kr, K)
    coeffs = correction.fit(m, S, fp, fq)
    P = correction.monomial_coeffs(coeffs)
    assert len(P) == correction.ell + 1
    for i, C in enumerate(P):
        assert C.ndim == (correction.ell - i) + 1
    np.testing.assert_allclose(_eval_monomial(P, x), correction.evaluate(coeffs, x), atol=1e-8, rtol=1e-8)


def _random_taylor_tensors(key: jr.PRNGKey, K: int, ell: int):
    """
    Random derivative tensors E[l], symmetric in the derivative axes, defining p_i(v) = sum_l E[l] v^l / l!
    """
    return [symmetrize_full(jr.normal(kk, (K,) * (l + 1), dtype=jnp.float64), n_axes=l, start_axis=1)
            for l, kk in enumerate(jr.split(key, ell + 1))]


def _contract(T, v, n):
    out = T
    for _ in range(n):
        out = out @ v
    return out


def _eval_taylor(E, v):
    """
    Evaluate the polynomial defined by derivative tensors E at v
    """
    out = jnp.zeros(v.shape, dtype=v.dtype)
    for l, T in enumerate(E):
        out = out + _contract(T, v, l) / float(math.factorial(l))
    return out


@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_monomial_to_hermite_matches_quadrature(ell, K):
    """
    The trace-correction formula agrees with the defining integral b[alpha, i] = E[H_alpha(v) p_i(v)]

    Gauss-Hermite with ell + 1 nodes per dimension is exact to total degree 2 ell + 1, so it integrates the
    degree-2 ell integrand exactly and serves as an independent oracle for the closed form
    """
    E = _random_taylor_tensors(jr.PRNGKey(11 * ell + K), K, ell)
    alpha_table, *_ = index_tables(K, ell)
    nodes, weights = gauss_hermite_nodes(K, ell + 1)
    vs, w = jnp.asarray(nodes), jnp.asarray(weights)
    values = jax.vmap(lambda v: _eval_taylor(E, v))(vs) # (N, K)
    H = jax.vmap(lambda v: hermite_products(v, alpha_table, ell))(vs) # (N, Na)
    reference = jnp.einsum("n,na,ni->ai", w, H, values)

    b = monomial_to_hermite(E, K, ell)
    assert b.shape == (alpha_table.shape[0], K)
    np.testing.assert_allclose(b, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("K", [1, 2, 3, 5])
@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_hermite_monomial_round_trip(ell, K):
    """
    monomial_to_hermite and hermite_to_monomial invert each other through a non-trivial whitened frame

    The frame is deliberately not the identity: sqrtS is symmetric but frame = sqrtS Q is not, so this also
    pins the orientation of the transport in hermite_to_monomial
    """
    key_A, key_m, key_x = jr.split(jr.PRNGKey(31 * ell + K), 3)
    S = _random_spd(key_A, K)
    m = jr.normal(key_m, (K,), dtype=jnp.float64)
    Q, lam, sqrtS, invsqrtS, frame = whiten_frame(S)

    E = _random_taylor_tensors(jr.PRNGKey(7 * ell + K), K, ell)
    coeffs = HermiteCoeffs(c=monomial_to_hermite(E, K, ell), Q=Q, sqrtS=sqrtS, invsqrtS=invsqrtS, m=m)
    P = hermite_to_monomial(coeffs, ell)

    # The monomial form must reproduce h(x) = frame nu(invframe (x - m)) in the original coordinates
    x = jr.normal(key_x, (K,), dtype=jnp.float64)
    invframe = Q.T @ invsqrtS
    reference = frame @ _eval_taylor(E, invframe @ (x - m))
    np.testing.assert_allclose(_eval_monomial(P, x), reference, atol=1e-9, rtol=1e-9)


def test_hermite_monomial_round_trip_ill_conditioned():
    """
    The round trip survives a marginal covariance with condition number 1e8
    """
    K, ell = 3, 2
    Q0, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(1), (K, K), dtype=jnp.float64))
    S = Q0 @ jnp.diag(jnp.array([1e4, 1.0, 1e-4])) @ Q0.T
    m = jr.normal(jr.PRNGKey(2), (K,), dtype=jnp.float64)
    Q, lam, sqrtS, invsqrtS, frame = whiten_frame(S)

    E = _random_taylor_tensors(jr.PRNGKey(4), K, ell)
    coeffs = HermiteCoeffs(c=monomial_to_hermite(E, K, ell), Q=Q, sqrtS=sqrtS, invsqrtS=invsqrtS, m=m)
    P = hermite_to_monomial(coeffs, ell)

    x = m + jr.normal(jr.PRNGKey(5), (K,), dtype=jnp.float64)
    reference = frame @ _eval_taylor(E, (Q.T @ invsqrtS) @ (x - m))
    np.testing.assert_allclose(_eval_monomial(P, x), reference, rtol=1e-8, atol=1e-8)


def test_taylor_ell4_shapes_and_finiteness():
    ell, K = 4, 3
    x, m, S, kr = _setup(13, K)
    fp, fq, _ = _make_polynomial_field(kr, K, degree=ell)
    correction = TaylorCorrection(ell=ell)
    coeffs = correction.fit(m, S, fp, fq)
    h = correction.evaluate(coeffs, x)
    P = correction.monomial_coeffs(coeffs)
    assert h.shape == (K,)
    assert jnp.all(jnp.isfinite(h))
    assert len(P) == ell + 1
    for i, C in enumerate(P):
        assert C.ndim == (ell - i) + 1
        assert jnp.all(jnp.isfinite(C))


# --------------------- Defaults and validation ---------------------
def test_default_correction():
    assert isinstance(DEFAULT_CORRECTION, HelmCorrection)
    assert DEFAULT_CORRECTION == LeastSquaresCorrection(ell=1, kappa=1, n_mc=1)


def test_invalid_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        TaylorCorrection(ell=-1)
    with pytest.raises(ValueError):
        LeastSquaresCorrection(ell=-1)
    with pytest.raises(ValueError):
        LeastSquaresCorrection(ell=1, kappa=-1)
    with pytest.raises(ValueError):
        LeastSquaresCorrection(ell=1, n_mc=0)
    with pytest.raises(ValueError):
        LeastSquaresCorrection(ell=1, n_nodes=0)
    with pytest.raises(ValueError):
        TaylorCorrection(ell=1.0)


def test_monte_carlo_fit_requires_key():
    K = 2
    x, m, S, kr = _setup(14, K)
    fp, fq = _make_smooth_field(kr, K)
    with pytest.raises(ValueError):
        DEFAULT_CORRECTION.fit(m, S, fp, fq)


# --------------------- KL estimator and corrected drift ---------------------
@pytest.mark.parametrize("correction", [TaylorCorrection(ell=1), TaylorCorrection(ell=2), LeastSquaresCorrection(ell=1, kappa=1, n_mc=1), LeastSquaresCorrection(ell=2, kappa=1, n_mc=4), LeastSquaresCorrection(ell=1, kappa=1, n_nodes=3)], ids=_id)
def test_kl_estimator_under_jit_and_grad(correction):
    """
    The KL estimator is finite and differentiable under jit for both corrections (the Stein path nests jacrev)
    """
    K = 3
    x, m, S, kr = _setup(17, K)
    fp, fq0 = _make_smooth_field(kr, K)
    Gt = jnp.diag(jnp.array([1.0, 0.5, 2.0]))
    key = jr.PRNGKey(0)

    def loss(theta):
        fq = lambda x_: theta * fq0(x_)
        return compute_kl_helmholtz_approx(m, S, fp, fq, Gt, key, correction=correction, n_z0=4)

    val, grad = jax.jit(jax.value_and_grad(loss))(1.0)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)
    assert val >= 0.0


@pytest.mark.parametrize("correction", [TaylorCorrection(ell=2), LeastSquaresCorrection(ell=2, kappa=1, n_mc=8)], ids=_id)
def test_corrected_drift_matches_kl_estimator(correction):
    """
    compute_kl_helmholtz_approx equals (1/2) E||G^{-1}(fp - fq*)||^2 for the drift returned by make_corrected_drift, whose
    Monte Carlo fit uses the key fold_in(key, 1) of the estimator
    """
    K = 3
    x, m, S, kr = _setup(19, K)
    fp, fq = _make_smooth_field(kr, K)
    Gt = jnp.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.0, 0.0, 1.5]])
    key = jr.PRNGKey(5)
    n_z0 = 8

    kl = compute_kl_helmholtz_approx(m, S, fp, fq, Gt, key, correction=correction, n_z0=n_z0)

    # Reproduce the state samples of the estimator and evaluate the corrected drift directly
    fq_star = make_corrected_drift(m, S, fp, fq, Gt, correction=correction, key=jr.fold_in(key, 1))
    _, _, sqrtS, _ = sym_sqrt_and_invsqrt(S)
    zs = jr.normal(key, shape=(n_z0, K), dtype=S.dtype)
    xs = m[None, :] + (sqrtS @ zs.T).T
    resid = jax.vmap(lambda x_: jnp.linalg.solve(Gt, fp(x_) - fq_star(x_)))(xs)
    kl_direct = 0.5 * jnp.mean(jnp.sum(jnp.square(resid), axis=-1))
    np.testing.assert_allclose(kl, kl_direct, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("correction", [TaylorCorrection(ell=2), LeastSquaresCorrection(ell=2, kappa=1, n_nodes=4), LeastSquaresCorrection(ell=2, kappa=1, n_mc=32)], ids=_id)
def test_corrected_drift_is_q_divergence_free_for_general_diffusion(correction):
    """
    fq* - fq is q-divergence-free in the original coordinates for a non-identity diffusion coefficient
    """
    K = 3
    x, m, S, kr = _setup(37, K)
    fp, fq = _make_smooth_field(kr, K)
    Gt = jnp.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.0, 0.0, 1.5]])
    fq_star = make_corrected_drift(m, S, fp, fq, Gt, correction=correction, key=jr.PRNGKey(0))
    h = lambda x_: fq_star(x_) - fq(x_)
    for i in range(3):
        xi = m + 0.7 * (i + 1) * (x - m)
        np.testing.assert_allclose(_q_divergence(h, xi, m, S), 0.0, atol=1e-6, err_msg=f"{_id(correction)} at point {i}")


# --------------------- Training loop ---------------------
def _tiny_training_problem():
    from helmholtz_sde.sde import LinearSDE
    from helmholtz_sde.likelihood import Gaussian
    from helmholtz_sde.posterior.encoder import ForwardGRUEncoder, weighted_ctx
    from helmholtz_sde.posterior.nn_posterior import InferenceNetwork

    K, B, T = 2, 3, 5
    obs_times = jnp.broadcast_to(jnp.linspace(0.0, 1.0, T), (B, T))
    ys = 0.1 * jr.normal(jr.PRNGKey(0), (B, T, K), dtype=jnp.float64)
    return dict(
        ys=ys,
        obs_times=obs_times,
        likelihood=Gaussian(),
        output_params={"C": jnp.eye(K), "d": jnp.zeros((K,)), "R": 0.1 * jnp.ones((K,))},
        t_max=1.0,
        encoder=ForwardGRUEncoder(8),
        process_ctx=weighted_ctx,
        post_net=InferenceNetwork(hidden_dim=8, K=K, depth=1),
        prior=LinearSDE(K),
        sde_params={"A": -jnp.eye(K), "b": jnp.zeros((K,))},
        init_params={"mu0": jnp.zeros((K,)), "V0": jnp.eye(K)},
    )


@pytest.mark.parametrize("div_free", [DEFAULT_CORRECTION, TaylorCorrection(ell=1), None], ids=_id)
def test_train_smoke(div_free):
    from helmholtz_sde.train import train

    n_iters = 3
    params, metrics = train(jr.PRNGKey(1), **_tiny_training_problem(), n_iters=n_iters, disable_pbar=True, div_free=div_free)
    for name, values in metrics.items():
        assert values.shape == (n_iters,), name
        assert jnp.all(jnp.isfinite(values)), name


def test_train_default_is_least_squares_and_single_iteration_runs():
    import inspect
    from helmholtz_sde.train import train

    assert inspect.signature(train).parameters["div_free"].default == LeastSquaresCorrection(ell=1, kappa=1, n_mc=1)
    params, metrics = train(jr.PRNGKey(2), **_tiny_training_problem(), n_iters=1, disable_pbar=True)
    assert metrics["loss"].shape == (1,)
    assert jnp.isfinite(metrics["loss"][0])


def test_train_rejects_string_correction():
    from helmholtz_sde.train import train

    with pytest.raises(TypeError):
        train(jr.PRNGKey(3), **_tiny_training_problem(), n_iters=1, disable_pbar=True, div_free="linear")


# --------------------- Posterior SDE ---------------------
def _tiny_posterior_sde(div_free, gauge="sqrt"):
    """
    Posterior SDE of the first trial of the tiny training problem after one training iteration, and the parameters
    """
    from helmholtz_sde.train import train
    from helmholtz_sde.posterior.drift import PosteriorSDE

    problem = _tiny_training_problem()
    params, _ = train(jr.PRNGKey(4), **problem, n_iters=1, disable_pbar=True, div_free=div_free)
    ctx_seq = problem["encoder"].apply(params["encoder_params"], problem["ys"][0])
    process_ctx = functools.partial(problem["process_ctx"], problem["obs_times"][0])
    return PosteriorSDE(problem["prior"], problem["post_net"], ctx_seq, process_ctx, gauge=gauge, div_free=div_free), params


@pytest.mark.parametrize("gauge", ["sqrt", "sym"])
def test_posterior_sde_without_correction_is_reference_drift(gauge):
    from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs, get_reference_drift_fn

    sde, params = _tiny_posterior_sde(None, gauge)
    x, t = jnp.array([0.3, -0.2]), jnp.array([0.4])
    mt, Rt, dmt, dRt = apply_inference_net_time_derivs(sde.post_net, params["posterior_params"], t, sde.ctx_seq, sde.process_ctx)
    fq = get_reference_drift_fn(gauge)(mt, Rt, dmt, dRt, jnp.eye(2))
    np.testing.assert_allclose(sde.drift(x, 0.4, params), fq(x), atol=1e-12) # the Ito term vanishes for the constant diffusion of LinearSDE
    np.testing.assert_allclose(sde.diffusion(x, 0.4, params), jnp.eye(2))
    m0, R0 = sde.marginal(0.0, params)
    assert m0.shape == (2,) and R0.shape == (2, 2)


def test_posterior_sde_correction_is_q_divergence_free():
    from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs, get_reference_drift_fn

    sde, params = _tiny_posterior_sde(LeastSquaresCorrection(ell=2, n_nodes=4))
    t = jnp.array([0.4])
    mt, Rt, dmt, dRt = apply_inference_net_time_derivs(sde.post_net, params["posterior_params"], t, sde.ctx_seq, sde.process_ctx)
    fq = get_reference_drift_fn("sqrt")(mt, Rt, dmt, dRt, jnp.eye(2))
    h = lambda x_: sde.drift(x_, 0.4, params) - fq(x_) # the correction, in the original coordinates since G = I
    S_inv = jnp.linalg.inv(Rt @ Rt.T)
    for x in [jnp.array([0.3, -0.2]), jnp.array([-1.0, 0.5])]:
        div_q = jnp.trace(jax.jacfwd(h)(x)) - h(x) @ S_inv @ (x - mt) # q-divergence of the correction
        assert abs(float(div_q)) < 1e-8
        assert float(jnp.linalg.norm(h(x))) > 0.0


def test_simulate_posterior_samples_shapes():
    from helmholtz_sde.posterior.drift import simulate_posterior_samples

    sde, params = _tiny_posterior_sde(LeastSquaresCorrection(ell=1, n_nodes=3))
    xs = simulate_posterior_samples(jr.PRNGKey(6), sde, params, t_max=1.0, n_timesteps=5, n_samples=4)
    assert xs.shape == (4, 6, 2)
    assert jnp.all(jnp.isfinite(xs))


def test_get_reference_drift_fn_rejects_unknown_gauge():
    from helmholtz_sde.posterior.drift import get_reference_drift_fn

    with pytest.raises(ValueError):
        get_reference_drift_fn("linear")
