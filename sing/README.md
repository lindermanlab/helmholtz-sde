# 🎶SING: SDE Inference via Natural Gradients🎶

NOTE: this is a stripped copy of the [SING repository](https://github.com/lindermanlab/sing) vendored for Helmholtz-SDE; the demo notebooks and figures are omitted.

This repository implements the SING method for variational inference in latent SDE models from our paper,

**SING: SDE Inference via Natural Gradients**\
Amber Hu*, Henry Smith*, Scott W. Linderman \
(*Equal contribution)\
Advances in Neural Information Processing Systems (NeurIPS), 2025.\
[arXiv](https://arxiv.org/abs/2506.17796)\
[OpenReview](https://openreview.net/forum?id=jmnt0F21K7)

SING is a method for fast and reliable variational inference in *latent SDE models*, which are used to uncover unobserved dynamical systems from noisy data. To do this, SING leverages natural gradient variational inference, which adapts updates to the geometry of the variational distribution and prior. This leads to faster convergence and greater stability during inference than previous methods, which in turn enables more accurate parameter learning and use of flexible priors. 

This codebase features an efficient implementation of SING which is parallelized over sequence length and batch size, enabling scalable inference on large datasets. It also implements SING-GP, an extension of SING to inference and learning for latent SDE models with drift functions modeled with Gaussian process priors. This includes the [gpSLDS model](https://github.com/lindermanlab/gpslds) from Hu et al. (NeurIPS, 2024).

## Getting started

For installing SING locally, we recommend using a virtual environment with Python version `>=3.9`. First, run:
```
pip install -U pip
```
Then, install JAX version `<= 0.5.3` in your virtual environment via the instructions [here](https://docs.jax.dev/en/latest/installation.html). For most use cases, we recommend installing JAX for GPUs with the command:
```
pip install -U "jax[cuda12]==0.5.3" "jaxlib==0.5.3"
```
Finally, install the package and its dependencies with either:
```
pip install -e .                # Install sing and core dependencies
pip install -e .[notebooks]     # Install with demo notebook dependencies
```

All source code can be found in the `sing/` folder.

## Tests

Tests for the sing codebase are written in `tests/test_sing.py`. You can run our test code with:
```
pytest tests/test_sing.py -q -s
```

## Citation

```
@article{hu2025sing,
  title={SING: SDE Inference via Natural Gradients},
  author={Hu, Amber and Smith, Henry D and Linderman, Scott},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```