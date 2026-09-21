# Linear SDE (Ornstein-Uhlenbeck spiral)

## Files
This subdirectory contains four files:
- `simple_rotation.ipynb`: Notebook demonstrating the functionality of Helmholtz-SDE on the Ornstein-Uhlenbeck spiral dataset
- `spiral_utils.py`: File containing functions for computing the exact posterior with SING
- `spiral_learning.py`: File for reproducing the learning experiments from the paper (learned prior and output model)
- `spiral_inference.py`: File for reproducing the inference experiments from the paper (known prior and output model)

## Data
The Ornstein-Uhlenbeck spiral example does not require loading a pre-existing dataset. 
The data is generated directly in `simple_rotation.ipynb`.

Running the first few cells of the notebook will produce two pickle files, `train.pkl` and `test.pkl`, containing the train and test trajectories from the Ornstein-Uhlenbeck spiral dataset. 
By default, these files are written to `experiments/datasets/spiral/spiral_pi`, which is the directory to pass to the scripts via `--dataset_dir`.

## Requirements
The scripts and the notebook compute the exact posterior with [SING](https://github.com/lindermanlab/sing) `(Hu et al., 2025)`.
The local copy is installed from the repository root with
```bash
pip install -e sing/
```

## Training
`spiral_inference.py` fixes the prior and output model to the truth and fits one posterior per trial. 
It does so for every combination of observation noise `--ssigmas` and number of observations per trial `--n_obs_grid`.
It selects the Helmholtz correction used during training with `--gauge {sym, sqrt}`, `--div_free {none,taylor,least_squares}` and `--ell`; for `least_squares`, it additionally takes `--kappa` and either `--n_mc` or `--n_nodes`.
For example,
```bash
python spiral_inference.py --output_dir results --output_name helmholtz --dataset_dir ../datasets/spiral/spiral_pi \
    --gauge sqrt --div_free least_squares --ell 1 --kappa 1 --n_mc 1
```
The posterior is defined on a grid by default (`--posterior_type grid`).
`--posterior_type gp` and `--posterior_type nn` fit a per trial GP posterior or inference network instead.

`spiral_learning.py` has the same structure as `spiral_inference.py`, but it additionally learns the prior and output model.
For example,
```bash
python spiral_learning.py --output_dir results --output_name helmholtz --dataset_dir ../datasets/spiral/spiral_pi \
    --gauge sqrt --div_free least_squares --ell 1 --kappa 1 --n_mc 1
```
The posterior is by default one GP posterior per trial with `--n_tau 200` inducing points and kernel hyperparameters shared across trials (`--posterior_type gp`).
`--posterior_type nn` fits an amortized inference network instead.
