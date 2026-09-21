# Noisy Lorenz attractor

## Files
This subdirectory contains three files:
- `lorenz_attractor.ipynb`: Notebook demonstrating the functionality of Helmholtz-SDE on the noisy Lorenz attractor dataset
- `lorenz_utils.py`: File containing functions for evaluating the learned dynamics
- `lorenz_experiment.py`: File for reproducing the results from the paper

## Data
The noisy Lorenz attractor example does not require loading a pre-existing dataset. 
The data is generated directly in `lorenz_attractor.ipynb`.

Running the first few cells of the notebook will produce two pickle files, `train.pkl` and `test.pkl`, containing the train and test trajectories from the stochastic Lorenz attractor dataset. 
By default, these files are written to `experiments/datasets/lorenz`, which is the directory to pass to the script via `--dataset_path`.
`lorenz_experiment.py` generates and writes the same dataset when `--dataset_path` is omitted.

## Training
`lorenz_experiment.py` selects the Helmholtz correction used during training with `--gauge {sym, sqrt}`, `--div_free {none,taylor,least_squares}` and `--ell`; for `least_squares`, it additionally takes `--kappa` and either `--n_mc` or `--n_nodes`.
For example,
```bash
python lorenz_experiment.py --output_dir results --output_name helmholtz --dataset_path ../datasets/lorenz \
    --learn_diff 0 --gauge sqrt --div_free least_squares --ell 1 --kappa 1 --n_mc 1
```
A learned state-dependent diffusion coefficient (`--learn_diff 1`) is only supported without a correction. 
