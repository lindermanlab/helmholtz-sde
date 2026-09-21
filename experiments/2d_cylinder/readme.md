# Fluid flow past a 2D cylinder

## Files
This subdirectory contains four files:
- `cylinder_flow.ipynb`: Notebook for preprocessing the raw CFD data: it performs a POD (PCA) reduction and writes the reduced `encoded_data.pkl` used by `train_cylinder.py` and `plot_cylinder_flow.ipynb`
- `train_cylinder.py`: File for training a latent SDE on the POD-reduced cylinder flow data
- `cylinder_utils.py`: File containing functions for simulating, decoding and plotting the POD-reduced flow, used by the notebooks
- `plot_cylinder_flow.ipynb`: Notebook for loading a trained checkpoint and producing the evaluation, forecasting and figure results

## Data
This example uses the 2D cylinder flow dataset from Course and Nair, 2023. 
The raw data is not included in this repository. 
It can be downloaded from Zenodo, following the instructions described in the [authors' codebase](https://github.com/coursekevin/svise).
Note that the full dataset `vortex.pkl` is approximately 36 GB.

Running `cylinder_flow.ipynb` reads `vortex.pkl`, reduces it via POD and writes `experiments/datasets/cylinder/encoded_data.pkl`. 
`encoded_data.pkl` is approximately 100 MB.
Both `train_cylinder.py` and `plot_cylinder_flow.ipynb` then read `experiments/datasets/cylinder/encoded_data.pkl`.

## Training
`train_cylinder.py` trains a latent SDE on the POD coefficients, subsampled every `--subsample_every` frames and observed with noise `--obs_noise` (proportional to the standard deviation of each mode by default).
The posterior approximation is a GP.
It selects the Helmholtz correction used during training with `--gauge {sqrt,sym}`, `--div_free {none,taylor,least_squares}`, and `--ell`; for `least_squares`, it additionally takes `--kappa` and `--n_mc`.
For example,
```bash
python train_cylinder.py --dataset_dir ../datasets/cylinder --gauge sqrt --div_free least_squares --ell 1 --kappa 1 --n_mc 1 --key_init 0
```
Checkpoints are written to `<ckpt_dir>/<method>/<name>.pkl`, where `<method>` is `{sde_matching,svise,taylor,least_squares}` and `<name>` records the observation settings, the gauge, the size of the drift network, the degree and estimator settings of the correction, and the replicate `--key_init`.
The checkpoint holds the parameters, the training metrics, and the settings of the run; `plot_cylinder_flow.ipynb` loads the checkpoint.


Note that `--gauge sqrt --div_free none` does not use a state dependent diffusion coefficient in this setting.