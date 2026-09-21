# Rotifer-algae predator-prey system

## Files
This subdirectory contains three files and one directory:
- `algae.ipynb`: Notebook demonstrating the functionality of Helmholtz-SDE on the Blasius et al., 2020 rotifer-algae chemostat dataset
- `blasius_wavelet_phase.py`: File containing functions for the wavelet phase analysis of posterior samples, following Blasius et al., 2020
- `wavelet_sanity_check.py`: File for validating the wavelet pipeline on synthetic sinusoids with known period and phase lag
- `WaveletAnalysis/`: Directory containing a copy of the Julia wavelet library used by `blasius_wavelet_phase.py`

## Data
This example uses the chemostat time series data obtained from Blasius et al., 2020. 
The [data](https://www.nature.com/articles/s41586-019-1857-0#data-availability) is made publicly available by the authors.

The notebook uses `C1.csv`, which contains 374 days of continuous recordings. 
Place it at `experiments/datasets/algae/C1.csv`, which is where `algae.ipynb` reads it from.

## Requirements
`algae.ipynb` and `wavelet_sanity_check.py` run the Julia library `WaveletAnalysis/` through [`juliacall`](https://pypi.org/project/juliacall/), which is installed with:
```bash
pip install juliacall
```
No separate Julia installation is required: on first use `juliacall` downloads a Julia runtime into `~/.julia` (about 1 GB; set `JULIA_DEPOT_PATH` to relocate it), and `blasius_wavelet_phase.py` adds the Julia packages the library depends on (`DSP`, `FFTW`, `StatsFuns`, `SpecialFunctions`) to the active Julia project. 
The first run needs internet access and takes a few minutes; later runs reuse the installation. 
An existing Julia installation is picked up if it is on the path or set via `PYTHON_JULIAPKG_EXE`.

`WaveletAnalysis/` is a copy of the [code](https://github.com/berndblasius/WaveletAnalysis) released with Blasius et al., 2020.