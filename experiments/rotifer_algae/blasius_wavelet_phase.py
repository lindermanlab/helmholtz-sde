"""
Wavelet phase analysis following Blasius et al., 2020 (Nature)
"""

import functools
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np


@functools.cache
def _init_julia() -> ModuleType:
    """
    Loads the WaveletAnalysis Julia library and defines the per-sample analysis helper
    """
    try:
        from juliacall import Main as jl
    except ImportError as e:
        raise ImportError("juliacall is required: pip install juliacall") from e

    # WaveletAnalysis depends on a few registered Julia packages that juliacall does not install automatically; add any
    # that are missing from the active Julia project
    jl.seval("import Pkg")
    jl.seval('for p in ["DSP", "FFTW", "StatsFuns", "SpecialFunctions"]; Base.identify_package(p) === nothing && Pkg.add(p); end')
    jl.include(str(Path(__file__).resolve().parent / "WaveletAnalysis" / "src" / "wcs.jl"))

    jl.seval("""
    function _blasius_single_sample(algae_in, rotifers_in, dt::Float64, nvoices::Int, noctave::Int, wco_threshold::Float64, omega0::Float64)
        algae = collect(Float64, algae_in)
        rotifers = collect(Float64, rotifers_in)
        n = length(algae)

        spower1, spower2, amplitude, phase, wco, period, _, coi = wcs(algae, rotifers, dt; nvoices=nvoices, noctave=noctave, param=omega0)

        # Adaptive period band around the peak of the global wavelet cross-spectrum
        global_wcs = mean(amplitude, dims=2)
        delta_ind = trunc(Int, 0.6 * nvoices)
        ind_max = argmax(global_wcs)[1]
        ind1 = max(1, ind_max - delta_ind)
        ind2 = min(length(period), ind_max + delta_ind)
        dominant_period = period[ind_max]

        # Wavelet power spectrum peak of each signal, masked by the cone of influence
        nscales = length(period)
        coi_mask = BitMatrix(undef, nscales, n)
        for ti in 1:n
            for si in 1:nscales
                coi_mask[si, ti] = period[si] <= coi[ti]
            end
        end
        amp1 = sqrt.(max.(spower1, 0.0))
        amp2 = sqrt.(max.(spower2, 0.0))
        gws1 = zeros(nscales)
        gws2 = zeros(nscales)
        for si in 1:nscales
            cnt = sum(coi_mask[si, :])
            if cnt > 0
                gws1[si] = sum(amp1[si, :] .* coi_mask[si, :]) / cnt
                gws2[si] = sum(amp2[si, :] .* coi_mask[si, :]) / cnt
            end
        end
        wps_peak_algae = period[argmax(gws1)]
        wps_peak_rotifers = period[argmax(gws2)]

        # Phases at the times of high coherence within the band, excluding the cone of influence and the band edges
        wco_max_ind, angle_max, wco_max = get_maxInd(wco, phase, n, ind1, ind2)
        indices_wco = findall(x -> x >= wco_threshold, wco_max)
        wco_scale = log2.(period[wco_max_ind])
        relevant = wco_scale .< log2.(coi)
        indices_wco = indices_wco[findall(x -> relevant[x], indices_wco)]
        indices_wco = indices_wco[findall(x -> (x != ind2 && x != ind1), wco_max_ind[indices_wco])]

        coherent_fraction = length(indices_wco) / n
        if length(indices_wco) == 0
            return NaN, NaN, coherent_fraction, dominant_period, NaN, wps_peak_algae, wps_peak_rotifers
        end
        theta, _, circ_std = circstats(pi * angle_max[indices_wco]) # circular mean and standard deviation of the coherent phases
        mean_period_coherent = mean(period[wco_max_ind[indices_wco]])
        return theta, circ_std, coherent_fraction, dominant_period, mean_period_coherent, wps_peak_algae, wps_peak_rotifers
    end
    """)
    return jl


def blasius_phase_analysis(samples: np.ndarray, t_grid: np.ndarray, nvoices: int = 100, noctave: int = 3, wco_threshold: float = 0.83, omega0: float = 6.0) -> Dict[str, Any]:
    """
    Wavelet phase analysis of Blasius et al., 2020 for each sample path (T, 2) of (algae, rotifers) abundances on the
    regular time grid t_grid
    """
    jl = _init_julia()
    samples = np.asarray(samples, dtype=np.float64)
    n_samples, T, n_channels = samples.shape
    if n_channels != 2:
        raise ValueError(f"expected 2 channels (algae, rotifers), got {n_channels}")
    dt = float(t_grid[1] - t_grid[0])
    print(f"Blasius et al., 2020 wavelet phase analysis of {n_samples} samples with {T} points at dt = {dt:.4f} days (omega0 = {omega0}, nvoices = {nvoices}, noctave = {noctave}, WCO threshold = {wco_threshold})")

    phase_lags_deg = np.full(n_samples, np.nan)
    coherent_fractions = np.zeros(n_samples)
    dom_periods = np.zeros(n_samples)
    mean_periods_coherent = np.full(n_samples, np.nan)
    wps_peak_algae = np.zeros(n_samples)
    wps_peak_rotifers = np.zeros(n_samples)
    for i in range(n_samples):
        algae, rotifers = np.ascontiguousarray(samples[i, :, 0]), np.ascontiguousarray(samples[i, :, 1])
        theta, _, coherent_fractions[i], dom_periods[i], mean_periods_coherent[i], wps_peak_algae[i], wps_peak_rotifers[i] = jl._blasius_single_sample(algae, rotifers, dt, nvoices, noctave, wco_threshold, omega0)
        phase_lags_deg[i] = np.degrees(theta)
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  sample {i + 1}/{n_samples}: phase lag {phase_lags_deg[i]:.1f} deg, coherent fraction {coherent_fractions[i]:.1%}")

    # Ensemble statistics: 
    # (1) the median over samples of the per-sample peak period,
    # (2) the arithmetic mean over samples with coherent phases of the per-sample circular mean phase lag,
    # converted to a lag in days with the median peak period, 
    # (3) the median over samples of the rotational coherence
    peak_periods = 0.5 * (wps_peak_algae + wps_peak_rotifers)
    peak_period = float(np.median(peak_periods))
    phase_lag_deg = float(np.nanmean(phase_lags_deg))
    lag_days = peak_period * phase_lag_deg / 360.0
    rotational_coherence = float(np.median(coherent_fractions))
    n_coherent = int(np.sum(~np.isnan(phase_lags_deg)))
    print(f"Peak period: {peak_period:.2f} days")
    print(f"Phase lag: {phase_lag_deg:.1f} +/- {np.nanstd(phase_lags_deg):.1f} deg ({n_coherent}/{n_samples} samples with coherent phases), i.e. {lag_days:.2f} days")
    print(f"Rotational coherence: {rotational_coherence:.1%}")
    print("Blasius et al., 2020 report a phase lag of 93 +/- 21 deg and a period of 6.7 days")
    return {
        "phase_lags_deg": phase_lags_deg,
        "coherent_fractions": coherent_fractions,
        "dom_periods": dom_periods,
        "mean_periods_coherent": mean_periods_coherent,
        "wps_peak_algae": wps_peak_algae,
        "wps_peak_rotifers": wps_peak_rotifers,
        "peak_periods": peak_periods,
        "peak_period": peak_period,
        "phase_lag_deg": phase_lag_deg,
        "lag_days": lag_days,
        "rotational_coherence": rotational_coherence,
    }
