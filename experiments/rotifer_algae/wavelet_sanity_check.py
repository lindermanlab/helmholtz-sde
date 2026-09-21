"""
Sanity check of the wavelet phase pipeline on deterministic sinusoids with known periods and a 90 degree phase lag
"""

import contextlib

import numpy as np

from blasius_wavelet_phase import blasius_phase_analysis


def main() -> None:
    duration, dt = 1000.0, 0.5 # days
    true_phase_deg = 90.0 # algae leads rotifers by 90 deg
    periods = [6.0, 7.0, 8.0, 9.0, 10.0] # one sample per period
    t = np.arange(0, duration, dt)
    samples = np.zeros((len(periods), len(t), 2))
    for i, period in enumerate(periods):
        omega = 2 * np.pi / period
        samples[i, :, 0] = np.sin(omega * t) # algae
        samples[i, :, 1] = np.sin(omega * t - np.radians(true_phase_deg)) # rotifers, lagging by 90 deg

    print(f"True phase lag: {true_phase_deg} deg, true periods: {periods}, duration: {duration} days, dt: {dt} days ({len(t)} points)\n")
    results = blasius_phase_analysis(samples, t, noctave=5)

    print("\nPer-sample recovery:")
    print(f"{'Period (true)':>14s} {'Period (WCS)':>14s} {'WPS (algae)':>14s} {'WPS (rot)':>14s} {'WPS (avg)':>14s} {'Period (coh)':>14s} {'Phase (deg)':>12s} {'Coh frac':>10s}")
    for i, period in enumerate(periods):
        wps_avg = 0.5 * (results["wps_peak_algae"][i] + results["wps_peak_rotifers"][i])
        print(f"{period:14.1f} {results['dom_periods'][i]:14.2f} {results['wps_peak_algae'][i]:14.2f} {results['wps_peak_rotifers'][i]:14.2f} {wps_avg:14.2f} {results['mean_periods_coherent'][i]:14.2f} {results['phase_lags_deg'][i]:12.1f} {results['coherent_fractions'][i]:10.1%}")


if __name__ == "__main__":
    # NOTE: the first run downloads a Julia runtime and its packages, so it needs network access
    with open("wavelet_sanity_check.log", "w") as log_file, contextlib.redirect_stdout(log_file):
        main()
    print("Results written to wavelet_sanity_check.log")
