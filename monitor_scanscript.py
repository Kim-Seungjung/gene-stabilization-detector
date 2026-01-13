#!/usr/bin/env python3

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import subprocess
from scipy import stats
from scipy.ndimage import uniform_filter1d


scan_dir = os.getenv("SCAN_DIR") # "/pscratch/sd/k/kim_sj/Case_2_3/scanfiles0000/"
output_dir = scan_dir
stop_dir = os.getenv("STOP_DIR")  #"/global/homes/k/kim_sj/gene/prob02"

inpar_dir = os.path.join(scan_dir, "in_par")

run_ids = sorted(
    f.split("_")[-1]
    for f in os.listdir(inpar_dir)
    if f.startswith("parameters_")
)

print(f"Detected {len(run_ids)} scan runs: {run_ids}")

poll_interval = 10       # seconds between checks
max_runtime = 1800    # safety stop after 30 minutes

# Automatically create output directory
os.makedirs(output_dir, exist_ok=True)


def stop(run_id):
    stop_file = os.path.join(stop_dir, f"GENE.stop_{run_id}")
    subprocess.run(["touch", stop_file], check=True)
    print(f"[STOP] Created {stop_file}")


def sliding_windows_indices(N, window_size, step):
    return np.arange(0, N - window_size + 1, step)

def instantaneous_frequency_and_amplitude(x, fs):
    analytic = signal.hilbert(x)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2*np.pi)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    return amp, inst_freq

def compute_psd(x, fs):
    f, Pxx = signal.welch(x, fs=fs, nperseg=max(len(x)//2, 8))
    return f, Pxx

def find_dominant_peak(f, Pxx, threshold_ratio=0.2):
    max_P = np.max(Pxx)
    peaks, props = signal.find_peaks(Pxx, height=max_P * threshold_ratio)
    if len(peaks) == 0:
        return None, None
    idx = peaks[np.argmax(Pxx[peaks])]
    return f[idx], Pxx[idx]


# ===== steady detector =====
def detect_steady_stabilization(time, gamma, 
                                window_fraction=0.10,
                                window_step_fraction=0.02,
                                tol_relstd=0.02,
                                tol_slope_rel=0.02,
                                consec_needed=3):
    eps = 1e-12
    Ttot = time[-1] - time[0]
    win_len = window_fraction * Ttot
    step = max(window_step_fraction * Ttot, 1e-12)

    window_starts = np.arange(time[0], time[-1] - win_len, step)

    passes = []
    details = []

    for t0 in window_starts:
        t1 = t0 + win_len
        m = (time >= t0) & (time <= t1)

        if m.sum() < 3:
            details.append((t0, np.nan, np.nan, np.nan))
            passes.append(False)
            continue

        g = gamma[m]
        t = time[m]

        mean_g = np.mean(g)
        std_g = np.std(g)

        slope, *_ = stats.linregress(t, g)
        rel_std = std_g / (abs(mean_g) + eps)
        rel_slope = abs(slope) / (abs(mean_g) + eps)

        ok = (rel_std <= tol_relstd) and (rel_slope <= tol_slope_rel)
        passes.append(ok)
        details.append((t0, mean_g, rel_std, rel_slope))

    passes = np.array(passes)

    for i in range(len(passes) - consec_needed + 1):
        if np.all(passes[i:i+consec_needed]):
            return details[i][0], details[i]

    return None, None


# ===== oscillatory detector =====
def detect_oscillatory_stabilization(time, gamma,
                                     fs,
                                     window_time=0.5,
                                     overlap=0.5,
                                     amp_tol=0.05,
                                     freq_tol=0.05,
                                     consec_needed=5,
                                     min_cycles = 5):
    window_size = max(int(round(window_time * fs)), 8)
    step = max(int(round(window_size * (1 - overlap))), 1)

    N = len(gamma)
    if N < window_size:
        return None, None

    starts = sliding_windows_indices(N, window_size, step)
    f_peaks = []
    amp_ptps = []
    times = []

    for s in starts:
        w = gamma[s : s + window_size]
        tcenter = time[s + window_size//2]
        times.append(tcenter)

        f, Pxx = signal.welch(w, fs=fs, nperseg=max(window_size//2, 8))
        if np.all(Pxx == 0):
            f_peaks.append(np.nan)
            amp_ptps.append(np.nan)
            continue

        idx = np.argmax(Pxx)
        f_peak = f[idx]

        if f_peak <= 0:
            f_peaks.append(0.0)
            amp_ptps.append(np.ptp(w))
            continue

        period = 1.0 / f_peak

        f_peaks.append(f_peak)
        amp_ptps.append(np.ptp(w))

    f_peaks = np.array(f_peaks)
    amp_ptps = np.array(amp_ptps)
    times = np.array(times)

    for i in range(0, len(starts) - consec_needed + 1):
        block_f = f_peaks[i : i + consec_needed]
        block_amp = amp_ptps[i : i + consec_needed]

        if np.any(np.isnan(block_f)) or np.all(block_f == 0):
            continue

        med_f = np.median(block_f)
        med_amp = np.median(block_amp) + 1e-12

        rel_f_change = np.max(np.abs(block_f - med_f)) / (abs(med_f) + 1e-12)
        rel_amp_change = np.max(np.abs(block_amp - med_amp)) / (abs(med_amp) + 1e-12)

        if (rel_f_change <= freq_tol) and (rel_amp_change <= amp_tol):
            stable_time = times[i + consec_needed - 1]
            details = {
                "stable_time": float(stable_time),
                "med_f": float(med_f),
                "rel_f_change": float(rel_f_change),
                "med_amp": float(med_amp),
                "rel_amp_change": float(rel_amp_change),
            }
            return stable_time, details

    return None, None


# ============================================================
# MONITORING LOOP
# ============================================================

def save_plot(time, gamma, steady_t, osc_t, run_id):
    """Save PNG showing detection lines."""
    plt.figure(figsize=(10,6))
    plt.plot(time, gamma, label="γ(t)")

    if steady_t is not None:
        plt.axvline(steady_t, color="gold", linewidth=2,
                    label=f"Steady at t={steady_t:.3g}")

    if osc_t is not None:
        plt.axvline(osc_t, color="red", linestyle="--", linewidth=2,
                    label=f"Oscillatory at t={osc_t:.3g}")

    plt.xlabel("Time")
    plt.ylabel("γ(t)")
    plt.title("Instantaneous Growth Rate — Stabilization Detection")
    plt.grid(True)
    plt.legend()

    outfile = os.path.join(output_dir, f"Detection_Of_Stabilization_{run_id}.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[Saved] {outfile}")


def monitor():
    print("===== Starting gamma monitor for scan =====")

    for run_id in run_ids:
        print(f"\n--- Monitoring run {run_id} ---")

        energy_file = os.path.join(scan_dir, f"energy_{run_id}")

        steady_found = False
        osc_found = False
        start_time = time.time()

        while True:
            if time.time() - start_time > max_runtime:
                print(f"[Timeout] Run {run_id}")
                break

            if not os.path.exists(energy_file):
                print(f"Waiting for {energy_file} ...")
                time.sleep(poll_interval)
                continue

            try:
                data = np.genfromtxt(energy_file)
                time_arr = data[:,0]
                W = data[:,1]
                dWdt = data[:,2]
            except:
                time.sleep(poll_interval)
                continue

            if len(time_arr) < 10:
                time.sleep(poll_interval)
                continue

            eps = 1e-12
            gamma = dWdt / (W + eps)
            fs = 1.0 / np.mean(np.diff(time_arr))

            steady_t, _ = detect_steady_stabilization(time_arr, gamma)
            osc_t, _ = detect_oscillatory_stabilization(time_arr, gamma, fs)

            if steady_t and not steady_found:
                steady_found = True
                save_plot(time_arr, gamma, steady_t, None, run_id)
                stop(run_id)
                break

            if osc_t and not osc_found:
                osc_found = True
                save_plot(time_arr, gamma, None, osc_t, run_id)
                stop(run_id)
                break

            time.sleep(poll_interval)

    print("\n===== All scan runs processed =====")



if __name__ == "__main__":
    monitor()
