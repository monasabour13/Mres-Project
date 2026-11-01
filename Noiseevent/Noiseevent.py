"""
Detect noise events in a raw audio file and label their sound source.

Step 1: find events from the A-weighted sound level (the method in the proposal):
        level > running background (L90) + threshold, for at least a minimum time.
Step 2: label each event with a pretrained audio model.

AudioFM (arXiv 2610.34183) has no public code or weights yet (October 2026).
PANNs (Kong et al., 2020) is used here as a stand-in. It is also trained on
AudioSet and gives frame-level outputs for 527 sound classes. When AudioFM is
released, only step 2 (label_events) needs to change.

Usage:
    python detect_noise_events.py data/traffic.wav --ckpt <path to .pth> --out results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_SR = 32000       # PANNs works at 32 kHz
FRAME_S = 0.125        # 125 ms frames, close to the "Fast" (LAF) time weighting
CHUNK_S = 60           # run the model on 60 s chunks, so full nights fit in memory

# AudioSet classes that matter for a bedroom at night. Traffic classes come first.
TRAFFIC_CLASSES = [
    "Traffic noise, roadway noise", "Vehicle", "Motor vehicle (road)", "Car",
    "Car passing by", "Truck", "Bus", "Motorcycle", "Vehicle horn, car horn, honking",
    "Accelerating, revving, vroom", "Tire squeal", "Emergency vehicle", "Siren",
]
OTHER_CLASSES = ["Speech", "Dog", "Bird", "Door", "Snoring", "Train", "Rail transport", "Aircraft"]


# ---------------------------------------------------------------- step 1: level
def a_weight_db(f):
    """IEC 61672 A-weighting gain in dB for frequencies f (Hz)."""
    f2 = f ** 2
    ra = (12194 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
    )
    return 20 * np.log10(np.maximum(ra, 1e-20)) + 2.0


def a_weighted_level(x, sr, calib_db):
    """A-weighted level per 125 ms frame, in dB re full scale + calibration offset."""
    n = int(FRAME_S * sr)
    n_frames = len(x) // n
    win = np.hanning(n)
    frames = x[: n_frames * n].reshape(n_frames, n) * win
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    gain = 10 ** (a_weight_db(np.fft.rfftfreq(n, 1 / sr)) / 10)
    gain[0] = 0.0
    mean_square = (power * gain).sum(axis=1) * 2 / (n * (win ** 2).sum())
    return 10 * np.log10(mean_square + 1e-12) + calib_db


def find_events(level, bg, threshold_db, min_dur_s, merge_gap_s):
    """Return (start, end) frame indices where level stays above bg + threshold."""
    above = level > bg + threshold_db
    edges = np.diff(np.r_[0, above.astype(int), 0])
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))

    merged = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) * FRAME_S < merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) * FRAME_S >= min_dur_s]


def describe_events(runs, level, bg):
    rows = []
    for i, (s, e) in enumerate(runs):
        peak = s + int(np.argmax(level[s:e]))
        background = float(np.median(bg[s:e]))
        rows.append({
            "event": i + 1,
            "onset_s": s * FRAME_S,
            "offset_s": e * FRAME_S,
            "duration_s": (e - s) * FRAME_S,
            "LAFmax_dB": float(level[peak]),
            "background_L90_dB": background,
            "emergence_dB": float(level[peak]) - background,
            "rise_time_s": (peak - s) * FRAME_S,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 2: model
def run_model(x32, ckpt, device):
    """Frame-level class probabilities (T x 527) and their times in seconds."""
    from panns_inference import SoundEventDetection, labels

    sed = SoundEventDetection(checkpoint_path=str(ckpt), device=device)
    chunk = CHUNK_S * MODEL_SR
    probs, times = [], []
    for start in range(0, len(x32), chunk):
        seg = x32[start:start + chunk]
        if len(seg) < MODEL_SR:          # skip a tail shorter than 1 s
            break
        out = sed.inference(seg[None, :].astype(np.float32))[0]
        probs.append(out)
        times.append(start / MODEL_SR + np.arange(len(out)) * (len(seg) / MODEL_SR) / len(out))
    return np.concatenate(probs), np.concatenate(times), list(labels)


def label_events(events, probs, times, labels):
    """For each event, take the max probability of each class inside the event."""
    idx = {name: labels.index(name) for name in TRAFFIC_CLASSES + OTHER_CLASSES if name in labels}
    traffic = [c for c in TRAFFIC_CLASSES if c in idx]

    extra = []
    for _, ev in events.iterrows():
        inside = (times >= ev.onset_s) & (times < ev.offset_s)
        p = probs[inside].max(axis=0)
        row = {f"p_{name}": float(p[i]) for name, i in idx.items()}
        best = max(idx, key=lambda c: p[idx[c]])
        row["source_label"] = best
        row["source_prob"] = float(p[idx[best]])
        row["is_road_traffic"] = best in traffic
        row["traffic_prob"] = float(max(p[idx[c]] for c in traffic))
        top3 = np.argsort(p)[::-1][:3]
        row["top3_all_classes"] = "; ".join(f"{labels[k]} ({p[k]:.2f})" for k in top3)
        extra.append(row)
    return pd.concat([events, pd.DataFrame(extra)], axis=1)


# ---------------------------------------------------------------- plot
def plot(x, sr, level, bg, threshold_db, events, probs, times, labels, out_png):
    t_level = np.arange(len(level)) * FRAME_S
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax[0].specgram(x, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    ax[0].set_ylim(0, 8000)
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title("Spectrogram")

    ax[1].plot(t_level, level, lw=0.8, label="LAF (125 ms)")
    ax[1].plot(t_level, bg, lw=1.5, label="Background L90")
    ax[1].plot(t_level, bg + threshold_db, "--", lw=1, label=f"L90 + {threshold_db:g} dB")
    for _, ev in events.iterrows():
        ax[1].axvspan(ev.onset_s, ev.offset_s, color="tab:red", alpha=0.2)
        ax[1].text(ev.onset_s, ev.LAFmax_dB, str(ev.event), fontsize=8)
    ax[1].set_ylabel("dB(A) (uncalibrated)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("Step 1: level-based events (red)")

    for name in ["Traffic noise, roadway noise", "Car", "Truck", "Motorcycle", "Speech"]:
        if name in labels:
            ax[2].plot(times, probs[:, labels.index(name)], lw=0.8, label=name)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlim(0, len(x) / sr)
    ax[2].set_ylabel("Probability")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_title("Step 2: model output per frame (PANNs, stand-in for AudioFM)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--ckpt", required=True, help="PANNs Cnn14_DecisionLevelMax checkpoint (.pth)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold-db", type=float, default=3.0, help="dB above background L90")
    ap.add_argument("--min-dur", type=float, default=1.0, help="minimum event duration (s)")
    ap.add_argument("--merge-gap", type=float, default=1.0, help="merge events closer than this (s)")
    ap.add_argument("--bg-window", type=float, default=300.0, help="L90 window (s); 300 = 5 min")
    ap.add_argument("--calib-db", type=float, default=0.0,
                    help="offset to convert dB re full scale to dB SPL (from a calibrator tone)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                   # mono
    print(f"Loaded {args.audio}: {len(x) / sr:.1f} s at {sr} Hz")

    # Step 1
    level = a_weighted_level(x, sr, args.calib_db)
    bg_frames = max(1, int(args.bg_window / FRAME_S))
    bg = pd.Series(level).rolling(bg_frames, center=True, min_periods=1).quantile(0.10).to_numpy()
    runs = find_events(level, bg, args.threshold_db, args.min_dur, args.merge_gap)
    events = describe_events(runs, level, bg)
    print(f"Step 1: {len(events)} level-based events")

    # Step 2
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    x32 = resample_poly(x, MODEL_SR, sr) if sr != MODEL_SR else x
    probs, times, labels = run_model(x32, args.ckpt, args.device)
    if len(events):
        events = label_events(events, probs, times, labels)

    events.to_csv(out / "events.csv", index=False)
    plot(x, sr, level, bg, args.threshold_db, events, probs, times, labels, out / "events_plot.png")

    cols = ["event", "onset_s", "duration_s", "LAFmax_dB", "emergence_dB", "rise_time_s",
            "source_label", "source_prob", "is_road_traffic"]
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(events[[c for c in cols if c in events]].to_string(index=False))
    print(f"Saved {out / 'events.csv'} and {out / 'events_plot.png'}")


if __name__ == "__main__":
    main()
