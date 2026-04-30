#!/usr/bin/env python3
"""Visualisierung für cmgan-george-lwf train_log.txt (ähnlich visualize.ipynb).

Läuft rekursiv über `runs/*/train_log.txt` unter dem aktuellen Ordner
und erzeugt pro Run ein 2x2-Plot (PESQ, STOI, SI-SDR, DNSMOS_OVR)
und ein kombiniertes PESQ-Plot aller Runs. Ergebnisse landen in `figures/`.
"""
import re
import glob
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def _extract_metric(line, patterns):
    for p in patterns:
        m = re.search(rf"{p}.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def load_training_curve(log_file: str):
    metrics = {
        "epoch": [],
        "train_loss": [],
        "mse": [],
        "pesq": [],
        "stoi": [],
        "si_sdr": [],
        "dnsmos_sig": [],
        "dnsmos_bak": [],
        "dnsmos_ovr": [],
    }

    patterns = {
        "train_loss": [r"train\s+loss"],
        "mse": [r"valid\s+mse\s+distance", r"valid[_\s-]*mse"],
        "pesq": [r"valid[_\s-]*pesq", r"pesq"],
        "stoi": [r"valid[_\s-]*stoi", r"stoi"],
        "si_sdr": [r"valid[_\s-]*si[-_\s]*sdr", r"valid[_\s-]*si[-_\s]*snr"],
        "dnsmos_sig": [r"dnsmos[_\s-]*sig"],
        "dnsmos_bak": [r"dnsmos[_\s-]*bak"],
        "dnsmos_ovr": [r"dnsmos[_\s-]*ovr", r"dnsmos[_\s-]*overall"],
    }

    epoch_re = re.compile(r"Epoch\s*[:=]\s*(\d+)", flags=re.IGNORECASE)

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = epoch_re.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            metrics["epoch"].append(epoch)
            for key in [k for k in patterns.keys()]:
                val = _extract_metric(line, patterns[key])
                metrics[key].append(val)

    # If no epochs found, return empty
    if not metrics["epoch"]:
        return None

    return metrics


def safe_name(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|()\s,=]+", "_", s)


def plot_run(curves: dict, label: str, out_dir: Path):
    keys = ["pesq", "stoi", "si_sdr", "dnsmos_ovr"]
    titles = ["Valid PESQ", "Valid STOI", "Valid SI-SDR", "Valid DNSMOS (OVR)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    axes = axes.ravel()

    for ax, key, title in zip(axes, keys, titles):
        y = curves.get(key, [])
        x = curves.get("epoch", [])
        # filter None
        x_plot = [e for e, v in zip(x, y) if v is not None]
        y_plot = [v for v in y if v is not None]
        if x_plot:
            ax.plot(x_plot, y_plot, marker="o", linewidth=1.6, markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(alpha=0.25)
        else:
            ax.set_title(f"{title} (nicht gefunden)")
            ax.axis("off")

    fig.suptitle(f"Training Curves — {label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = out_dir / f"training_curves_{safe_name(label)}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_png}")


def plot_combined_pesq(all_curves: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 6), dpi=120)
    found = False
    for label, curves in all_curves.items():
        y = curves.get("pesq", [])
        x = curves.get("epoch", [])
        if not any(v is not None for v in y):
            continue
        found = True
        x_plot = [e for e, v in zip(x, y) if v is not None]
        y_plot = [v for v in y if v is not None]
        ax.plot(x_plot, y_plot, marker="o", linewidth=1.8, markersize=3, label=label)

    if not found:
        print("Keine PESQ-Daten für kombinierte Ansicht gefunden.")
        return

    ax.set_title("Valid PESQ — alle Runs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PESQ")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = out_dir / "training_curves_pesq_combined.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_png}")


def main(base_folder: str):
    base = Path(base_folder)
    search_pattern = str(base / "runs" / "**" / "train_log.txt")
    hits = sorted(glob.glob(search_pattern, recursive=True))
    if not hits:
        # fallback: if user passed a single train_log.txt
        candidate = Path(base_folder)
        if candidate.is_file() and candidate.name == "train_log.txt":
            hits = [str(candidate)]

    if not hits:
        print(f"Keine train_log.txt unter {base_folder}/runs gefunden.")
        return

    out_dir = Path(base_folder) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    for log in hits:
        curves = load_training_curve(log)
        if curves is None:
            print(f"Keine Epochenzeilen in {log} gefunden — übersprungen.")
            continue
        rel = os.path.relpath(os.path.dirname(log), start=base_folder)
        label = rel
        all_curves[label] = curves
        plot_run(curves, label, out_dir)

    plot_combined_pesq(all_curves, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cmgan-george-lwf runs")
    parser.add_argument("base", nargs="?", default=".", help="Base folder (default: current folder)")
    args = parser.parse_args()
    main(args.base)
