"""Reproduce key numerical results from the BAAIWorm release.

This script intentionally uses only NumPy and Matplotlib. It validates the
published artifacts without requiring NEURON, CUDA, OptiX, or the C++ GUI.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_correlation_artifacts() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    trial = ROOT / "eworm_learn" / "trial10"
    data = ROOT / "eworm_learn" / "components" / "cb2022_data"
    config = json.loads((trial / "000_circuit_search_config.json").read_text())
    allowed = set(config["search_config"]["output_cell_names"])
    all_names = (data / "Ca_corr_mat_cell_name.txt").read_text().split("\t")
    ids = [i for i, name in enumerate(all_names) if name in allowed]
    names = [all_names[i] for i in ids]
    target_all = np.loadtxt(data / "Ca_corr_mat.txt")
    target = target_all[np.ix_(ids, ids)]
    initial_v = np.load(trial / "v_initial_eworm_v4.npy")
    final_v = np.load(trial / "v_final_eworm_v4.npy")
    return target, np.corrcoef(initial_v), np.corrcoef(final_v), names


def correlation_reproduction(out_dir: Path) -> dict[str, float]:
    target, initial, final, names = load_correlation_artifacts()
    initial_mse = float(np.nanmean((initial - target) ** 2))
    final_mse = float(np.nanmean((final - target) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        (target, initial, final),
        ("Experimental target", f"Before optimization (MSE={initial_mse:.4f})",
         f"Released final voltages (MSE={final_mse:.4f})"),
    ):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Neuron index")
    fig.colorbar(image, ax=axes, label="Pearson correlation", shrink=0.85)
    fig.savefig(out_dir / "correlation_reproduction.png", dpi=180)
    plt.close(fig)

    centered = np.load(ROOT / "eworm_learn" / "trial10" / "v_final_eworm_v4.npy")
    centered = centered - centered.mean(axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = (s**2) / np.sum(s**2)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.8)
    for i, name in enumerate(names):
        if name.startswith(("AV", "VA", "DA", "VB", "DB")):
            ax.annotate(name, scores[i], fontsize=6, alpha=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("PCA of released 65-neuron membrane potentials")
    fig.savefig(out_dir / "pca_reproduction.png", dpi=180)
    plt.close(fig)

    return {
        "correlation_initial_mse": initial_mse,
        "correlation_final_mse": final_mse,
        "paper_reported_mse": 0.076,
        "pc1_explained_variance": float(explained[0]),
        "pc2_explained_variance": float(explained[1]),
    }


def reservoir_reproduction(out_dir: Path) -> dict[str, float]:
    base = ROOT / "eworm" / "ghost_in_mesh_sim" / "data" / "tuned" / "video_offline"
    with (base / "video_offline_model_output_soma.pkl").open("rb") as handle:
        voltage = np.asarray(pickle.load(handle))
    target = np.load(base / "gt_eworm.muscle-220428-152601.npy").T

    # Paper/code settings: 5/3 ms neural step, 100 ms body step, 40 samples washout.
    dt_scale = 60
    voltage_100ms = voltage.reshape(80, 300, dt_scale).mean(axis=2)
    target_voltage = target * 100.0 - 80.0
    washout, alpha = 40, 1e-3
    x = voltage_100ms[:, washout:].T
    y = target_voltage[:, washout:].T
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    predicted = ((voltage_100ms.T @ weights).T + 80.0) / 100.0

    released_prediction = np.load(base / "video_offline_eworm.muscle-220428-152601.npy").T
    # The release omits 24 frames and applies (+80)/100 twice before saving
    # (pre_interaction.py lines 413 and 447). Undo the second scaling here.
    released_prediction_unscaled = released_prediction * 100.0 - 80.0
    overlap = predicted[:, 24:]
    released_rmse = float(np.sqrt(np.mean((overlap - released_prediction_unscaled) ** 2)))
    train_rmse = float(np.sqrt(np.mean((predicted[:, washout:] - target[:, washout:]) ** 2)))
    per_muscle_corr = [
        np.corrcoef(predicted[i, washout:], target[i, washout:])[0, 1]
        for i in range(target.shape[0])
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    extent = (0, 30, 96, 0)
    axes[0].imshow(target, aspect="auto", cmap="turbo", extent=extent)
    axes[0].set_title("Target activation of 96 muscles")
    axes[1].imshow(predicted, aspect="auto", cmap="turbo", extent=extent)
    axes[1].set_title("Recomputed 80-neuron to 96-muscle reservoir output")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Muscle index")
    fig.savefig(out_dir / "reservoir_reproduction.png", dpi=180)
    plt.close(fig)

    return {
        "reservoir_train_rmse_activation": train_rmse,
        "reservoir_median_muscle_correlation": float(np.nanmedian(per_muscle_corr)),
        "released_prediction_rmse_after_undoing_duplicate_scale": released_rmse,
        "reservoir_weight_frobenius_norm": float(np.linalg.norm(weights)),
    }


def single_neuron_reproduction(out_dir: Path) -> dict[str, dict[str, float]]:
    base = ROOT / "eworm" / "model_figure" / "single_neuron"
    neuron_names = ("AIYL", "AVAL", "AWAL", "AWCL", "RIML", "VD05")
    voltage_clamp = {"AWCL", "VD05"}
    metrics: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(len(neuron_names), 2, figsize=(12, 17), constrained_layout=True)

    for row, name in enumerate(neuron_names):
        sim_path = base / "simulation_data" / f"{name}_simulation_trace.pkl"
        exp_path = base / "electrophysiology_data" / f"{name}_electrophysiology_trace.pkl"
        with sim_path.open("rb") as handle:
            sim = pickle.load(handle)
        with exp_path.open("rb") as handle:
            exp = pickle.load(handle)
        sim_y, sim_t = np.asarray(sim["voltage"], float), np.asarray(sim["time"], float)
        exp_y, exp_t = np.asarray(exp["voltage"], float), np.asarray(exp["time"], float)
        if sim_t.ndim == 2:
            sim_t = sim_t[0]
        scale = 1000.0 if name in voltage_clamp else 1.0

        ax = axes[row, 0]
        for trace in exp_y:
            ax.plot(exp_t, trace * scale, color="0.72", lw=0.7)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, sim_y.shape[0]))
        for trace, color in zip(sim_y, colors):
            ax.plot(sim_t, trace * scale, color=color, lw=0.8)
        ax.set_title(f"{name}: experiment (gray) and model")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (pA)" if name in voltage_clamp else "Voltage (mV)")

        # Same steady-state interval used by the authors' figure notebook.
        sim_slice = slice(int(sim_y.shape[1] * 4 / 7), int(sim_y.shape[1] * 5.9 / 7))
        exp_slice = slice(int(exp_y.shape[1] * 4 / 7), int(exp_y.shape[1] * 5.9 / 7))
        sim_steady = sim_y[:, sim_slice].mean(axis=1) * scale
        exp_steady = exp_y[:, exp_slice].mean(axis=1) * scale
        stim = np.asarray(sim["stim"], float)
        stim_level = np.median(stim[:, stim.shape[1] // 2], axis=0) if stim.ndim == 1 else np.median(
            stim[:, stim.shape[1] // 2 : stim.shape[1] // 2 + 1], axis=1
        )
        ax = axes[row, 1]
        ax.plot(stim_level, exp_steady, "o-", color="tab:red", label="experiment", ms=4)
        ax.plot(stim_level, sim_steady, "*-", color="black", label="model", ms=5)
        ax.set_title(f"{name}: steady-state response")
        ax.set_xlabel("Voltage (mV)" if name in voltage_clamp else "Current (pA)")
        ax.set_ylabel("Current (pA)" if name in voltage_clamp else "Voltage (mV)")
        ax.legend(frameon=False, fontsize=8)

        rmse = float(np.sqrt(np.mean((sim_steady - exp_steady) ** 2)))
        corr = float(np.corrcoef(sim_steady, exp_steady)[0, 1])
        metrics[name] = {"steady_state_rmse": rmse, "steady_state_correlation": corr}

    fig.savefig(out_dir / "single_neuron_reproduction.png", dpi=180)
    plt.close(fig)
    return metrics


def circuit_reproduction(out_dir: Path) -> dict[str, float | int]:
    # Pickles refer to the repository's lightweight abstract_circuit module.
    sys.path.insert(0, str(ROOT / "eworm" / "ghost_in_mesh_sim"))
    with (ROOT / "eworm_learn" / "trial10" / "optimal_abs_circuit.pkl").open("rb") as handle:
        circuit = pickle.load(handle)
    gap = np.asarray([c.weight * 1e4 for c in circuit.connections if c.category == "gj"])
    exc = np.asarray([c.weight * 4.9 for c in circuit.connections if c.category == "syn" and c.weight > 0])
    inh = np.asarray([c.weight * 2.0 for c in circuit.connections if c.category == "syn" and c.weight < 0])
    syn = np.concatenate((exc, inh))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(gap, bins=np.arange(0, max(4.5, gap.max() + 0.5), 0.5), edgecolor="black")
    axes[0].set_title(f"Gap junctions (n={gap.size})")
    axes[0].set_xlabel("Scaled conductance")
    axes[0].set_ylabel("Count")
    axes[1].hist(syn, bins=20, edgecolor="black", color="tab:orange")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title(f"Graded synapses (n={syn.size})")
    axes[1].set_xlabel("Scaled signed conductance")
    axes[1].set_ylabel("Count")
    fig.savefig(out_dir / "connection_weight_reproduction.png", dpi=180)
    plt.close(fig)
    return {
        "neurons": len(circuit.cells),
        "connections_total": len(circuit.connections),
        "gap_junctions": int(gap.size),
        "synapses": int(syn.size),
        "excitatory_synapses": int(exc.size),
        "inhibitory_synapses": int(inh.size),
        "excitatory_to_inhibitory_ratio": float(exc.size / inh.size),
    }


def body_kinematics_reproduction(out_dir: Path) -> dict[str, float | int]:
    state_path = (
        ROOT / "eworm" / "ghost_in_mesh_sim" / "data" / "state"
        / "worm_states_300_220428-152601.json"
    )
    state = json.loads(state_path.read_text())
    frames = np.asarray(state["Frames"], float)
    n_points = int(state["NumSamplings"])
    positions = frames[:, 6 : 6 + 3 * n_points].reshape(-1, n_points, 3)
    velocities = frames[:, 6 + 3 * n_points :].reshape(-1, n_points, 3)
    time = np.arange(frames.shape[0]) * 0.1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    axis_names = ("X: head-tail", "Y: left-right", "Z: dorsal-ventral")
    colors = plt.cm.turbo(np.linspace(0, 1, n_points))
    for dim in range(3):
        pos_scale = max(np.ptp(positions[:, :, dim]), 1e-8) * 0.16
        vel_scale = max(np.ptp(velocities[:, :, dim]), 1e-8) * 0.16
        for point, color in enumerate(colors):
            axes[0, dim].plot(time, positions[:, point, dim] + point * pos_scale,
                              color=color, lw=0.65)
            axes[1, dim].plot(time, velocities[:, point, dim] + point * vel_scale,
                              color=color, lw=0.65)
        axes[0, dim].set_title(axis_names[dim])
        axes[1, dim].set_xlabel("Time (s)")
        axes[0, dim].set_yticks([])
        axes[1, dim].set_yticks([])
    axes[0, 0].set_ylabel("Relative position\n(head to tail, offset)")
    axes[1, 0].set_ylabel("Relative velocity\n(head to tail, offset)")
    fig.savefig(out_dir / "body_kinematics_reproduction.png", dpi=180)
    plt.close(fig)

    component_velocity_rms = np.sqrt(np.mean(velocities**2, axis=(0, 1)))
    point_speed_rms = np.sqrt(np.mean(np.sum(velocities**2, axis=2), axis=0))
    return {
        "frames": int(frames.shape[0]),
        "sampled_body_points": n_points,
        "duration_seconds": float(time[-1] + 0.1),
        "velocity_rms_x": float(component_velocity_rms[0]),
        "velocity_rms_y": float(component_velocity_rms[1]),
        "velocity_rms_z": float(component_velocity_rms[2]),
        "head_speed_rms": float(point_speed_rms[0]),
        "center_speed_rms": float(point_speed_rms[n_points // 2]),
        "tail_speed_rms": float(point_speed_rms[-1]),
        "tail_to_head_speed_ratio": float(point_speed_rms[-1] / point_speed_rms[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "correlation_and_pca": correlation_reproduction(args.output),
        "reservoir_readout": reservoir_reproduction(args.output),
        "single_neuron_electrophysiology": single_neuron_reproduction(args.output),
        "optimized_circuit": circuit_reproduction(args.output),
        "body_kinematics": body_kinematics_reproduction(args.output),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
