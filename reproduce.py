"""Reproduce key numerical results from the BAAIWorm release.

This script intentionally uses only NumPy and Matplotlib. It validates the
published artifacts without requiring NEURON, CUDA, OptiX, or the C++ GUI.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
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


def correlation_reproduction(out_dir: Path, source_data_root: Path | None = None) -> dict[str, float]:
    target, initial, final, names = load_correlation_artifacts()
    initial_mse = float(np.nanmean((initial - target) ** 2))
    final_mse = float(np.nanmean((final - target) ** 2))

    official = None
    official_mse = None
    if source_data_root is not None:
        official_path = source_data_root / "Supplementary Figure 3" / "control.npy"
        if official_path.exists():
            official = np.corrcoef(np.load(official_path).astype(float))
            official_mse = float(np.nanmean((official - target) ** 2))

    matrices = [target, initial, final]
    titles = ["Experimental target",
              f"Before optimization (MSE={initial_mse:.4f})",
              f"Repository final trace (MSE={final_mse:.4f})"]
    if official is not None:
        matrices.append(official)
        titles.append(f"Official Source Data control (MSE={official_mse:.4f})")
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.6), constrained_layout=True)
    for ax, matrix, title in zip(
        axes, matrices, titles,
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

    metrics = {
        "correlation_initial_mse": initial_mse,
        "correlation_final_mse": final_mse,
        "paper_reported_mse": 0.076,
        "pc1_explained_variance": float(explained[0]),
        "pc2_explained_variance": float(explained[1]),
    }
    if official_mse is not None:
        metrics["official_source_control_mse"] = official_mse
        metrics["absolute_difference_from_paper_mse"] = abs(official_mse - 0.076)
    return metrics


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


def _hoc_morphology(neuron_name: str) -> dict[str, np.ndarray]:
    """Extract section point coordinates directly from an exported NEURON HOC file."""
    text = (ROOT / "eworm" / "components" / "model" / f"{neuron_name}.hoc").read_text()
    sections: dict[str, list[list[float]]] = {}
    for match in re.finditer(
        r"(?m)^\s*(\w+)\s*\{\s*pt3dadd\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,"
        r"\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)\s*\}", text
    ):
        section = match.group(1)
        sections.setdefault(section, []).append([float(match.group(i)) for i in range(2, 6)])
    return {name: np.asarray(points) for name, points in sections.items()}


def supplementary1_electrophysiology_reproduction(out_dir: Path) -> dict[str, object]:
    """Rebuild Supplementary Fig. 1: morphology, traces, steady and peak I-V curves."""
    base = ROOT / "eworm" / "model_figure" / "single_neuron"
    neuron_names = ("AWCL", "AIYL", "AVAL", "RIML", "VD05")
    display_names = ("AWC(L)", "AIY(L)", "AVA(L)", "RIM(L)", "VD5")
    voltage_clamp = {"AWCL", "VD05"}
    fig, axes = plt.subplots(len(neuron_names), 4, figsize=(18, 18), constrained_layout=True)
    metrics: dict[str, object] = {}

    for row, (name, display_name) in enumerate(zip(neuron_names, display_names)):
        with (base / "simulation_data" / f"{name}_simulation_trace.pkl").open("rb") as handle:
            sim = pickle.load(handle)
        with (base / "electrophysiology_data" / f"{name}_electrophysiology_trace.pkl").open("rb") as handle:
            exp = pickle.load(handle)
        sim_y, exp_y = np.asarray(sim["voltage"], float), np.asarray(exp["voltage"], float)
        sim_t, exp_t = np.asarray(sim["time"], float), np.asarray(exp["time"], float)
        if sim_t.ndim == 2:
            sim_t = sim_t[0]
        scale = 1000.0 if name in voltage_clamp else 1.0
        stim = np.asarray(sim["stim"], float)
        stim_level = stim[:, stim.shape[1] // 2]

        ax = axes[row, 0]
        morphology = _hoc_morphology(name)
        for section, points in morphology.items():
            ax.plot(points[:, 0], points[:, 1], color="tab:blue",
                    lw=3.0 if "Soma" in section else 1.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        ax.text(-0.08, 0.5, display_name, transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=12, fontweight="bold")

        ax = axes[row, 1]
        for trace in exp_y:
            ax.plot(exp_t, trace * scale, color="0.65", lw=0.8)
        for trace in sim_y:
            ax.plot(sim_t, trace * scale, color="tab:blue", lw=0.8, alpha=0.85)
        ax.set(xlabel="Time (s)", ylabel="Current (pA)" if name in voltage_clamp else "Voltage (mV)")

        sim_steady_slice = slice(int(sim_y.shape[1] * 4 / 7), int(sim_y.shape[1] * 5.9 / 7))
        exp_steady_slice = slice(int(exp_y.shape[1] * 4 / 7), int(exp_y.shape[1] * 5.9 / 7))
        sim_steady = sim_y[:, sim_steady_slice].mean(axis=1) * scale
        exp_steady = exp_y[:, exp_steady_slice].mean(axis=1) * scale

        changed = np.any(np.abs(stim - stim[:, :1]) > 1e-9, axis=0)
        indices = np.flatnonzero(changed)
        onset, offset = int(indices[0]), int(indices[-1] + 1)
        early_end = min(offset, onset + max(4, int((offset - onset) * 0.10)))
        exp_onset = int(onset / sim_y.shape[1] * exp_y.shape[1])
        exp_early_end = int(early_end / sim_y.shape[1] * exp_y.shape[1])

        def initial_peak(values: np.ndarray, start: int, end: int) -> np.ndarray:
            baseline = values[:, max(0, start - 20):start].mean(axis=1)
            segment = values[:, start:end]
            peak_index = np.abs(segment - baseline[:, None]).argmax(axis=1)
            return segment[np.arange(segment.shape[0]), peak_index] * scale

        sim_peak = initial_peak(sim_y, onset, early_end)
        exp_peak = initial_peak(exp_y, exp_onset, max(exp_onset + 2, exp_early_end))
        x_values = stim_level
        xlabel = "Voltage (mV)" if name in voltage_clamp else "Current (pA)"
        ylabel = "Current (pA)" if name in voltage_clamp else "Voltage (mV)"
        for column, model_values, experimental_values, title in (
            (2, sim_steady, exp_steady, "Steady-state I-V"),
            (3, sim_peak, exp_peak, "Initial-peak I-V"),
        ):
            ax = axes[row, column]
            ax.plot(x_values, experimental_values, "o-", color="tab:red", ms=4, label="Experiment")
            ax.plot(x_values, model_values, "*-", color="black", ms=5, label="Model")
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if row == 0:
                ax.set_title(title)
            if row == 0 and column == 3:
                ax.legend(frameon=False, fontsize=8)

        metrics[name] = {
            "morphology_sections": len(morphology),
            "morphology_points": int(sum(len(points) for points in morphology.values())),
            "steady_state_rmse": float(np.sqrt(np.mean((sim_steady - exp_steady) ** 2))),
            "steady_state_correlation": float(np.corrcoef(sim_steady, exp_steady)[0, 1]),
            "initial_peak_rmse": float(np.sqrt(np.mean((sim_peak - exp_peak) ** 2))),
            "initial_peak_correlation": float(np.corrcoef(sim_peak, exp_peak)[0, 1]),
        }

    for column, title in enumerate(("Morphology", "Response to stimuli", "Steady-state I-V", "Initial-peak I-V")):
        axes[0, column].set_title(title)
    fig.savefig(out_dir / "supplementary1_electrophysiology_reproduction.png", dpi=180)
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


def perturbation_reproduction(data_dir: Path, out_dir: Path) -> dict[str, dict[str, object]]:
    """Reproduce Fig. 6 from the official Zenodo Source Data."""
    cases = (
        ("control", "Control", "control_0.pkl"),
        ("remove_neurite", "Remove neurite", "remove_neurite_64.pkl"),
        ("shuffle_location", "Shuffle locations", "shuffle_location_64.pkl"),
        ("shuffle_syn", "Shuffle synapse weights", "shuffle_weight_syn_64.pkl"),
        ("shuffle_gj", "Shuffle gap-junction weights", "shuffle_weight_gj_64.pkl"),
        ("remove_syn", "Remove synapses", "remove_syn_64.pkl"),
        ("remove_gj", "Remove gap junctions", "remove_gj_64.pkl"),
    )
    corr_dir = data_dir / "corr_map"
    pos_dir = data_dir / "relative_pos"
    vel_dir = data_dir / "relative_vel"
    control_corr = np.corrcoef(np.load(corr_dir / "control.npy"))
    metrics: dict[str, dict[str, object]] = {}
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 22), constrained_layout=True)
    time = np.arange(200) * 0.1

    for row, (key, label, stem) in enumerate(cases):
        voltages = np.load(corr_dir / f"{key}.npy")
        corr = np.corrcoef(voltages)
        pos = np.loadtxt(pos_dir / f"{stem}_rl.txt", delimiter=",")
        vel = np.loadtxt(vel_dir / f"{stem}_rv.txt", delimiter=",")

        axes[row, 0].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for index, part in enumerate(("Head", "Center", "Tail")):
            axes[row, 1].plot(time, pos[:, index], label=part, lw=0.9)
            axes[row, 2].plot(time, vel[:, index], label=part, lw=0.9)
        axes[row, 1].set_ylim(bottom=0)
        axes[row, 2].set_ylim(bottom=0)
        if row == 0:
            axes[row, 0].set_title("Neural correlation matrix")
            axes[row, 1].set_title("Relative position magnitude")
            axes[row, 2].set_title("Relative velocity magnitude")
            axes[row, 1].legend(frameon=False, ncol=3, fontsize=8)
            axes[row, 2].legend(frameon=False, ncol=3, fontsize=8)
        if row == len(cases) - 1:
            axes[row, 1].set_xlabel("Time (s)")
            axes[row, 2].set_xlabel("Time (s)")

        metrics[key] = {
            "correlation_mse_vs_control": float(np.mean((corr - control_corr) ** 2)),
            "mean_relative_position_head_center_tail": [float(x) for x in pos.mean(axis=0)],
            "mean_relative_velocity_head_center_tail": [float(x) for x in vel.mean(axis=0)],
            "tail_velocity_change_vs_control": None,
        }

    control_tail = metrics["control"]["mean_relative_velocity_head_center_tail"][2]
    for values in metrics.values():
        tail = values["mean_relative_velocity_head_center_tail"][2]
        values["tail_velocity_change_vs_control"] = float((tail / control_tail) - 1.0)
    fig.savefig(out_dir / "figure6_perturbation_reproduction.png", dpi=180)
    plt.close(fig)
    return metrics


def figure4_source_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    position = np.stack([np.loadtxt(data_dir / f"pos_{axis}.txt", delimiter=",")
                         for axis in "xyz"], axis=-1)
    velocity = np.stack([np.loadtxt(data_dir / f"vel_{axis}.txt", delimiter=",")
                         for axis in "xyz"], axis=-1)
    time = np.arange(position.shape[1]) * 0.1
    pos_mag = np.linalg.norm(position, axis=-1)
    vel_mag = np.linalg.norm(velocity, axis=-1)
    colors = plt.cm.turbo(np.linspace(0, 1, position.shape[0]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for index, color in enumerate(colors):
        axes[0].plot(time, pos_mag[index], color=color, lw=0.9)
        axes[1].plot(time, vel_mag[index], color=color, lw=0.9)
    axes[0].set(title="Figure 4 official body-point displacement", ylabel="Relative position magnitude")
    axes[1].set(title="Head-to-tail propagation in velocity", xlabel="Time (s)",
                ylabel="Relative velocity magnitude")
    fig.savefig(out_dir / "figure4_source_reproduction.png", dpi=180)
    plt.close(fig)
    rms_speed = np.sqrt(np.mean(vel_mag**2, axis=1))
    return {"samples": int(position.shape[1]), "body_points": int(position.shape[0]),
            "head_rms_speed": float(rms_speed[0]), "tail_rms_speed": float(rms_speed[-1]),
            "tail_to_head_rms_speed_ratio": float(rms_speed[-1] / rms_speed[0])}


def _load_trailing_comma_matrix(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",")
    if values.ndim == 2:
        values = values[:, ~np.all(np.isnan(values), axis=0)]
    return values


def _spectral_concentration(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0))[1:] ** 2
    spectrum = power.mean(axis=1)
    return float(spectrum.max() / spectrum.sum())


def loop_dynamics_reproduction(closed_dir: Path, open_dir: Path, out_dir: Path) -> dict[str, object]:
    cases = []
    for label, directory, prefix in (("Closed loop", closed_dir, "online"),
                                     ("Open loop", open_dir, "offline")):
        cases.append((label,
                      _load_trailing_comma_matrix(directory / f"{prefix}_input.txt"),
                      _load_trailing_comma_matrix(directory / f"{prefix}_motor_neuron.txt"),
                      _load_trailing_comma_matrix(directory / f"{prefix}_muscle.txt")))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    metrics = {}
    for column, (label, stimulus, motor, muscle) in enumerate(cases):
        axes[0, column].plot(np.arange(stimulus.size) * 0.1, stimulus, color="black", lw=1)
        axes[0, column].set(title=f"{label}: input", ylabel="Input")
        axes[1, column].imshow(motor.T, aspect="auto", origin="lower", cmap="coolwarm")
        axes[1, column].set(title=f"{label}: 80 motor neurons", ylabel="Neuron")
        axes[2, column].imshow(muscle.T, aspect="auto", origin="lower", cmap="turbo")
        axes[2, column].set(title=f"{label}: 96 muscles", ylabel="Muscle", xlabel="Time sample")
        key = label.lower().replace(" ", "_")
        metrics[key] = {"samples": int(motor.shape[0]),
                        "motor_spectral_concentration": _spectral_concentration(motor),
                        "muscle_spectral_concentration": _spectral_concentration(muscle)}
    fig.savefig(out_dir / "closed_vs_open_loop_reproduction.png", dpi=180)
    plt.close(fig)
    return metrics


def _xlsx_numeric_column(path: Path, sheet_number: int) -> np.ndarray:
    """Read a numeric XLSX worksheet without adding a pandas dependency."""
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    values = []
    for cell in root.findall(".//x:c", namespace):
        value = cell.find("x:v", namespace)
        if value is not None and value.text is not None:
            values.append(float(value.text))
    return np.asarray(values)


def supplementary2_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    """Replot the chemical-synapse and gap-junction distance distributions."""
    workbook = data_dir / "syn_gj_dist.xlsx"
    chemical = _xlsx_numeric_column(workbook, 1)
    gap = _xlsx_numeric_column(workbook, 2)
    parameters = {"Chemical synapses": (chemical, 0.44, 0.63),
                  "Gap junctions": (gap, 0.70, 0.40)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.linspace(0.002, max(chemical.max(), gap.max()), 600)
    for ax, (label, (values, mu, shape)) in zip(axes, parameters.items()):
        density = np.sqrt(shape / (2 * np.pi * x**3)) * np.exp(
            -shape * (x - mu) ** 2 / (2 * mu**2 * x)
        )
        ax.hist(values, bins=45, density=True, alpha=0.55, color="tab:blue",
                label=f"Official data (n={values.size:,})")
        ax.plot(x, density, color="black", lw=2,
                label=fr"Inverse Gaussian ($\mu$={mu:.2f}, $\lambda$={shape:.2f})")
        ax.axvline(values.mean(), color="tab:red", ls="--", lw=1.3,
                   label=f"mean={values.mean():.3f}")
        ax.set(title=label, xlabel="Normalized connection location", ylabel="Probability density")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "supplementary2_connection_location_reproduction.png", dpi=180)
    plt.close(fig)

    return {
        "chemical_count": int(chemical.size),
        "chemical_mean_location": float(chemical.mean()),
        "chemical_median_location": float(np.median(chemical)),
        "chemical_inverse_gaussian_mu": 0.44,
        "chemical_inverse_gaussian_shape": 0.63,
        "gap_junction_count": int(gap.size),
        "gap_junction_mean_location": float(gap.mean()),
        "gap_junction_median_location": float(np.median(gap)),
        "gap_junction_inverse_gaussian_mu": 0.70,
        "gap_junction_inverse_gaussian_shape": 0.40,
    }


def supplementary9_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    """Quantify voltage propagation along the AVAL axon and two dendrites."""
    branches = (("Axon", "dynamic_trace_axon.pkl"),
                ("Dendrite 1", "dynamic_trace_dend1.pkl"),
                ("Dendrite 2", "dynamic_trace_dend2.pkl"))
    processed = []
    metrics = {}
    for label, filename in branches:
        with (data_dir / filename).open("rb") as handle:
            voltage = np.asarray(pickle.load(handle), dtype=np.float32)
        active = np.flatnonzero(np.any(voltage != 0, axis=1))
        voltage = voltage[active]
        baseline = voltage[:, :1000].mean(axis=1)
        peak = voltage.max(axis=1)
        amplitude = peak - baseline
        stride = 50
        processed.append((label, voltage[:, ::stride], peak, amplitude))
        key = label.lower().replace(" ", "_")
        metrics[key] = {
            "active_compartments": int(voltage.shape[0]),
            "soma_peak_mv": float(peak[0]),
            "distal_peak_mv": float(peak[-1]),
            "distal_to_soma_depolarization_ratio": float(amplitude[-1] / amplitude[0]),
        }
        del voltage

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (label, sampled, peak, amplitude) in enumerate(processed):
        time = np.linspace(0, 10, sampled.shape[1])
        image = axes[0, column].imshow(sampled, aspect="auto", origin="lower",
                                       extent=(0, 10, 0, sampled.shape[0] - 1),
                                       cmap="turbo", vmin=-35, vmax=15)
        axes[0, column].set(title=f"{label}: membrane voltage", xlabel="Time (s)",
                            ylabel="Compartment from soma")
        distance = np.linspace(0, 1, peak.size)
        axes[1, column].plot(distance, peak, label="Peak voltage", color="tab:red")
        axes[1, column].plot(distance, amplitude, label="Depolarization amplitude",
                             color="tab:blue")
        axes[1, column].set(title=f"{label}: spatial attenuation",
                            xlabel="Normalized distance from soma", ylabel="mV")
        axes[1, column].legend(frameon=False, fontsize=8)
    fig.colorbar(image, ax=axes[0, :], label="Membrane voltage (mV)", shrink=0.82)
    fig.savefig(out_dir / "supplementary9_neurite_propagation_reproduction.png", dpi=180)
    plt.close(fig)
    return metrics


def supplementary4_muscle_wave_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    """Reproduce the authors' cross-correlation estimate of muscle-wave delay."""
    muscle = _load_trailing_comma_matrix(data_dir / "online_muscle.txt")[300:601].T
    original_x = np.arange(muscle.shape[1])
    interpolated_x = np.arange(0, muscle.shape[1] - 1, 0.1)
    traces = np.asarray([np.interp(interpolated_x, original_x, trace) for trace in muscle])
    cross_correlations = np.asarray([
        np.correlate(traces[0], trace, mode="full") for trace in traces
    ])
    calculation_range = np.arange(2500, 3000)
    summed = []
    for offset in range(60):
        value = 0.0
        for quadrant in range(4):
            for muscle_index in range(24):
                value += cross_correlations[quadrant * 24 + muscle_index,
                                            calculation_range + muscle_index * offset].sum()
        summed.append(value)
    summed = np.asarray(summed)
    delays = np.arange(summed.size) * 0.01
    best_index = int(np.argmax(summed))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(muscle, aspect="auto", origin="lower", cmap="turbo",
                   extent=(0, 30, 0, 96))
    axes[0].set(title="Closed-loop muscle activation", xlabel="Time in analysis window (s)",
                ylabel="Muscle index")
    axes[1].plot(delays, summed, color="black", lw=1.8)
    axes[1].axvline(delays[best_index], color="tab:red", ls="--",
                    label=f"maximum at {delays[best_index]:.2f} s")
    axes[1].axhline(summed[best_index], color="tab:red", ls=":", alpha=0.7)
    axes[1].set(title="Summed cross-correlation", xlabel="Adjacent-muscle delay (s)",
                ylabel="Sum(cross-correlation)", xlim=(0, 0.5))
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "supplementary4_muscle_wave_delay_reproduction.png", dpi=180)
    plt.close(fig)
    return {
        "analysis_samples": int(muscle.shape[1]), "muscles": int(muscle.shape[0]),
        "interpolation_factor": 10,
        "best_adjacent_muscle_delay_seconds": float(delays[best_index]),
        "maximum_summed_cross_correlation": float(summed[best_index]),
    }


def supplementary7_readout_weights_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    """Visualize the released 80-motor-neuron to 96-muscle readout matrix."""
    with (data_dir / "video_online_wout.pkl").open("rb") as handle:
        weights = np.asarray(pickle.load(handle), dtype=float)
    limit = float(np.percentile(np.abs(weights), 99))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image = axes[0].imshow(weights, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set(title="Released closed-loop readout weights", xlabel="Muscle index",
                ylabel="Motor-neuron index")
    fig.colorbar(image, ax=axes[0], label="Weight", shrink=0.85)
    axes[1].hist(weights.ravel(), bins=80, color="tab:blue", alpha=0.75)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(title="Readout-weight distribution", xlabel="Weight", ylabel="Count")
    fig.savefig(out_dir / "supplementary7_readout_weights_reproduction.png", dpi=180)
    plt.close(fig)
    return {
        "motor_neurons": int(weights.shape[0]), "muscles": int(weights.shape[1]),
        "minimum_weight": float(weights.min()), "maximum_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "weight_standard_deviation": float(weights.std()),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "positive_fraction": float(np.mean(weights > 0)),
        "negative_fraction": float(np.mean(weights < 0)),
        "fraction_abs_weight_below_1": float(np.mean(np.abs(weights) < 1)),
    }


def supplementary6_multiseed_reproduction(data_dir: Path, out_dir: Path) -> dict[str, object]:
    """Aggregate closed-loop behavioral outcomes across all perturbation seeds."""
    groups = (
        ("control", "Control"), ("remove_neurite", "Remove neurites"),
        ("shuffle_location", "Shuffle locations"),
        ("shuffle_weight_syn", "Shuffle synaptic weights"),
        ("shuffle_weight_gj", "Shuffle gap-junction weights"),
        ("remove_syn", "Remove synapses"), ("remove_gj", "Remove gap junctions"),
    )
    raw: dict[str, list[dict[str, float]]] = {}
    for prefix, _ in groups:
        runs = []
        for path in sorted(data_dir.glob(f"{prefix}_*.pkl")):
            with path.open("rb") as handle:
                result = pickle.load(handle)
            behavior = np.asarray(result["behavior_value"], dtype=np.float32)
            world_head = np.asarray(result["world_head_location"], dtype=np.float32)
            speed = np.linalg.norm(np.moveaxis(behavior[3:6], 0, -1), axis=-1)
            head_speed, tail_speed = float(speed[:, 0].mean()), float(speed[:, -1].mean())
            runs.append({
                "seed": int(path.stem.rsplit("_", 1)[1]),
                "net_head_displacement": float(np.linalg.norm(world_head[-1] - world_head[0])),
                "head_path_length": float(np.linalg.norm(np.diff(world_head, axis=0), axis=1).sum()),
                "mean_head_speed": head_speed, "mean_tail_speed": tail_speed,
                "tail_to_head_speed_ratio": tail_speed / head_speed,
            })
            del result, behavior, world_head, speed
            gc.collect()
        raw[prefix] = runs

    metric_names = ("net_head_displacement", "mean_head_speed", "mean_tail_speed",
                    "tail_to_head_speed_ratio")
    control = {name: raw["control"][0][name] for name in metric_names}
    summary: dict[str, object] = {}
    t95 = {1: 0.0, 5: 2.776, 10: 2.262}
    for prefix, _ in groups:
        condition = {"n": len(raw[prefix]), "seeds": [run["seed"] for run in raw[prefix]]}
        for name in metric_names:
            values = np.asarray([run[name] for run in raw[prefix]])
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
            half_width = t95.get(values.size, 1.96) * sd / np.sqrt(values.size)
            condition[name] = {
                "values": [float(value) for value in values], "mean": mean, "sd": sd,
                "ci95": [mean - half_width, mean + half_width],
                "relative_change_vs_control": float(mean / control[name] - 1.0),
            }
        summary[prefix] = condition

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    labels = [label for _, label in groups]
    plot_metrics = (("net_head_displacement", "Net head displacement"),
                    ("mean_head_speed", "Mean head speed"),
                    ("mean_tail_speed", "Mean tail speed"),
                    ("tail_to_head_speed_ratio", "Tail/head speed ratio"))
    rng = np.random.default_rng(2024)
    for ax, (name, title) in zip(axes.flat, plot_metrics):
        for index, (prefix, _) in enumerate(groups):
            values = np.asarray(summary[prefix][name]["values"])
            mean = summary[prefix][name]["mean"]
            low, high = summary[prefix][name]["ci95"]
            ax.scatter(index + rng.uniform(-0.10, 0.10, values.size), values,
                       s=24, alpha=0.65, color="tab:blue")
            ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                        color="black", capsize=5, lw=1.5)
        ax.axhline(control[name], color="tab:red", ls="--", lw=1, alpha=0.7)
        ax.set(title=f"{title} (mean and 95% t CI)", xticks=np.arange(len(labels)),
               xticklabels=labels)
        ax.tick_params(axis="x", rotation=28)
    fig.savefig(out_dir / "supplementary6_multiseed_statistics.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_output")
    parser.add_argument(
        "--figure6-data",
        type=Path,
        default=Path(r"D:\BAAIWorm_Source_Fig6"),
        help="Directory containing the selectively extracted Zenodo Figure 6 data",
    )
    parser.add_argument(
        "--source-data-root", type=Path,
        default=Path(r"D:\BAAIWorm_Source_Data\Source_Data"),
        help="Root of the fully extracted official Zenodo Source Data",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "correlation_and_pca": correlation_reproduction(args.output, args.source_data_root),
        "reservoir_readout": reservoir_reproduction(args.output),
        "single_neuron_electrophysiology": single_neuron_reproduction(args.output),
        "supplementary_figure1_electrophysiology": supplementary1_electrophysiology_reproduction(args.output),
        "optimized_circuit": circuit_reproduction(args.output),
        "body_kinematics": body_kinematics_reproduction(args.output),
    }
    if args.figure6_data.exists():
        metrics["figure6_perturbations"] = perturbation_reproduction(
            args.figure6_data, args.output
        )
    if args.source_data_root.exists():
        metrics["supplementary_figure2_connection_locations"] = supplementary2_reproduction(
            args.source_data_root / "Supplementary Figure 2", args.output
        )
        metrics["supplementary_figure4_muscle_wave_delay"] = supplementary4_muscle_wave_reproduction(
            args.source_data_root / "Supplementary Figure 4", args.output
        )
        metrics["supplementary_figure6_multiseed_statistics"] = supplementary6_multiseed_reproduction(
            args.source_data_root / "Supplementary Figure 6", args.output
        )
        metrics["supplementary_figure7_readout_weights"] = supplementary7_readout_weights_reproduction(
            args.source_data_root / "Supplementary Figure 7", args.output
        )
        metrics["supplementary_figure9_neurite_propagation"] = supplementary9_reproduction(
            args.source_data_root / "Supplementary Figure 9", args.output
        )
        metrics["figure4_source_data"] = figure4_source_reproduction(
            args.source_data_root / "Figure 4", args.output
        )
        metrics["closed_vs_open_loop"] = loop_dynamics_reproduction(
            args.source_data_root / "Figure 5",
            args.source_data_root / "Supplementary Figure 8", args.output
        )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
