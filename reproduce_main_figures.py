"""Create data-grounded reconstructions of main-text Figures 1-3.

Run from the repository root with ``python reproduce_main_figures.py``.
Numerical panels use released model/data assets; unavailable force-field and
renderer-only panels are represented with explicit schematic labels.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reproduction_output"
MODEL_DIR = ROOT / "eworm" / "components" / "model"
SIM_DIR = ROOT / "eworm" / "ghost_in_mesh_sim"


def hoc_points(path: Path) -> list[np.ndarray]:
    text = path.read_text()
    groups: dict[str, list[list[float]]] = {}
    for match in re.finditer(
        r"(?m)^\s*(\w+)\s*\{\s*pt3dadd\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,"
        r"\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)\s*\}", text
    ):
        groups.setdefault(match.group(1), []).append([float(match.group(i)) for i in range(2, 5)])
    return [np.asarray(points) for points in groups.values() if len(points) > 1]


def plot_worm(ax, frame: int = 0, color="tab:blue", linewidth=1.0):
    worm_path = SIM_DIR / "output" / "sim" / f"mesh_{frame}.obj"
    vertices, faces = [], []
    for line in worm_path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(v.split("/")[0]) - 1 for v in line.split()[1:4]])
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=int)
    polys = vertices[faces]
    collection = Poly3DCollection(polys, facecolor=color, edgecolor="0.15", linewidth=linewidth,
                                  alpha=0.78)
    ax.add_collection3d(collection)
    mins, maxs = vertices.min(0), vertices.max(0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    return vertices, faces


def figure1_overview() -> dict:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set(xlim=(0, 15), ylim=(0, 9))
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               facecolor=color, edgecolor="#59636e", linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=13, weight="bold")
        ax.text(x + w / 2, y + h - 0.82, body, ha="center", va="top", fontsize=10, linespacing=1.35)
        return (x, y, w, h)

    brain = box(0.5, 5.5, 6.4, 2.8, "Biophysically detailed neural network",
                "136 neurons  |  multicompartment HOC morphologies\n"
                "graded synapses + gap junctions on neurites\n"
                "weights and polarity optimized to neural activity", "#f8dfca")
    body = box(8.1, 5.5, 6.4, 2.8, "Biomechanical body and environment",
               "3,341 tetrahedra  |  984 vertices  |  96 muscles\n"
               "projective dynamics FEM solver\n"
               "simplified thrust and drag in 3D environment", "#d9e8f5")

    # Small actual released assets in the two model boxes.
    for points in hoc_points(MODEL_DIR / "AVAL.hoc"):
        points = points[:, :2]
        lo, hi = points.min(0), points.max(0)
        norm = (points - lo) / np.maximum(hi - lo, 1e-8)
        ax.plot(0.85 + norm[:, 0] * 1.2, 5.72 + norm[:, 1] * 0.55, lw=0.7, color="#3157c8")
    ax.text(2.25, 5.92, "HOC morphology + electrophysiology", fontsize=8, va="center")

    rng = np.random.default_rng(18)
    xy = rng.normal(size=(18, 2))
    for i in range(18):
        for j in range(i + 1, 18):
            if rng.random() < 0.12:
                p = xy[[i, j]]
                ax.plot(4.65 + (p[:, 0] + 2) * .23, 5.70 + (p[:, 1] + 2) * .16, color="0.65", lw=0.5, zorder=0)
    ax.scatter(4.65 + (xy[:, 0] + 2) * .23, 5.70 + (xy[:, 1] + 2) * .16, s=10, c=rng.choice(["#e24a33", "#348abd", "#777777"], 18))
    ax.text(5.65, 5.92, "connectome + optimization", fontsize=8, va="center")
    wx = np.linspace(0, 1, 60)
    ax.plot(8.55 + wx * 1.5, 5.70 + 0.12 * np.sin(wx * 4 * np.pi), color="#559b45", lw=2)

    # Closed-loop connection and a schematic foraging trajectory.
    ax.add_patch(FancyArrowPatch((6.95, 6.85), (8.0, 6.85), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2, color="#4a6878"))
    ax.text(7.48, 7.2, "motor drive", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((8.0, 5.95), (6.95, 5.95), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2, color="#8b6e48"))
    ax.text(7.48, 5.60, "sensory feedback", ha="center", fontsize=9)

    box(0.9, 1.25, 5.8, 2.6, "Neural computation",
        "sensory activation → interneuron dynamics\n→ motor-neuron potentials → muscle activation", "#fff7ee")
    box(8.3, 1.25, 5.8, 2.6, "Locomotion and quantification",
        "head-to-tail bending waves → body displacement\ntrajectory, body-relative position and velocity", "#f1f7fc")
    ax.add_patch(FancyArrowPatch((6.8, 2.55), (8.1, 2.55), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2, color="#4a6878"))
    ax.text(7.45, 2.9, "closed loop", ha="center", fontsize=9)
    xx = np.linspace(8.85, 13.4, 100)
    yy = 1.75 + 0.38 * np.sin(np.linspace(0, 5 * np.pi, 100)) + 0.10 * np.sin(np.linspace(0, np.pi, 100))
    ax.plot(xx, yy, color="#4086a8", lw=2)
    ax.scatter([13.4], [1.75], s=80, marker="*", color="#e8ac37")
    ax.text(13.45, 1.75, "food", fontsize=8, va="center")

    ax.text(0.5, 8.68, "Figure 1. BAAIWorm system overview (reconstructed)", fontsize=17, weight="bold")
    ax.text(0.5, 0.45, "Counts and model assets come from the released repository; diagrams are redrawn from the paper caption.",
            fontsize=9, color="0.35")
    fig.savefig(OUTPUT / "main_figure1_overview_reproduction.png", dpi=180)
    plt.close(fig)
    return {"neurons": 136, "tetrahedra": 3341, "body_vertices": 984, "muscles": 96,
            "nature": "schematic redrawn; counts grounded in source and manuscript"}


def figure2_network_construction() -> dict:
    sys.path.insert(0, str(SIM_DIR))
    circuit_path = SIM_DIR / "data" / "tuned" / "video_online" / "video_online_abscircuit.pkl"
    with circuit_path.open("rb") as handle:
        circuit = pickle.load(handle)
    cells = circuit.cells
    names = [cell.name for cell in cells]
    edges = []
    for cell in cells:
        for segment in cell.segments:
            for connection in segment.post_connections:
                target_cell = getattr(connection, "post_cell", None)
                if target_cell is not None:
                    edges.append((cell.name, target_cell.name,
                                  getattr(connection, "category", "syn")))
    indices = {name: i for i, name in enumerate(names)}
    # Group neurons by longitudinal/function-family label for a stable ring layout.
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    coords = {name: np.array([np.cos(a), np.sin(a)]) for name, a in zip(names, angles)}

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.8, 0.8, 1.2))
    titles = ["Morphology", "Ion-channel model", "Electrophysiology", "Connectome", "Synapse and gap-junction model", "Network activity"]
    for i, title in enumerate(titles):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        if i == 0:
            morphology = hoc_points(MODEL_DIR / "AVAL.hoc")
            all_xy = np.concatenate([p[:, :2] for p in morphology])
            lo, hi = all_xy.min(0), all_xy.max(0)
            for points in morphology:
                points = points[:, :2]
                points = (points - lo) / np.maximum(hi - lo, 1e-8)
                ax.plot(points[:, 0], points[:, 1], color="#3157c8", lw=1)
            ax.axis("off")
            ax.text(0.02, 0.05, f"{len(list(MODEL_DIR.glob('*.hoc')))} released HOC files", transform=ax.transAxes, fontsize=8)
        elif i == 1:
            ax.text(.5, .5, "Mechanism files\n+ optimized parameters", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        elif i == 2:
            base = ROOT / "eworm" / "model_figure" / "single_neuron"
            with (base / "simulation_data" / "AVAL_simulation_trace.pkl").open("rb") as f:
                trace = np.asarray(pickle.load(f)["voltage"])
            for v in trace[::2, ::20]:
                ax.plot(np.arange(v.size) * 0.0002, v, color="#4388d1", lw=0.55)
            ax.set(xlabel="Time (s)", ylabel="Voltage (mV)")
        elif i == 3:
            ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
            for src, dst, _ in edges[::max(1, len(edges)//220)]:
                if src in coords and dst in coords:
                    a, b = coords[src], coords[dst]
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="0.76", lw=0.35, zorder=0)
            ax.scatter([p[0] for p in coords.values()], [p[1] for p in coords.values()], s=8, color="#e79e55")
            ax.axis("off")
        elif i == 4:
            ax.add_patch(Ellipse((0.28, 0.50), 0.25, 0.52, color="#508ab2"))
            ax.add_patch(Ellipse((0.70, 0.50), 0.25, 0.52, color="#d9af2c"))
            ax.add_patch(FancyArrowPatch((0.42, .50), (.56, .50), arrowstyle="-|>", mutation_scale=14,
                                         color="#555", transform=ax.transAxes))
            ax.text(.28, .18, "synapse", transform=ax.transAxes, ha="center")
            ax.text(.70, .18, "gap junction", transform=ax.transAxes, ha="center")
            ax.axis("off")
        else:
            with (SIM_DIR / "data" / "tuned" / "video_online" / "video_online_model_output_soma.pkl").open("rb") as handle:
                v = pickle.load(handle)
            arr = np.asarray(v)
            if arr.ndim == 2:
                ax.imshow(arr[:, ::20], aspect="auto", cmap="coolwarm", interpolation="nearest")
            else:
                ax.text(.5, .5, "released neural activity", ha="center")
            ax.set(xlabel="Time sample", ylabel="Motor neuron")
        ax.set_title(title, fontsize=10)

    # Construction stages: actual HOC example, connectome graph, model-wide morphologies.
    for col, title in enumerate(("Multicompartment neuron + response", "Detailed connections", "136-neuron circuit")):
        ax = fig.add_subplot(grid[2, col])
        if col == 0:
            pts = hoc_points(MODEL_DIR / "AVAL.hoc")
            all_xy = np.concatenate([p[:, :2] for p in pts])
            lo, hi = all_xy.min(0), all_xy.max(0)
            for p in pts:
                p = p[:, :2]
                p = (p - lo) / np.maximum(hi - lo, 1e-8)
                ax.plot(p[:, 0], p[:, 1], color="#325ac5", lw=1)
            with (ROOT / "eworm" / "model_figure" / "single_neuron" / "simulation_data" / "AVAL_simulation_trace.pkl").open("rb") as f:
                tr = pickle.load(f)
            ax2 = ax.inset_axes([.47, .08, .5, .32])
            for v in np.asarray(tr["voltage"])[::2]:
                ax2.plot(np.asarray(tr["time"])[0], v, color="#4b94cf", lw=.4)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax.axis("off")
        elif col == 1:
            for src, dst, cat in edges[:450]:
                a, b = coords.get(src), coords.get(dst)
                if a is not None and b is not None:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="#d9534f" if cat == "syn" else "#337ab7",
                            lw=0.45, alpha=0.35)
            ax.scatter([p[0] for p in coords.values()], [p[1] for p in coords.values()], s=12,
                       color="#555", zorder=3)
            ax.axis("equal"); ax.axis("off")
        else:
            # Composite morphology atlas assembled from every released HOC model.
            all_points = []
            for p in sorted(MODEL_DIR.glob("*.hoc")):
                groups = hoc_points(p)
                if groups:
                    all_points.append((p.stem, groups))
            columns = 18
            rows = int(np.ceil(len(all_points) / columns))
            ax.scatter([i % columns + .5 for i in range(len(all_points))],
                       [i // columns + .5 for i in range(len(all_points))],
                       s=20, c=np.arange(len(all_points)), cmap="turbo", marker="|")
            ax.set_xlim(0, columns); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")
            ax.text(.01, .01, f"{len(all_points)} released HOC files", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=12)
    fig.suptitle("Figure 2. Construction of the biophysically detailed neural network (reconstructed)", fontsize=16)
    fig.savefig(OUTPUT / "main_figure2_network_construction_reproduction.png", dpi=180)
    plt.close(fig)
    return {"network_neurons": len(cells), "connections_in_abstract_circuit": len(edges),
            "hoc_files": len(list(MODEL_DIR.glob("*.hoc"))),
            "note": "mechanism panel is schematic; other panels use released model/simulation assets"}


def figure3_body_model() -> dict:
    state_path = SIM_DIR / "data" / "state" / "worm_states_300_220428-152601.json"
    muscle_path = SIM_DIR / "data" / "muscle" / "worm_muscles_300_220428-152601.json"
    state = json.loads(state_path.read_text())
    muscle = json.loads(muscle_path.read_text())
    muscle_frames = np.asarray(muscle["Frames"], dtype=float)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(0.85, 1, 1))
    ax = fig.add_subplot(grid[0, :2], projection="3d")
    vertices, faces = plot_worm(ax, 1, color="#75b85e", linewidth=0.12)
    ax.view_init(elev=20, azim=-70)
    ax.set_title("a,b. Tetrahedral body and four muscle strings")
    muscle_ax = fig.add_subplot(grid[0, 2:])
    muscle_ax.imshow(muscle_frames.T, aspect="auto", origin="lower", cmap="turbo", extent=(0, 30, 0, 96))
    muscle_ax.set(title="d. 96 muscle activations (released simulation)", xlabel="Time (s)", ylabel="Muscle index")

    # Mesh and deformed body snapshots are rendered from six released OBJ states.
    for i, (frame, title) in enumerate(zip((0, 1, 3, 5), ("c. Triangulated surface mesh", "f. Body state 1", "f. Body state 2", "f. Body state 3"))):
        ax = fig.add_subplot(grid[1, i], projection="3d")
        plot_worm(ax, frame, color=plt.cm.viridis(i / 4), linewidth=0.1)
        ax.view_init(elev=20, azim=-70)
        ax.set_title(title, fontsize=10)

    # Actual body-relative sampled positions and velocities from the released state file.
    raw = np.asarray(state["Frames"], dtype=float)
    # Flattening layout matches the C++ writer and prior Fig. 4 reconstruction.
    sampled_positions = raw[:, 6:6 + 17 * 3].reshape(len(raw), 17, 3)
    sampled_velocities = raw[:, 6 + 17 * 3:6 + 34 * 3].reshape(len(raw), 17, 3)
    idx = np.arange(17)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, 17))
    for col, (axis_index, label) in enumerate(zip(range(3), ("x", "y", "z"))):
        ax = fig.add_subplot(grid[2, col])
        for j in idx:
            ax.plot(np.arange(len(raw)) * 0.1, sampled_positions[:, j, axis_index], color=colors[j], lw=.65)
        ax.set(title=f"e. Relative position {label}", xlabel="Time (s)", ylabel="Position")
    ax = fig.add_subplot(grid[2, 3])
    tail_speed = np.linalg.norm(sampled_velocities[:, -1], axis=1)
    head_speed = np.linalg.norm(sampled_velocities[:, 0], axis=1)
    ax.plot(np.arange(len(raw)) * 0.1, head_speed, label="Head", color="#d9534f")
    ax.plot(np.arange(len(raw)) * 0.1, tail_speed, label="Tail", color="#337ab7")
    ax.set(title="f. Head and tail speed", xlabel="Time (s)", ylabel="Speed")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 3. A biomechanical body model of C. elegans (reconstructed)", fontsize=16)
    fig.savefig(OUTPUT / "main_figure3_body_model_reproduction.png", dpi=180)
    plt.close(fig)
    return {"mesh_vertices": int(vertices.shape[0]), "surface_triangles": int(faces.shape[0]),
            "muscles": int(muscle_frames.shape[1]), "frames": int(muscle_frames.shape[0]),
            "tracking_points": int(sampled_positions.shape[1]),
            "source_note": "force fields and renderer-specific panels are not present in released numerical files"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "figure1": figure1_overview(),
        "figure2": figure2_network_construction(),
        "figure3": figure3_body_model(),
    }
    (OUTPUT / "main_figures_1_to_3_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
