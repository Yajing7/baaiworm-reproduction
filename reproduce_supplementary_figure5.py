"""Annotate the released BAAIWorm GUI capture for Supplementary Figure 5.

This is a reconstruction from the repository's published GUI screenshot; it
does not launch the live CUDA/OptiX simulator.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reproduction_output"
SOURCE = ROOT / "img" / "GUI.png"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    image = mpimg.imread(SOURCE)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.11)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("Supplementary Figure 5. BAAIWorm GUI (reconstructed from released capture)",
                 fontsize=18, weight="bold", pad=18)

    # Axes coordinates (origin lower left) make callouts independent of the
    # source image's pixel resolution.
    labels = [
        ("Neural network dynamics", (0.03, 0.96), (0.16, 0.59)),
        ("C. elegans body", (0.39, 0.17), (0.49, 0.46)),
        ("Locomotion trajectory", (0.19, 0.77), (0.39, 0.49)),
        ("Four muscle-string signals", (0.70, 0.98), (0.90, 0.83)),
    ]
    for label, text_xy, point_xy in labels:
        ax.annotate(label, xy=point_xy, xycoords="axes fraction",
                    xytext=text_xy, textcoords="axes fraction",
                    ha="left", va="center", fontsize=10, weight="bold",
                    bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#52616b", "alpha": .92},
                    arrowprops={"arrowstyle": "->", "color": "#263746", "lw": 1.5})

    fig.text(.04, .055,
             "Source: repository img/GUI.png. The saved capture does not visibly identify a food-source marker; none was added.",
             fontsize=9, color="#444")
    fig.savefig(OUTPUT / "supplementary5_gui_reproduction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
