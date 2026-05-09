# generate the method overview figure (hook diagram + expected 2x2 pattern)
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# draw the two-panel method diagram and save as PNG
def main():
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "method_diagram.png"
    )
    output_path = os.path.abspath(output_path)

    figure, (top_axes, bottom_axes) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1]}
    )

    top_axes.set_xlim(0, 10)
    top_axes.set_ylim(0, 5)
    top_axes.axis("off")
    top_axes.set_title(
        "Forward-Hook Ablation (single transformer block)",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    def draw_box(axes, x, y, width, height, label, color, fontsize=10):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            edgecolor="black",
            facecolor=color,
        )
        axes.add_patch(box)
        axes.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )

    def draw_arrow(axes, x1, y1, x2, y2):
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.5,
            color="black",
        )
        axes.add_patch(arrow)

    draw_box(top_axes, 0.2, 3, 1.6, 1, "Prompt", "#E8F0FE")
    draw_arrow(top_axes, 1.8, 3.5, 2.6, 3.5)
    draw_box(top_axes, 2.6, 3, 2.0, 1, "Block L\n(target layer)", "#FFF1C2")
    draw_arrow(top_axes, 4.6, 3.5, 5.4, 3.5)
    draw_box(top_axes, 5.4, 3, 2.4, 1, "Forward hook\n(modify h)", "#FCE4E4")
    draw_arrow(top_axes, 7.8, 3.5, 8.6, 3.5)
    draw_box(top_axes, 8.6, 3, 1.2, 1, "Output", "#E8F0FE")

    top_axes.text(
        2.6 + 1.0,
        3 - 0.4,
        "h",
        ha="center",
        va="top",
        fontsize=11,
        fontstyle="italic",
    )
    top_axes.text(
        7.8 + 0.4,
        3 - 0.4,
        "h'",
        ha="center",
        va="top",
        fontsize=11,
        fontstyle="italic",
    )

    top_axes.text(
        0.2,
        1.7,
        "DoM:",
        fontsize=12,
        fontweight="bold",
    )
    top_axes.text(
        1.0,
        1.7,
        r"$h' = h - (h \cdot \hat{r})\,\hat{r}$",
        fontsize=13,
    )
    top_axes.text(
        4.0,
        1.7,
        "(subtract component along one direction)",
        fontsize=10,
        color="gray",
    )

    top_axes.text(
        0.2,
        0.7,
        "ActSVD:",
        fontsize=12,
        fontweight="bold",
    )
    top_axes.text(
        1.2,
        0.7,
        r"$h' = h - (h\,V^{T})\,V$",
        fontsize=13,
    )
    top_axes.text(
        4.0,
        0.7,
        "(subtract projection onto rank-4 subspace)",
        fontsize=10,
        color="gray",
    )

    top_axes.text(
        0.2,
        -0.1,
        "No model weights are changed. Hook is fully reversible.",
        fontsize=10,
        color="black",
        style="italic",
    )

    bottom_axes.set_xlim(0, 10)
    bottom_axes.set_ylim(0, 5)
    bottom_axes.axis("off")
    bottom_axes.set_title(
        "Expected Pattern if Safety and Utility Are Separable",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    column_x = [3.0, 6.0]
    row_y = [2.6, 1.0]
    cell_width = 2.8
    cell_height = 1.2

    bottom_axes.text(
        column_x[0] + cell_width / 2,
        row_y[0] + cell_height + 0.4,
        "Ablate safety",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    bottom_axes.text(
        column_x[1] + cell_width / 2,
        row_y[0] + cell_height + 0.4,
        "Ablate utility",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    bottom_axes.text(
        column_x[0] - 0.6,
        row_y[0] + cell_height / 2,
        "DoM",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    bottom_axes.text(
        column_x[0] - 0.6,
        row_y[1] + cell_height / 2,
        "ActSVD",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    cells = [
        (column_x[0], row_y[0], "ASR up\nUtility flat", "#D4F1D4"),
        (column_x[1], row_y[0], "ASR flat\nUtility down", "#D4F1D4"),
        (column_x[0], row_y[1], "ASR up\nUtility flat", "#D4F1D4"),
        (column_x[1], row_y[1], "ASR flat\nUtility down", "#D4F1D4"),
    ]
    for x, y, label, color in cells:
        draw_box(bottom_axes, x, y, cell_width, cell_height, label, color, fontsize=11)

    bottom_axes.text(
        0.2,
        -0.2,
        "Anything else means safety and utility are NOT cleanly separable.",
        fontsize=10,
        color="black",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
