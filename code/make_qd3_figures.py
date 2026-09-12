"""Generate the compact QD3 figure set for the Neural Processing Letters paper.

The script reads only completed blind-test CSV files.  It writes vector PDF and
600-dpi PNG copies to ``qtfw/figures``.  The visual language follows the earlier
BC-Delta paper: restrained typography, pale fills, thin gray rules, and a
red/gray/blue palette.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Palette inherited from the earlier BC-Delta manuscript, with blue added for
# the query-side state/reference.
INK = "#252A30"
RED = "#B6403A"
DARK_RED = "#87312E"
BLUE = "#3E6F9E"
DARK_BLUE = "#2D5278"
GRAY = "#7D858D"
MID_GRAY = "#A9AFB4"
LIGHT_GRAY = "#E4E6E8"
PALE_GRAY = "#F4F5F6"
PALE_RED = "#F7E8E6"
PALE_BLUE = "#E8F0F7"
WHITE = "#FFFFFF"

TEXT_WIDTH_IN = 7.12

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.7,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight",
                pad_inches=0.035, facecolor=WHITE)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight",
                pad_inches=0.035, facecolor=WHITE)
    plt.close(fig)


def style_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID_GRAY)
    ax.tick_params(length=2.5, color=MID_GRAY)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GRAY, linewidth=0.55, zorder=0)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.11, 1.06, letter, transform=ax.transAxes, fontsize=9.3,
            fontweight="bold", color=INK, va="bottom")
    ax.set_title(title, loc="left", pad=9, color=INK, fontweight="bold")


def _wire(ax: plt.Axes, points: list[tuple[float, float]],
          *, color: str = GRAY, lw: float = 1.15,
          dashed: bool = False) -> None:
    """Draw a strictly rectilinear connector with one final arrowhead."""
    if len(points) < 2:
        raise ValueError("a wire needs at least two points")
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if not (math.isclose(x0, x1) or math.isclose(y0, y1)):
            raise ValueError(f"non-orthogonal wire segment: {(x0, y0)} -> {(x1, y1)}")
    style = (0, (3, 2.2)) if dashed else "solid"
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color=color, linewidth=lw, linestyle=style,
                solid_joinstyle="miter", zorder=2)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=8.5,
        color=color, linewidth=lw, linestyle=style, shrinkA=0, shrinkB=0,
        connectionstyle="arc3", zorder=2,
    ))


_BOX_TEXT: list[tuple[plt.Text, tuple[float, float, float, float], str]] = []


def _box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float],
         title: str, lines: list[str], *, edge: str, face: str,
         title_color: str | None = None, title_size: float = 8.1,
         body_size: float = 7.0, line_step: float = 0.23) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.07",
        linewidth=1.15, edgecolor=edge, facecolor=face, zorder=3,
    ))
    title_artist = ax.text(x + 0.12, y + h - 0.22, title, ha="left", va="center",
                           fontsize=title_size, fontweight="bold",
                           color=title_color or edge, zorder=4)
    _BOX_TEXT.append((title_artist, (x, y, w, h), title))
    first_y = y + h - 0.52
    for index, line in enumerate(lines):
        artist = ax.text(x + 0.12, first_y - line_step * index, line,
                         ha="left", va="center", fontsize=body_size,
                         color=INK, zorder=4)
        _BOX_TEXT.append((artist, (x, y, w, h), line))


def _check_box_text_fit(fig: plt.Figure, ax: plt.Axes) -> None:
    """Fail generation if any architecture label escapes its own box."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    failures = []
    for artist, (x, y, w, h), label in _BOX_TEXT:
        bbox = artist.get_window_extent(renderer=renderer).transformed(inverse)
        margin_x, margin_y = 0.055, 0.045
        if (bbox.x0 < x + margin_x or bbox.x1 > x + w - margin_x
                or bbox.y0 < y + margin_y or bbox.y1 > y + h - margin_y):
            failures.append(
                f"{label!r}: [{bbox.x0:.2f},{bbox.x1:.2f}] x "
                f"[{bbox.y0:.2f},{bbox.y1:.2f}] outside box "
                f"[{x:.2f},{x+w:.2f}] x [{y:.2f},{y+h:.2f}]"
            )
    if failures:
        raise RuntimeError("Architecture text overflow:\n" + "\n".join(failures))


def architecture_figure() -> None:
    """A grid-aligned two-lane view of the two recurrent dynamics."""
    _BOX_TEXT.clear()
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.20))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.85)
    ax.axis("off")

    # The internal system occupies one clean rectangular frame.  The write and
    # query lanes use the same column grid, so no connector is diagonal.
    ax.add_patch(FancyBboxPatch(
        (2.05, 0.30), 8.18, 4.22,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.95, linestyle=(0, (4, 3)), edgecolor=MID_GRAY,
        facecolor="none", zorder=0,
    ))
    ax.text(2.30, 4.30, "Query Dual Dynamics Delta (QD³)", fontsize=9.2,
            fontweight="bold", color=INK, ha="left", va="center")

    # External events and output.
    _box(ax, (0.10, 2.92), (1.65, 1.03), "WRITE",
         [r"$(\mathbf{k}_t,\mathbf{v}_t)$"], edge=INK, face=WHITE,
         title_color=INK, title_size=8.2, body_size=7.7)
    _box(ax, (0.10, 0.77), (1.65, 1.03), "QUERY",
         [r"$\mathbf{q}_j$"], edge=INK, face=WHITE,
         title_color=INK, title_size=8.2, body_size=7.7)
    _box(ax, (10.62, 0.86), (1.20, 0.86), "OUTPUT",
         [r"$\widehat{\mathbf{v}}_j$"], edge=INK, face=WHITE,
         title_color=INK, title_size=7.7, body_size=7.0)

    # Top row: write path.
    _box(ax, (2.40, 2.78), (2.45, 1.18), "Query importance",
         [r"$h_j(\mathbf{k}_t)=$",
          r"$\sum_i a_{j,i}(\mathbf{u}_{j,i}^{\mathsf{T}}\mathbf{k}_t)^2/s_j$"],
         edge=INK, face=WHITE, title_color=INK,
         title_size=8.2, body_size=7.15, line_step=0.24)
    _box(ax, (5.30, 2.78), (1.65, 1.18), "Clipped rate",
         [r"$\widetilde\eta_t=$",
          r"$\min\{\eta(1+\beta h_j),c\}$"],
         edge=GRAY, face=PALE_GRAY, title_color=INK,
         title_size=8.2, body_size=7.15, line_step=0.24)
    _box(ax, (7.40, 2.67), (2.45, 1.40), "Delta value memory",
         [r"$\widehat{\mathbf{v}}_t=\mathbf{M}_{t-1}\mathbf{k}_t$",
          r"$\mathbf{e}_t=\mathbf{v}_t-\widehat{\mathbf{v}}_t$",
          r"$\mathbf{M}_t\leftarrow\mathbf{M}_{t-1}+\widetilde\eta_t\mathbf{e}_t\mathbf{k}_t^{\mathsf{T}}$"],
         edge=INK, face=WHITE, title_color=INK,
         title_size=8.2, body_size=7.05, line_step=0.23)

    # Bottom row: query path.
    _box(ax, (2.40, 0.67), (2.70, 1.45), "Low-rank query trace",
         [r"$\mathbf{z}_j=\mathbf{U}_{j-1}^{\mathsf{T}}\mathbf{q}_j$",
          r"$\mathbf{r}_j=\mathbf{q}_j-\mathbf{U}_{j-1}\mathbf{z}_j$",
          r"$\mathbf{U}_j\leftarrow\mathbf{U}_{j-1}+\eta_u\mathbf{r}_j\mathbf{z}_j^{\mathsf{T}}$",
          r"$\mathbf{a}_j\leftarrow\lambda\mathbf{a}_{j-1}+\mathbf{z}_j^{\odot2}$"],
         edge=INK, face=PALE_GRAY, title_color=INK,
         title_size=8.2, body_size=6.85, line_step=0.205)
    _box(ax, (7.40, 0.75), (2.45, 1.18), "Recall",
         [r"$\widehat{\mathbf{v}}_j=\mathbf{M}_t\mathbf{q}_j$",
          "value state unchanged"], edge=GRAY, face=WHITE,
         title_color=INK, title_size=8.2, body_size=7.2)

    # Orthogonal write lane.
    _wire(ax, [(1.75, 3.44), (2.40, 3.44)], color=INK)
    _wire(ax, [(4.85, 3.37), (5.30, 3.37)], color=INK)
    _wire(ax, [(6.95, 3.37), (7.40, 3.37)], color=INK)

    # Orthogonal query lane and the two cross-state signals.
    _wire(ax, [(1.75, 1.29), (2.40, 1.29)], color=INK)
    _wire(ax, [(3.63, 2.12), (3.63, 2.78)], color=INK)
    _wire(ax, [(8.63, 2.67), (8.63, 1.93)], color=GRAY)

    # The same query is served through a lower bus that never crosses a box.
    _wire(ax, [(1.75, 1.29), (1.92, 1.29), (1.92, 0.47),
               (6.98, 0.47), (6.98, 1.34), (7.40, 1.34)],
          color=INK, lw=1.0)
    _wire(ax, [(9.85, 1.34), (10.62, 1.34)], color=INK)

    # Small, isolated state annotations; none sit on top of a connector.
    ax.text(3.78, 2.38, "read history", color=GRAY, fontsize=6.8,
            ha="left", va="center")
    ax.text(8.78, 2.28, r"$\mathbf{M}_t$", color=GRAY, fontsize=6.8,
            ha="left", va="center")
    ax.text(5.92, 0.58, r"same $\mathbf{q}_j$", color=GRAY,
            fontsize=6.8, ha="center", va="center")
    ax.text(10.78, 2.34, r"$\beta=0$  $\Rightarrow$  Plain Delta",
            color=GRAY, fontsize=6.5, ha="center", va="center")

    _check_box_text_fit(fig, ax)
    save(fig, "Fig1_qd3_architecture")


def _index_scores(rows: list[dict[str, str]], filters: dict[str, str]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            output[row["method"]][int(row["seed"])] = float(row["score"])
    return output


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    # All formal experiments use n=16.  Keeping the exact t critical here avoids
    # adding SciPy as a plotting-only dependency.
    critical = 2.131449545559323 if len(values) == 16 else 1.96
    half = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _paired_relative(rows: list[dict[str, str]], filters: dict[str, str],
                     method: str, baseline: str) -> tuple[float, float, float]:
    scores = _index_scores(rows, filters)
    common = sorted(set(scores[method]) & set(scores[baseline]))
    if len(common) != 16:
        raise ValueError(f"Expected 16 paired blind seeds for {filters}, got {len(common)}")
    gain = np.asarray([
        100.0 * (scores[baseline][seed] - scores[method][seed])
        / scores[baseline][seed] for seed in common
    ])
    return _mean_ci(gain)


def _grouped_gain_panel(ax: plt.Axes, rows: list[dict[str, str]],
                        conditions: list[tuple[str, dict[str, str]]],
                        *, letter: str, title: str, ylim: tuple[float, float]) -> None:
    methods = (
        ("write_trace", "Write trace", GRAY),
        ("dqd", "QD³", RED),
        ("full_query_trace", "Full query trace", BLUE),
    )
    x = np.arange(len(conditions), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)
    for offset, (method, label, color) in zip(offsets, methods):
        means, lows, highs = [], [], []
        for _, filters in conditions:
            mean, low, high = _paired_relative(rows, filters, method, "plain")
            means.append(mean); lows.append(low); highs.append(high)
        means_arr = np.asarray(means)
        errors = np.vstack((means_arr - np.asarray(lows), np.asarray(highs) - means_arr))
        bars = ax.bar(x + offset, means_arr, width=width * 0.90, color=color,
                      alpha=0.90, edgecolor=WHITE, linewidth=0.45,
                      label=label, zorder=3)
        ax.errorbar(x + offset, means_arr, yerr=errors, fmt="none", ecolor=color,
                    elinewidth=0.75, capsize=1.7, capthick=0.7, zorder=4)
        for rect, mean in zip(bars, means_arr):
            center = rect.get_x() + rect.get_width() / 2
            if offset < 0:
                label_x, align = center, "center"
            elif offset == 0:
                label_x, align = center - 0.018, "right"
            else:
                label_x, align = center + 0.018, "left"
            ax.text(label_x, mean + (0.55 if ylim[1] > 10 else 0.08),
                    f"{mean:+.1f}", ha=align, va="bottom", color=color,
                    fontsize=6.0, rotation=0)
    ax.axhline(0, color=INK, linewidth=0.7, zorder=2)
    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_ylim(*ylim)
    ax.set_ylabel("NRMSE reduction vs. Plain Delta (%)")
    panel(ax, letter, title)
    style_axis(ax)


def main_results_figure() -> None:
    """Four independent scales keep both large and boundary effects legible."""
    main_rows = read_csv(RESULTS / "main_blind_v1" / "per_seed.csv")
    wiki_rows = read_csv(RESULTS / "wikitext2_blind_v1" / "per_seed.csv")
    ordered_rows = read_csv(RESULTS / "wikitext2_ordered_v1" / "per_seed.csv")

    def draw(ax: plt.Axes, rows: list[dict[str, str]],
             filters: dict[str, str], title: str,
             ylim: tuple[float, float], yticks: list[float]) -> None:
        methods = (
            ("write_trace", "Write trace", GRAY, "o"),
            ("dqd", "QD³", RED, "D"),
            ("full_query_trace", "Full trace", BLUE, "s"),
        )
        span = ylim[1] - ylim[0]
        for xpos, (method, _, color, marker) in enumerate(methods):
            mean, low, high = _paired_relative(rows, filters, method, "plain")
            ax.bar(xpos, mean, width=0.52, color=color, alpha=0.90,
                   edgecolor=WHITE, linewidth=0.55, zorder=2)
            ax.errorbar(xpos, mean,
                        yerr=np.asarray([[mean - low], [high - mean]]),
                        fmt="none", color=color, elinewidth=0.9,
                        capsize=2.3, capthick=0.8, zorder=3)
            if mean >= 0:
                label_y, valign = high + 0.025 * span, "bottom"
            else:
                label_y, valign = low - 0.025 * span, "top"
            ax.text(xpos, label_y, f"{mean:+.2f}", color=color,
                    fontsize=6.35, ha="center", va=valign, zorder=4)
        ax.axhline(0, color=INK, linewidth=0.7, zorder=1)
        ax.set_xlim(-0.35, 2.55)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks(range(3), [item[1] for item in methods])
        ax.set_title(title, loc="left", pad=6, color=INK,
                     fontsize=8.2, fontweight="bold")
        style_axis(ax)

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.35),
                             constrained_layout=True)
    draw(axes[0, 0], main_rows,
         {"task": "synthetic_2d", "condition": "concentrated"},
         "(a) Synthetic · concentrated", (-2.0, 34.5), [0, 10, 20, 30])
    draw(axes[0, 1], main_rows,
         {"task": "synthetic_2d", "condition": "uniform"},
         "(b) Synthetic · uniform", (-0.75, 0.15), [-0.6, -0.4, -0.2, 0.0])
    draw(axes[1, 0], ordered_rows, {"condition": "ordered"},
         "(c) WikiText-2 · ordered tokens", (0.0, 3.45), [0, 1, 2, 3])
    draw(axes[1, 1], wiki_rows, {"condition": "uniform"},
         "(d) WikiText-2 · uniform queries", (0.0, 2.05), [0, 0.5, 1.0, 1.5, 2.0])
    axes[0, 0].set_ylabel("NRMSE reduction vs. Plain (%)")
    axes[1, 0].set_ylabel("NRMSE reduction vs. Plain (%)")
    save(fig, "Fig2_main_results")


def _load_robustness(task: str) -> list[dict[str, str]]:
    first = read_csv(RESULTS / "robustness_blind_v1" / "per_seed.csv")
    rest = read_csv(RESULTS / f"robustness_{task}_remaining_v1" / "per_seed.csv")
    return [row for row in first + rest if row.get("task") == task]


def mechanism_figure() -> None:
    """The two most useful ablations: query skew and low-rank capacity."""
    word_rows = _load_robustness("word")
    module_rows = _load_robustness("module")
    ablation_rows = read_csv(RESULTS / "ablation_blind_v1" / "per_seed.csv")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.72),
                             gridspec_kw={"width_ratios": (0.92, 1.24)},
                             constrained_layout=True)

    ax = axes[0]
    ax.set_title("(a) Query-skew response", loc="left", pad=6,
                 color=INK, fontweight="bold")
    alphas = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00, 1.25])
    for rows, task, label, color, marker in (
        (word_rows, "word", "Token workload", BLUE, "o"),
        (module_rows, "module", "Import graph", RED, "D"),
    ):
        means, lows, highs = [], [], []
        for alpha in alphas:
            filters = {"task": task, "condition": f"alpha_{alpha:.2f}"}
            mean, low, high = _paired_relative(rows, filters, "dqd", "write_trace")
            means.append(mean); lows.append(low); highs.append(high)
        means_arr = np.asarray(means)
        errors = np.vstack((means_arr - np.asarray(lows), np.asarray(highs) - means_arr))
        ax.errorbar(alphas, means_arr, yerr=errors, color=color, marker=marker,
                    markersize=4.0, linewidth=1.35, elinewidth=0.65,
                    capsize=1.7, zorder=3)
        ax.text(1.34, means_arr[-1], label, color=color, fontsize=6.5,
                fontweight="bold", ha="left", va="center")
    ax.axhline(0, color=GRAY, linewidth=0.75)
    ax.axvline(1.0, color=MID_GRAY, linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(1.0, 7.7, "observed", ha="center", va="top", color=GRAY,
            fontsize=6.4)
    ax.set_xlim(-0.05, 1.63)
    ax.set_ylim(-0.65, 8.0)
    ax.set_xticks(alphas)
    ax.set_xlabel(r"query-frequency exponent $\alpha$")
    ax.set_ylabel("QD³ advantage over write trace (%)")
    style_axis(ax)

    ax = axes[1]
    ax.set_title("(b) Low-rank approximation", loc="left", pad=6,
                 color=INK, fontweight="bold")
    ranks = [1, 2, 4, 8, 16]
    x = np.arange(6)
    task_specs = (
        ("synthetic_2d", "Synthetic field", GRAY, "o"),
        ("word", "Token workload", BLUE, "s"),
        ("module", "Import graph", RED, "D"),
    )
    for task, label, color, marker in task_specs:
        plain = _index_scores(ablation_rows, {"task": task, "condition": "concentrated" if task == "synthetic_2d" else "observed"})["plain"]
        means, lows, highs = [], [], []
        filters = {"task": task,
                   "condition": "concentrated" if task == "synthetic_2d" else "observed"}
        indexed = _index_scores(ablation_rows, filters)
        for rank in ranks:
            method = f"dqd_r{rank}"
            common = sorted(set(plain) & set(indexed[method]))
            values = np.asarray([100.0 * (plain[s] - indexed[method][s]) / plain[s]
                                 for s in common])
            mean, low, high = _mean_ci(values)
            means.append(mean); lows.append(low); highs.append(high)
        common = sorted(set(plain) & set(indexed["full_query_trace"]))
        full_values = np.asarray([
            100.0 * (plain[s] - indexed["full_query_trace"][s]) / plain[s]
            for s in common
        ])
        mean, low, high = _mean_ci(full_values)
        means.append(mean); lows.append(low); highs.append(high)
        means_arr = np.asarray(means)
        errors = np.vstack((means_arr - np.asarray(lows), np.asarray(highs) - means_arr))
        ax.errorbar(x, means_arr, yerr=errors, color=color, marker=marker,
                    markersize=3.8, linewidth=1.25, elinewidth=0.60,
                    capsize=1.5, label=label, zorder=3)
        ax.plot(x[-1], means_arr[-1], marker="*", markersize=7.2, color=color,
                markeredgecolor=WHITE, markeredgewidth=0.45, zorder=5)
        ax.text(5.18, means_arr[-1], label, color=color, fontsize=6.35,
                fontweight="bold", ha="left", va="center")
    ax.axvline(4.5, color=LIGHT_GRAY, linewidth=0.85)
    ax.set_xticks(x, ("1", "2", "4", "8", "16", "Full"))
    ax.set_xlim(-0.25, 6.15)
    ax.set_xlabel(r"query-trace rank $r$ (Full = $d_k\times d_k$)")
    ax.set_ylabel("NRMSE reduction vs. Plain Delta (%)")
    ax.set_ylim(15, 35.5)
    style_axis(ax)

    save(fig, "Fig3_mechanism_and_ablation")


def main() -> None:
    architecture_figure()
    main_results_figure()
    mechanism_figure()
    print(f"Wrote three PDF/PNG figure sets to {FIGURES}")


if __name__ == "__main__":
    main()
