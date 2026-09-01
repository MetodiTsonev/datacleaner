"""Regenerate the plots for Chapter 5.

Every figure in the thesis is produced here, never by hand, so it can be remade after a
bug fix instead of quietly going stale. Numbers come from running the real pipeline -
nothing below is typed in from a previous run.

matplotlib is an authoring tool and is deliberately not part of the system: see Р13 in
writing/decisions.md, and project/tests/test_dependencies.py, which fails if src/ ever
imports it.

    python scripts/figures.py            # all of them
    python scripts/figures.py --quick    # 3 seeds instead of 10, for a fast look

Roughly four minutes at the full ten seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; this runs headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project"))

from src import clean, detect, evaluate, features, loader, profile
from src import plan as planner

OUT = ROOT / "writing" / "figures"
CENSUS = ROOT / "project" / "data" / "input" / "adult-census.csv"
TARGET = "income"
SHARES = (0.0, 0.1, 0.2, 0.4)
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 20260830)

# Greyscale-safe: meaning is carried by marker and line style as well as colour, because
# the thesis may be printed in black and white.
RAW_STYLE = {"color": "#B0413E", "marker": "o", "linestyle": "--"}
CLEAN_STYLE = {"color": "#1F4E79", "marker": "s", "linestyle": "-"}


def _setup() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),   # fits the text width of a portrait A4 page
        "figure.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _bg(value: float) -> str:
    """Thousands separated by a space, as Bulgarian typography wants: 36 196."""
    return f"{value:,.0f}".replace(",", "\u00a0")


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):  # png to look at, pdf for the document
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote writing/figures/{name}.png")


def measure(seeds: tuple[int, ...]) -> pd.DataFrame:
    """Run the whole pipeline once per (seed, damage level). This is the slow part."""
    frame = loader.read_table(CENSUS).frame
    rows = []
    for seed in seeds:
        for share in SHARES:
            comparison = evaluate.compare(
                frame, target=TARGET, corruption=share, seed=seed
            )
            rows.append({
                "seed": seed, "corruption": share,
                "raw": comparison.raw.auc, "cleaned": comparison.cleaned.auc,
                "difference": comparison.difference,
                "pipeline_rows": comparison.cleaned.rows,
                "naive_rows": comparison.naive_rows,
            })
        print(f"  seed {seed} done")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figure 8

def figure_08_difference_by_damage(data: pd.DataFrame) -> None:
    """AUC difference against how much of the file is broken.

    The band is the full spread across seeds, not a confidence interval, and it is drawn
    because it is wider than the effect - which is the point being made.
    """
    grouped = data.groupby("corruption")["difference"]
    mean, low, high = grouped.mean(), grouped.min(), grouped.max()
    x = np.array(mean.index) * 100
    n_seeds = data["seed"].nunique()  # not hard-coded: --quick uses fewer

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(x, low, high, color="#1F4E79", alpha=0.15,
                    label=f"разсейване между {n_seeds}-те разделяния")
    ax.plot(x, mean, **CLEAN_STYLE, label="средна разлика (след — преди)")

    for xi, yi in zip(x, mean, strict=True):
        ax.annotate(f"{yi:+.4f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 10 if yi > 0 else -16), ha="center", fontsize=8)

    ax.set_xlabel("повредени клетки (%)")
    ax.set_ylabel("разлика в AUC")
    ax.set_xticks(x)
    ax.legend(loc="lower right", frameon=False)
    ax.margins(y=0.22)
    _save(fig, "fig-08-difference-by-damage")


# ------------------------------------------------------------------- figure 9

def figure_09_usable_rows(data: pd.DataFrame) -> None:
    """Rows left to train on. The strongest number in the chapter."""
    grouped = data.groupby("corruption")[["pipeline_rows", "naive_rows"]].mean()
    x = np.array(grouped.index) * 100

    fig, ax = plt.subplots()
    ax.plot(x, grouped.pipeline_rows, **CLEAN_STYLE, label="след конвейера")
    ax.plot(x, grouped.naive_rows, **RAW_STYLE,
            label="ако просто изтриете непълните редове")
    ax.set_yscale("log")
    ax.set_xlabel("повредени клетки (%)")
    ax.set_ylabel("използваеми обучаващи редове (логаритмична ос)")
    ax.set_xticks(x)

    # Labels go above the point, and the last one is nudged inward: below the final
    # point is off the bottom of the axes, where it was being clipped.
    for i, (xi, yi) in enumerate(zip(x, grouped.naive_rows, strict=True)):
        ax.annotate(_bg(yi), (xi, yi), textcoords="offset points",
                    xytext=(-14 if i == len(x) - 1 else 0, 10),
                    ha="right" if i == len(x) - 1 else "center", fontsize=8)
    ax.annotate(_bg(grouped.pipeline_rows.iloc[-1]),
                (x[-1], grouped.pipeline_rows.iloc[-1]), textcoords="offset points",
                xytext=(-8, 10), ha="center", fontsize=8)

    ax.set_ylim(bottom=max(1, grouped.naive_rows.min() * 0.4))
    ax.legend(loc="center left", frameon=False)
    _save(fig, "fig-09-usable-rows")


# ------------------------------------------------------------------ figure 10

def figure_10_capital_gain() -> None:
    """Why the pipeline loses accuracy on a clean file.

    Left: the column as it is - a spike at zero with a long tail. Right: after log1p.
    The skew statistic improves greatly and the distances a linear model relies on are
    compressed, which is the whole finding.
    """
    frame = loader.read_table(CENSUS).frame
    profiles = profile.profile_frame(frame)
    findings = detect.detect(frame, profiles, target=TARGET)
    result = clean.run(frame, planner.build(findings, target=TARGET), target=TARGET)

    before = pd.to_numeric(result.train["capital_gain"], errors="coerce").dropna()
    _, _, report = features.build(
        result.train, result.test, profile.profile_frame(result.train), target=TARGET
    )
    after = np.log1p(before)
    zero_share = float((before == 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, values, title in (
        (axes[0], before, "преди"),
        (axes[1], after, "след log1p"),
    ):
        ax.hist(values, bins=40, color="#1F4E79")
        ax.set_yscale("log")
        ax.set_title(f"{title} — асиметрия {values.skew():+.2f}", fontsize=10)
        ax.set_ylabel("брой редове (лог.)")
    axes[0].set_xlabel("capital_gain")
    axes[1].set_xlabel("log1p(capital_gain)")

    # The finding, drawn rather than left to be inferred: two values a linear model can
    # easily tell apart end up almost on top of each other.
    for value in (5_000, 20_000):
        axes[0].axvline(value, color="#B0413E", linestyle=":", linewidth=1.2)
        axes[1].axvline(np.log1p(value), color="#B0413E", linestyle=":", linewidth=1.2)
    axes[1].annotate(
        f"{np.log1p(5_000):.1f} и {np.log1p(20_000):.1f}", xy=(0.97, 0.94),
        xycoords="axes fraction", ha="right", fontsize=8, color="#B0413E",
    )
    axes[0].annotate("5\u00a0000 и 20\u00a0000", xy=(0.97, 0.94), xycoords="axes fraction",
                     ha="right", fontsize=8, color="#B0413E")

    fig.suptitle(f"{zero_share:.1%} от стойностите са нула", fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "fig-10-capital-gain-log")

    logged = report.skew_before.get("capital_gain")
    if logged is not None:
        print(f"  (pipeline logged capital_gain: {logged:+.2f} -> "
              f"{report.skew_after['capital_gain']:+.2f})")


# ------------------------------------------------------- figure 2 (generated)

#: Which layer each module sits in, for grouping. Anything unlisted lands in "other",
#: which is the signal that this table needs updating rather than the diagram being wrong.
LAYERS = {
    "text": "основа", "finding": "основа",
    "loader": "четене", "profile": "четене",
    "checks": "откриване", "detect": "откриване", "validate": "откриване",
    "plan": "поправяне", "clean": "поправяне", "anomalies": "поправяне",
    "features": "подготовка", "evaluate": "доказателство", "report": "доказателство",
}


def figure_02_modules() -> None:
    """Write the module dependency graph as Mermaid, read from the actual imports.

    Drawn by hand this diagram is wrong within a week - the first attempt at it had
    every arrow reversed, because a data-flow picture and an import graph look alike
    until you check. Generating it means the thesis cannot claim an architecture the
    code does not have.
    """
    import ast

    src_dir = ROOT / "project" / "src"
    modules = sorted(p.stem for p in src_dir.glob("*.py") if p.stem != "__init__")

    edges: set[tuple[str, str]] = set()
    for path in src_dir.glob("*.py"):
        if path.stem == "__init__":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] != "src":
                    continue
                targets = [parts[1]] if len(parts) > 1 else [a.name for a in node.names]
                edges.update(
                    (path.stem, t) for t in targets if t in modules and t != path.stem
                )

    lines = [
        "%% Фигура 2 — зависимости между модулите в src/.",
        "%% ГЕНЕРИРАН ФАЙЛ. Не се редактира на ръка: пресъздава се с",
        "%%   python scripts/figures.py",
        "%% Стрелка A --> B означава: A внася B.",
        "flowchart LR",
    ]
    for layer in dict.fromkeys(LAYERS.values()):
        members = [m for m in modules if LAYERS.get(m, "друго") == layer]
        if not members:
            continue
        lines.append(f'    subgraph {layer.upper()}["{layer}"]')
        lines.append("        direction TB")
        lines += [f'        {m}["{m}.py"]' for m in members]
        lines.append("    end")
    for other in sorted({m for m in modules if m not in LAYERS}):
        lines.append(f'    {other}["{other}.py"]')
    lines.append("")
    lines += [f"    {a} --> {b}" for a, b in sorted(edges)]

    out = OUT / "fig-02-modules.mmd"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    unplaced = [m for m in modules if m not in LAYERS]
    print(f"  wrote writing/figures/fig-02-modules.mmd "
          f"({len(modules)} modules, {len(edges)} dependencies)")
    if unplaced:
        print(f"  NOTE: not assigned to a layer, add them to LAYERS: {unplaced}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="3 seeds instead of 10 - for checking layout, not for the thesis")
    args = parser.parse_args()

    seeds = SEEDS[:3] if args.quick else SEEDS
    if args.quick:
        print("QUICK MODE - 3 seeds. Do not use these figures in the thesis.")

    _setup()
    print(f"Measuring: {len(seeds)} seeds x {len(SHARES)} damage levels...")
    data = measure(seeds)
    data.to_csv(OUT / "measurements.csv", index=False)
    print(f"  wrote writing/figures/measurements.csv ({len(data)} runs)")

    figure_08_difference_by_damage(data)
    figure_09_usable_rows(data)
    figure_10_capital_gain()
    figure_02_modules()
    print("Done.")


if __name__ == "__main__":
    main()
