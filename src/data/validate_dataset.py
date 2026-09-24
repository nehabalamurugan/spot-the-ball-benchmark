"""Validate the packaged responses and print a coverage table.

    python src/data/validate_dataset.py

Hard failures (exit code 1) mean the release is internally inconsistent: a
missing file, a missing column, an image with no ground truth, or a cell label
outside the 6 x 10 grid.

Deviations from the intended run configuration are reported as warnings rather
than failures. They are real properties of the data that the paper analysed and
are documented in `data/README.md`; this script surfaces them so nobody has to
discover them by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import (  # noqa: E402
    CELL_RE,
    GROUND_TRUTH,
    LEVELS,
    MODELS,
    RESPONSES,
    SPORTS,
    control_images,
    image_key,
    image_number,
    load_ground_truth,
    normalise_cell,
)

N_IMAGES_MODEL = 52
N_IMAGES_HUMAN = 50
N_PARTICIPANTS = 50
N_CLICKS = 3

# Iterations per image the generation scripts were configured to run.
EXPECTED_ITERS = {0: 50, 1: 50, 2: 20}

# Deviations that are real and documented, not signs of a packaging error.
KNOWN_DEVIATIONS = {
    ("gpt", 2): "run with 10 iterations per image instead of 20",
    ("qwen", 1, "basketball"): "contains extra iterations from an appended re-run",
}

LEVEL0_LEVEL1_COLUMNS = {"image", "iteration", "reasoning", "cell"}
LEVEL2_COLUMNS = {"image", "iteration", "q1_answer", "q2_answer", "q3_answer", "reasoning", "cell"}

errors: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def check_human(sport: str) -> dict:
    path = RESPONSES / "humans" / f"{sport}_clicks.csv"
    if not path.exists():
        fail(f"missing {path.relative_to(path.parents[3])}")
        return {}
    df = pd.read_csv(path)

    missing = {"id", "order", "image", "click", "row", "col"} - set(df.columns)
    if missing:
        fail(f"humans/{sport}_clicks.csv missing columns: {sorted(missing)}")
        return {}

    if df["id"].nunique() != N_PARTICIPANTS:
        warn(f"humans/{sport}: {df['id'].nunique()} participants (expected {N_PARTICIPANTS})")
    if df["image"].nunique() != N_IMAGES_HUMAN:
        warn(f"humans/{sport}: {df['image'].nunique()} images (expected {N_IMAGES_HUMAN})")

    if not df["row"].between(1, 6).all():
        fail(f"humans/{sport}: row values outside 1-6")
    if not df["col"].between(1, 10).all():
        fail(f"humans/{sport}: col values outside 1-10")

    gt = load_ground_truth(sport)
    orphans = {int(i) for i in df["image"].unique()} - set(gt)
    if orphans:
        fail(f"humans/{sport}: images with no ground truth: {sorted(orphans)}")

    expected_rows = N_PARTICIPANTS * N_IMAGES_HUMAN * N_CLICKS
    if len(df) != expected_rows:
        warn(f"humans/{sport}: {len(df)} rows (expected {expected_rows})")

    return {
        "source": f"humans/{sport}",
        "rows": len(df),
        "images": df["image"].nunique(),
        # clicks per participant per image, not rows per image
        "per_image": round(len(df) / (df["id"].nunique() * df["image"].nunique()), 1),
        "unparsed": 0,
        "note": f"{df['id'].nunique()} participants x {N_CLICKS} clicks",
    }


def check_model(model: str, level: int, sport: str) -> dict:
    path = RESPONSES / "models" / model / f"level{level}_{sport}.csv"
    if not path.exists():
        fail(f"missing models/{model}/level{level}_{sport}.csv")
        return {}
    df = pd.read_csv(path)

    required = LEVEL2_COLUMNS if level == 2 else LEVEL0_LEVEL1_COLUMNS
    missing = required - set(df.columns)
    if missing:
        fail(f"models/{model}/level{level}_{sport}.csv missing columns: {sorted(missing)}")
        return {}

    images = {image_key(v) for v in df["image"].unique()}
    if len(images) != N_IMAGES_MODEL:
        warn(
            f"models/{model}/level{level}_{sport}: {len(images)} images "
            f"(expected {N_IMAGES_MODEL})"
        )

    gt = load_ground_truth(sport)
    controls = control_images(sport)
    # Control stimuli still show the ball, so some sports have no hidden-ball
    # ground truth for them. That is expected; a non-control orphan is not.
    orphans = {
        img
        for img in images
        if image_number(img) not in gt and image_number(img) not in controls
    }
    if orphans:
        fail(f"models/{model}/level{level}_{sport}: images with no ground truth: {sorted(orphans)}")

    # Any non-empty cell that is not a valid grid label is a packaging bug.
    raw = df["cell"].dropna().astype(str).str.strip().str.upper()
    raw = raw[(raw != "") & (raw != "ERROR")]
    bad = sorted(set(raw[~raw.str.match(CELL_RE)]))
    if bad:
        fail(f"models/{model}/level{level}_{sport}: invalid cell labels: {bad[:5]}")

    unparsed = sum(normalise_cell(v) is None for v in df["cell"])
    per_image = round(len(df) / len(images), 1) if images else 0

    expected = EXPECTED_ITERS[level]
    note = KNOWN_DEVIATIONS.get((model, level)) or KNOWN_DEVIATIONS.get((model, level, sport))
    if abs(per_image - expected) > 1 and not note:
        warn(
            f"models/{model}/level{level}_{sport}: {per_image} iterations/image "
            f"(expected {expected})"
        )

    return {
        "source": f"{model}/level{level}_{sport}",
        "rows": len(df),
        "images": len(images),
        "per_image": per_image,
        "unparsed": unparsed,
        "note": note or "",
    }


def check_processed() -> None:
    path = Path(GROUND_TRUTH["basketball"]).parents[2] / "processed" / "all_responses.csv"
    if not path.exists():
        warn("data/processed/all_responses.csv not built yet -- run build_dataset.py")
        return
    df = pd.read_csv(path)
    if df["in_human_set"].sum() == 0:
        fail("all_responses.csv: no rows flagged in_human_set")
    # `correct` must be defined wherever a cell parsed AND ground truth exists.
    inconsistent = (
        df["pred_cell"].notna() & df["gt_cells"].notna() & df["correct"].isna()
    ).sum()
    if inconsistent:
        fail(f"all_responses.csv: {inconsistent} scoreable rows with undefined `correct`")

    # Every row lacking ground truth must be a control stimulus.
    stray = (df["gt_cells"].isna() & ~df["is_control"]).sum()
    if stray:
        fail(f"all_responses.csv: {stray} non-control rows with no ground truth")

    # The control naming mismatch across model runs must not leak through.
    for sport in SPORTS:
        n = df[(df["sport"] == sport) & df["is_control"]]["image"].nunique()
        expected = len(control_images(sport))
        if n != expected:
            fail(f"all_responses.csv: {sport} flagged {n} control images (expected {expected})")


def main() -> int:
    rows: list[dict] = []

    for sport in SPORTS:
        if not GROUND_TRUTH[sport].exists():
            fail(f"missing ground truth for {sport}")
            continue
        r = check_human(sport)
        if r:
            rows.append(r)

    for model in MODELS:
        for level in LEVELS:
            for sport in SPORTS:
                r = check_model(model, level, sport)
                if r:
                    rows.append(r)

    check_processed()

    print(f"{'source':<34}{'rows':>8}{'images':>8}{'iters/img':>11}{'unparsed':>10}  note")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['source']:<34}{r['rows']:>8,}{r['images']:>8}"
            f"{r['per_image']:>11}{r['unparsed']:>10}  {r.get('note', '')}"
        )

    total = sum(r["rows"] for r in rows)
    print("-" * 95)
    print(f"{'TOTAL':<34}{total:>8,}")

    if warnings:
        print(f"\n{len(warnings)} documented deviation(s):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\nAll structural checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
