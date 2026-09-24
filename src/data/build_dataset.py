"""Build the tidy `all_responses.csv` from the raw human and model response files.

Reads only files committed to this repository -- no API keys, no network.
Running this script reproduces everything under `data/processed/`.

    python src/data/build_dataset.py

Grid convention
---------------
Every image is overlaid with a 6 x 10 grid. Rows are letters A-F (top to
bottom), columns are numbers 1-10 (left to right), so a cell label looks like
"D5". Humans recorded integer (row, col); models emitted the label directly.

Ground truth
------------
A ball can span more than one cell, so ground truth is a semicolon-separated
set of labels (e.g. "A5;B5"). A prediction counts as correct when it falls in
that set.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
RESPONSES = DATA / "responses"
IMAGES = DATA / "images"
OUT_DIR = DATA / "processed"

SPORTS = ["basketball", "soccer", "volleyball"]
MODELS = ["gpt", "gemini", "llama", "qwen"]
LEVELS = [0, 1, 2]

GROUND_TRUTH = {
    "basketball": IMAGES / "basketball" / "bb_gt.csv",
    "soccer": IMAGES / "soccer" / "soccer_gt.csv",
    "volleyball": IMAGES / "volleyball" / "vball_gt.csv",
}

N_ROWS, N_COLS = 6, 10
CELL_RE = re.compile(r"^([A-F])(10|[1-9])$")

TIDY_COLUMNS = [
    "sport",
    "level",
    "respondent_type",
    "model",
    "participant_id",
    "image",
    "image_file",
    "iteration",
    "pred_cell",
    "pred_row",
    "pred_col",
    "gt_cells",
    "correct",
    "parse_failed",
    "is_control",
    "in_human_set",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rowcol_to_cell(row: int, col: int) -> str | None:
    """Map a 1-indexed (row, col) pair to a grid label such as "D5"."""
    if not (1 <= row <= N_ROWS and 1 <= col <= N_COLS):
        return None
    return f"{chr(ord('A') + row - 1)}{col}"


def cell_to_rowcol(cell: str | None) -> tuple[int | None, int | None]:
    """Inverse of :func:`rowcol_to_cell`; returns (None, None) if unparseable."""
    if not isinstance(cell, str):
        return None, None
    m = CELL_RE.match(cell.strip().upper())
    if not m:
        return None, None
    return ord(m.group(1)) - ord("A") + 1, int(m.group(2))


def normalise_cell(value) -> str | None:
    """Return a canonical cell label, or None when the model emitted nothing usable.

    Model runs that failed to parse leave the `cell` column empty; a handful of
    aborted API calls wrote the sentinel "ERROR" instead.
    """
    if not isinstance(value, str):
        return None
    cell = value.strip().upper()
    if not cell or cell == "ERROR":
        return None
    return cell if CELL_RE.match(cell) else None


def image_key(value) -> str:
    """Strip the file extension so model and human image ids line up.

    Model files refer to "63.png" / "52_wball.png"; humans and the ground-truth
    tables use the bare stem.
    """
    return str(value).strip().rsplit(".", 1)[0]


def image_number(key: str) -> int | None:
    """Leading integer of an image key ("52_wball" -> 52)."""
    m = re.match(r"(\d+)", key)
    return int(m.group(1)) if m else None


def load_ground_truth(sport: str) -> dict[int, str]:
    """Map image number -> semicolon-separated ground-truth cell span."""
    gt = pd.read_csv(GROUND_TRUTH[sport])
    return {int(r.image): str(r.cell).strip().upper() for r in gt.itertuples()}


def control_images(sport: str) -> set[int]:
    """Image numbers of the `*_wball` control stimuli, read from the image set.

    These must come from the stimulus directory rather than from the response
    files: the generation runs disagree on naming. gpt and gemini recorded the
    volleyball controls as "51.png"/"52.png" while llama and qwen recorded the
    same two images as "51_wball.png"/"52_wball.png", so keying off the response
    filename would flag the controls for only half the models.
    """
    return {
        n
        for p in (IMAGES / sport).glob("*wball*")
        if (n := image_number(p.stem)) is not None
    }


def score(pred_cell: str | None, gt_cells: str | None) -> bool | None:
    """True/False when both sides are known, None when the comparison is undefined."""
    if pred_cell is None or not gt_cells:
        return None
    return pred_cell in {c.strip() for c in gt_cells.split(";") if c.strip()}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_human(sport: str) -> pd.DataFrame:
    """One row per human click (3 clicks per participant per image)."""
    df = pd.read_csv(RESPONSES / "humans" / f"{sport}_clicks.csv")
    out = pd.DataFrame(
        {
            "sport": sport,
            # Humans saw no level manipulation; typed so it concatenates with
            # the models' integer levels instead of coercing them to float.
            "level": pd.array([pd.NA] * len(df), dtype="Int64"),
            "respondent_type": "human",
            "model": pd.array([None] * len(df), dtype="string"),
            "participant_id": df["id"].astype("Int64"),
            "image": df["image"].astype("Int64"),
            "image_file": df["image"].astype(str),
            "iteration": df["click"].astype("Int64"),
            "pred_row": df["row"].astype("Int64"),
            "pred_col": df["col"].astype("Int64"),
        }
    )
    out["pred_cell"] = [
        rowcol_to_cell(r, c) if pd.notna(r) and pd.notna(c) else None
        for r, c in zip(df["row"], df["col"])
    ]
    return out


def load_model(model: str, level: int, sport: str) -> pd.DataFrame:
    """One row per model prediction. Unparseable rows are kept, not dropped."""
    path = RESPONSES / "models" / model / f"level{level}_{sport}.csv"
    df = pd.read_csv(path)
    cells = [normalise_cell(v) for v in df["cell"]]
    rowcols = [cell_to_rowcol(c) for c in cells]
    return pd.DataFrame(
        {
            "sport": sport,
            "level": pd.array([level] * len(df), dtype="Int64"),
            "respondent_type": "model",
            "model": pd.array([model] * len(df), dtype="string"),
            "participant_id": pd.array([pd.NA] * len(df), dtype="Int64"),
            "image": pd.array(
                [image_number(image_key(v)) for v in df["image"]], dtype="Int64"
            ),
            "image_file": df["image"].map(image_key),
            "iteration": df["iteration"].astype("Int64"),
            "pred_cell": cells,
            "pred_row": pd.array([rc[0] for rc in rowcols], dtype="Int64"),
            "pred_col": pd.array([rc[1] for rc in rowcols], dtype="Int64"),
        }
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    human_images: dict[str, set[int]] = {}

    for sport in SPORTS:
        human = load_human(sport)
        human_images[sport] = set(human["image"])
        frames.append(human)
        for model in MODELS:
            for level in LEVELS:
                frames.append(load_model(model, level, sport))

    tidy = pd.concat(frames, ignore_index=True)

    # Join ground truth on the numeric part of the image id.
    gt_lookup = {sport: load_ground_truth(sport) for sport in SPORTS}
    tidy["gt_cells"] = [
        gt_lookup[s].get(img) for s, img in zip(tidy["sport"], tidy["image"])
    ]

    tidy["correct"] = [score(p, g) for p, g in zip(tidy["pred_cell"], tidy["gt_cells"])]
    tidy["parse_failed"] = tidy["pred_cell"].isna()

    # `*_wball` stimuli still contain the ball; they were shown to models as a
    # control and withheld from participants. Matched by image number so the
    # inconsistent naming across model runs cannot hide any of them.
    controls = {sport: control_images(sport) for sport in SPORTS}
    tidy["is_control"] = [
        img in controls[s] for s, img in zip(tidy["sport"], tidy["image"])
    ]

    # Models saw 52 images per sport, participants saw 50. Human/model
    # comparisons must be restricted to the shared subset.
    tidy["in_human_set"] = [
        img in human_images[s] for s, img in zip(tidy["sport"], tidy["image"])
    ]

    tidy["correct"] = tidy["correct"].astype("boolean")
    return tidy[TIDY_COLUMNS]


def summarise(tidy: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per condition, on the images humans and models both saw.

    `parse_failed` rows stay in the denominator: a model that failed to answer
    got the trial wrong, and silently dropping those rows would inflate scores.
    """
    scope = tidy[tidy["in_human_set"] & ~tidy["is_control"]].copy()
    scope["correct_filled"] = scope["correct"].fillna(False).astype(bool)

    grouped = scope.groupby(
        ["sport", "respondent_type", "model", "level"], dropna=False, observed=True
    )
    summary = grouped.agg(
        n_responses=("correct_filled", "size"),
        n_correct=("correct_filled", "sum"),
        n_parse_failed=("parse_failed", "sum"),
        n_images=("image", "nunique"),
    ).reset_index()
    summary["accuracy"] = (summary["n_correct"] / summary["n_responses"]).round(4)
    summary["parse_failure_rate"] = (
        summary["n_parse_failed"] / summary["n_responses"]
    ).round(4)
    return summary.sort_values(
        ["sport", "respondent_type", "model", "level"]
    ).reset_index(drop=True)


def main() -> int:
    missing = [p for p in GROUND_TRUTH.values() if not p.exists()]
    if missing:
        print("Missing ground-truth files:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tidy = build()
    tidy.to_csv(OUT_DIR / "all_responses.csv", index=False)
    print(f"Wrote {OUT_DIR/'all_responses.csv'}  ({len(tidy):,} rows)")

    summary = summarise(tidy)
    summary.to_csv(OUT_DIR / "accuracy_by_condition.csv", index=False)
    print(f"Wrote {OUT_DIR/'accuracy_by_condition.csv'}  ({len(summary):,} rows)")

    print("\nAccuracy by model and level (all sports pooled, shared 50 images):")
    pooled = (
        tidy[tidy["in_human_set"] & ~tidy["is_control"]]
        .assign(correct_filled=lambda d: d["correct"].fillna(False).astype(bool))
        .groupby(["respondent_type", "model", "level"], dropna=False, observed=True)["correct_filled"]
        .agg(["size", "mean"])
    )
    for (rtype, model, level), row in pooled.iterrows():
        label = "human" if rtype == "human" else f"{model} L{level}"
        print(f"  {label:<14} n={int(row['size']):>6}  acc={row['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
