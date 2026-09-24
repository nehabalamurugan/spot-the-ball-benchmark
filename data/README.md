# Data

Every human and model response behind the paper, plus the stimuli and ground
truth needed to score them.

```
data/
├── images/                     # stimuli: 52 images per sport + ground truth
│   ├── basketball/  bb_gt.csv
│   ├── soccer/      soccer_gt.csv
│   └── volleyball/  vball_gt.csv
├── responses/                  # raw responses, exactly as collected
│   ├── humans/
│   │   ├── {sport}_clicks.csv          # 50 participants x 50 images x 3 clicks
│   │   └── {sport}_explanations.csv    # free-text rationales
│   └── models/{gpt,gemini,llama,qwen}/level{0,1,2}_{sport}.csv
└── processed/                  # generated -- do not edit by hand
    ├── all_responses.csv
    └── accuracy_by_condition.csv
```

Regenerate `processed/` from `responses/` at any time. No API keys, no network:

```bash
pip install -r requirements.txt
python src/data/build_dataset.py      # writes data/processed/
python src/data/validate_dataset.py   # checks the release, prints coverage
```

## The task

The ball is inpainted out of a sports photograph. A 6 x 10 grid is overlaid on
the image, and the respondent names the cell most likely to contain the hidden
ball.

**Grid convention.** Rows are letters `A`-`F` (top to bottom), columns are
numbers `1`-`10` (left to right), so a label looks like `D5`. Humans recorded an
integer `(row, col)` pair; models emitted the label directly. `build_dataset.py`
maps between the two.

**Ground truth.** A ball can straddle more than one cell, so `cell` in the
`*_gt.csv` files is a semicolon-separated set (`A5;B5`). A prediction is correct
when it lands anywhere in that set. Chance is roughly 1/60, higher for
multi-cell targets.

## Prompting levels

Models were run at three levels of scaffolding. Humans received no equivalent
manipulation, so human rows have an empty `level`.

| Level | Prompt |
|---|---|
| 0 | The ball has been removed; infer where it is. Nothing else. |
| 1 | Same, plus a hint that player location, gaze, and pose are informative. |
| 2 | Chain-of-thought: the model first answers *where are the players*, *where are they looking*, *how are they positioned*, then infers the location from its own answers. |

Level 0 and 1 were run at 50 samples per image, level 2 at 20. The exact prompt
text lives in `src/prompts/{model}.py`.

## `processed/all_responses.csv`

One row per response — every human click and every model prediction, 96,982 rows.

| Column | Description |
|---|---|
| `sport` | `basketball`, `soccer`, or `volleyball` |
| `level` | Prompting level `0`/`1`/`2`; empty for humans |
| `respondent_type` | `human` or `model` |
| `model` | `gpt`, `gemini`, `llama`, `qwen`; empty for humans |
| `participant_id` | Participant number within sport; empty for models |
| `image` | **Canonical image id** (integer). Join key to `*_gt.csv` and across respondents |
| `image_file` | Filename stem as written by that run — provenance only, see note below |
| `iteration` | Click number 1-3 for humans; sample index for models |
| `pred_cell` | Predicted cell label, e.g. `D5`; empty when the response could not be parsed |
| `pred_row`, `pred_col` | `pred_cell` as integers (row 1-6, col 1-10) |
| `gt_cells` | Ground-truth cell set, `;`-separated; empty for control images without ground truth |
| `correct` | `pred_cell ∈ gt_cells`; empty when undefined |
| `parse_failed` | `True` when no usable cell could be extracted |
| `is_control` | `True` for `*_wball` stimuli, which still show the ball |
| `in_human_set` | `True` for the 50 images per sport that participants also saw |

### Two filters you almost always want

```python
import pandas as pd
df = pd.read_csv("data/processed/all_responses.csv")
comparable = df[df.in_human_set & ~df.is_control]
```

- **`in_human_set`** — models were run on 52 images per sport, participants saw
  50. Any human-vs-model comparison must be restricted to the shared subset or
  the two groups are scored on different stimuli.
- **`is_control`** — the `*_wball` images still contain the ball. They were
  shown to models as a sanity check and withheld from participants. Basketball
  and volleyball have two each; soccer has none.

Note that soccer's two model-only images (302, 347) are *ordinary* stimuli that
participants simply did not see — they are excluded by `in_human_set`, not by
`is_control`.

### Scoring convention

`parse_failed` rows are **kept**, and `accuracy_by_condition.csv` counts them in
the denominator: a model that failed to produce an answer got that trial wrong.
Dropping them would inflate accuracy for exactly the models that struggled most
with the output format. To score only well-formed responses instead, filter on
`~parse_failed` yourself.

### Joining back to the reasoning text

`all_responses.csv` deliberately omits the free-text `reasoning` column, which
would roughly triple the repository size. It is preserved in the raw files under
`responses/models/`, joinable on `(model, level, sport, image_file, iteration)`.

## Known issues

These are real properties of the data the paper analysed. They are documented
rather than silently repaired, and `validate_dataset.py` reports them on every
run.

| Issue | Detail |
|---|---|
| **Gemini level 2 provenance** | Gemini's level-2 results come from the run that included worked example images (`*_withexample` in the working repository). The earlier plain level-2 attempt is not published: its volleyball run returned API errors for every row, and its soccer run had the wrong shape (48 images x 50 samples). |
| **GPT level 2 sample count** | Run at 10 samples per image rather than the intended 20. Its level-2 cells are therefore estimated from half as many samples as the other models'. |
| **`qwen/level1_basketball.csv`** | Contains 3,882 rows (~74.7 per image) because a re-run was appended to an earlier one. Published unmodified; de-duplicate on `(image, iteration)` if your analysis needs exactly 50. |
| **Parse failures** | Some responses contained no extractable grid cell — up to 7.5% for GPT at levels 0-1, near 0% for llama and qwen. See `parse_failure_rate` in `accuracy_by_condition.csv`. |
| **Short runs** | A few files fall slightly under their target (e.g. `qwen/level0_soccer.csv` at 2,553 of 2,600) where individual API calls failed and were not retried. |
| **Control image naming** | The same control stimuli were recorded as `51.png`/`52.png` by GPT and Gemini but `51_wball.png`/`52_wball.png` by llama and qwen. `build_dataset.py` reconciles this by keying on the image number, which is why you should join on `image` rather than `image_file`. |
| **Control ground truth** | Basketball's `*_gt.csv` includes rows for its control images; volleyball's does not. Controls are excluded from scoring either way. |

## Human free-text explanations

`responses/humans/{sport}_explanations.csv` holds 500 rationales per sport: each
participant explained 10 of their randomly chosen responses. Columns are `id`,
`order`, `image`, `response`. These are open-ended text and were not collected
for every trial, so they do not align one-to-one with `{sport}_clicks.csv`.
