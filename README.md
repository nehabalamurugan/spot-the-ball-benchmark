# Spot The Ball: A Benchmark for Visual Social Inference

[![Paper](https://img.shields.io/badge/arXiv-2511.00261-b31b1b.svg)](https://arxiv.org/abs/2511.00261)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/nehabalamurugan/spot-the-ball)
[![Website](https://img.shields.io/badge/Website-Project%20Page-blue.svg)](https://nehabalamurugan.github.io/spot-the-ball-benchmark/)

Code and data for **[Spot The Ball: A Benchmark for Visual Social
Inference](https://arxiv.org/abs/2511.00261)** (arXiv:2511.00261).

People can infer a hidden element of a scene — a ball removed from a photograph —
from subtle behavioral cues: where players are looking, how they are posed, which
way their bodies are turned. This benchmark asks whether vision–language models
do the same.

The ball is inpainted out of a sports photograph, a 6 × 10 grid is laid over the
image, and the respondent names the cell most likely to hide it.

## Headline result

Accuracy on the 50 images per sport that humans and models both saw, pooled
across sports:

| Respondent | Accuracy |
|---|---|
| **Humans** | **26.4%** |
| GPT | 7.0% |
| Gemini | 5.8% |
| Qwen | 4.4% |
| Llama | 4.0% |

All four models land within a few points of the ~1.7% chance rate, while humans
clear it by more than an order of magnitude. Additional prompt scaffolding does
not close the gap — see `data/processed/accuracy_by_condition.csv` for the full
breakdown by sport, model, and level.

## What's here

```
data/
├── images/       52 stimuli per sport + ground-truth cell spans
├── responses/    every human click and model prediction, as collected
└── processed/    tidy, scored, analysis-ready tables
src/
├── data/         build + validate the released dataset
├── prompts/      the scripts used to query each model
├── analysis/     center bias, player proximity, embeddings
└── players/      player-proximity notebook
docs/             project website (GitHub Pages)
```

**96,982 responses** in total: 22,500 human clicks (50 participants × 50 images
× 3 clicks per sport) and 74,482 model predictions (4 models × 3 prompting
levels × 3 sports).

## Quickstart

```bash
pip install -r requirements.txt

python src/data/build_dataset.py      # rebuild data/processed/ from raw responses
python src/data/validate_dataset.py   # verify the release, print a coverage table
```

Neither script needs an API key or network access — they read only files
committed here. Regenerating the *model responses* themselves does need keys;
see [ENV.md](ENV.md).

```python
import pandas as pd

df = pd.read_csv("data/processed/all_responses.csv")

# Restrict to the stimuli humans and models both saw, excluding control images
comparable = df[df.in_human_set & ~df.is_control]

comparable.groupby(["respondent_type", "model", "level"], dropna=False)["correct"] \
          .apply(lambda s: s.fillna(False).mean())
```

## Prompting levels

| Level | Prompt |
|---|---|
| 0 | The ball has been removed; infer where it is. |
| 1 | Same, plus a hint that player location, gaze, and pose are informative. |
| 2 | Chain-of-thought: the model answers three scene questions first, then infers from its own answers. |

## Data documentation

**[`data/README.md`](data/README.md)** is the reference: full column dictionary,
the grid and ground-truth conventions, the two filters you almost always want,
the scoring convention for unparseable responses, and a **Known issues** table
covering every deviation between the intended run configuration and what was
actually collected. Read it before analysing.

## Citation

```bibtex
@misc{balamurugan2025spottheball,
  title         = {Spot The Ball: A Benchmark for Visual Social Inference},
  author        = {Balamurugan, Neha and Wu, Sarah and Chun, Adam and Gaw, Gabe and Eyzaguirre, Cristobal and Gerstenberg, Tobias},
  year          = {2025},
  eprint        = {2511.00261},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.00261}
}
```
