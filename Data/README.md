# Data layout

## Included in git

| Path | Description |
|------|-------------|
| `GPT2SP Data/Split/` | Train/val/test CSVs for 16 Choetkiertikul projects |
| `GPT2SP Data/Raw/` | Raw project exports (optional; splits are sufficient) |

Columns in split CSVs: `Issue` (title + description text), `Storypoint` (numeric label).

## Generated locally (gitignored)

| Path | How to create |
|------|----------------|
| `Embeddings/` | `python scripts/cache_embeddings.py --model mpnet` |

Files named `mpnet_{project}_{split}.npy` — 768-d mpnet embeddings.

## Provenance

- Splits follow [GPT2SP](https://github.com/awsm-research/gpt2sp) / [EfficientSPEComparativeLearning](https://github.com/hil-se/EfficientSPEComparativeLearning).
- Do not redistribute outside research terms of the original dataset authors.
