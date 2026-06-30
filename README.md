# Robust Few-Shot Bayesian Scale Calibration for Cross-Project Story Point Estimation

Code for the paper submitted to **Software Quality Journal (Springer)**.

**Method**: After z-space SBERT-regression, restore predictions with **median–MAD few-shot Bayesian calibration** (K=20, no GPU).

**Main result**: average MAE **3.291** vs published SBERT-regression baseline **3.626** (16 projects, LOPO, Wilcoxon p=0.0125).

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_reproduce.sh          # one-command main reproduction
├── robust_spe/               # core library
│   ├── config.py             # paths, projects, hyperparameters
│   ├── data.py               # CSV / embedding I/O
│   ├── calibration.py        # median-MAD & mean-std calibration
│   ├── model.py              # PyTorch linear regressor
│   └── pipeline.py           # LOPO evaluation
├── scripts/
│   ├── cache_embeddings.py   # precompute mpnet embeddings
│   ├── run_main.py           # Table 1 (main results)
│   ├── run_ablation.py       # Table 2 (K / λ ablation)
│   └── run_significance.py   # Wilcoxon + failure analysis
├── results/                  # committed JSON outputs
├── Data/
│   └── GPT2SP Data/Split/    # Choetkiertikul 16-project splits (GPT2SP)
├── vendor/hil-se/            # upstream hil-se scripts (TensorFlow)
└── archive/exploration/      # internal experiment scripts (not needed to reproduce paper)
```

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step 1: cache embeddings (~5 min first run; ~100MB on disk, gitignored)
python scripts/cache_embeddings.py --model mpnet

# Step 2: main experiment (~1.5 h CPU/MPS; reproduces Table 1)
python scripts/run_main.py

# Step 3: significance tests (~instant)
python scripts/run_significance.py
```

Or:

```bash
bash run_reproduce.sh
```

**Ablation** (optional, ~3 h):

```bash
python scripts/run_ablation.py
```

---

## Data

- **Splits**: `Data/GPT2SP Data/Split/` — train/val/test CSVs for 16 projects (from [GPT2SP](https://github.com/awsm-research/gpt2sp) / [hil-se](https://github.com/hil-se/EfficientSPEComparativeLearning)).
- **Embeddings**: generated under `Data/Embeddings/` (not in git). Run `cache_embeddings.py` first.
- **Baseline numbers**: published SBERT-regression MAE from Li et al. (arXiv:2507.14642); stored in `robust_spe/config.py`.

---

## Expected output

After `run_main.py`, `results/results_robust_full.json` should show:

| Metric | Value |
|--------|-------|
| Average baseline MAE | 3.626 |
| Average med+MAD MAE | ~3.291 |
| Win rate vs baseline | 11/16 |

Pre-computed results are committed under `results/` for reference.

---

## Hardware

- **CPU** or **Apple MPS** or **CUDA** (auto-detected).
- No GPU required; mpnet encoding runs on CPU in `cache_embeddings.py`.

---

## Acknowledgments

- Dataset and splits: Choetkiertikul et al.; GPT2SP (Fu & Tantithamthavorn).
- SBERT-regression baseline code: [hil-se/EfficientSPEComparativeLearning](https://github.com/hil-se/EfficientSPEComparativeLearning) (vendored under `vendor/hil-se/`).
- Our contribution: `robust_spe/` package and calibration pipeline (`scripts/`).

---

## Citation

```bibtex
@article{yin2026robust,
  title={Robust Few-Shot Bayesian Scale Calibration for Cross-Project Story Point Estimation},
  author={Yin, Banghui},
  journal={Software Quality Journal},
  year={2026},
  note={Submitted}
}
```

---

## Publishing this repo on GitHub

This directory was originally cloned from hil-se. For a clean repo under your account:

```bash
rm -rf .git          # remove upstream clone metadata
git init
git add .
git commit -m "Initial release: Robust Few-Shot SPE calibration"
git remote add origin https://github.com/abluobo/RobustSPE.git
git push -u origin main
```

Suggested repo name: **RobustSPE**
