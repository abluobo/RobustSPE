"""Shared paths, project list, and paper baselines."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SPLITS = ROOT / "Data" / "GPT2SP Data" / "Split"
EMBEDDINGS_DIR = ROOT / "Data" / "Embeddings"
RESULTS_DIR = ROOT / "results"

PROJECTS = [
    "appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
    "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
    "springxd", "talenddataquality", "talendesb", "titanium", "usergrid",
]

# Published SBERT-regression MAE (Li et al., hil-se EfficientSPEComparativeLearning).
PAPER_BASELINE = {
    "appceleratorstudio": 2.50, "aptanastudio": 4.49, "bamboo": 1.91,
    "clover": 3.60, "datamanagement": 7.41, "duracloud": 2.71,
    "jirasoftware": 2.79, "mesos": 1.91, "moodle": 12.21, "mule": 2.69,
    "mulestudio": 3.38, "springxd": 2.11, "talenddataquality": 3.54,
    "talendesb": 1.88, "titanium": 3.32, "usergrid": 1.56,
}

# Mean-std few-shot calibration (K=20, λ=10) on our pipeline — for ablation table.
MUSIG_K20 = {
    "appceleratorstudio": 2.203, "aptanastudio": 4.089, "bamboo": 1.206,
    "clover": 3.102, "datamanagement": 9.573, "duracloud": 1.276,
    "jirasoftware": 2.301, "mesos": 1.518, "moodle": 11.543, "mule": 2.530,
    "mulestudio": 3.376, "springxd": 2.013, "talenddataquality": 3.485,
    "talendesb": 1.204, "titanium": 3.171, "usergrid": 1.017,
}

FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
EMBEDDING_PREFIX = "mpnet"
EMBEDDING_DIM = 768

DEFAULT_K = 20
DEFAULT_LAMBDA = 10
N_REPEAT = 10
EPOCHS = 1000
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAD_SCALE = 0.675

K_VALUES = [5, 10, 20, 30, 50]
LAMBDA_VALUES = [1, 5, 10, 20, 50]
