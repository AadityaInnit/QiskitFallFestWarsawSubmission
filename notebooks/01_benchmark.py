# %% [markdown]
# # When does a quantum kernel have a chance?
# A small, reproducible comparison on balanced breast-cancer data and a circles
# control. This notebook opens the saved validated results. Set `RERUN = True`
# to rerun the benchmark with the current Python environment (several minutes).
#
# ## Protocol
# One stratified 112/48 split from 160 samples, seed 0. Every method sees the
# same features and the same 5-fold × 3-repeat outer splits. Standardization,
# PCA and angle scaling are fit inside each inner training fold. Inner 3-fold
# CV tunes classical hyperparameters and quantum map, repetitions, bandwidth,
# and SVM C. The final holdout fits tune on training data only.
#
# The holdout was already examined during development, so it is a descriptive
# comparison rather than fresh confirmation. Fold SD is not a confidence interval.
# %%
from pathlib import Path
import sys
import subprocess
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/qkb").is_dir())
sys.path.insert(0, str(ROOT / "src"))
RERUN = False
if RERUN:
    subprocess.run([sys.executable, "scripts/run_benchmark.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/run_benchmark.py", "--dataset", "circles"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/make_figures.py"], cwd=ROOT, check=True)
# %%
benchmark = pd.read_csv(ROOT / "results/breast_cancer_4q/benchmark.csv")
display(benchmark.round(4))
display(Image(filename=str(ROOT / "figures/benchmark.png")))
# %% [markdown]
# ## Paired comparison
# The quantum row evaluates the entire selection procedure. The map may change
# across outer folds. Differences below are paired on identical folds; repeated
# folds share data, so a naive independent-fold t-test would overstate certainty.
# %%
paired = pd.read_csv(ROOT / "results/breast_cancer_4q/paired_comparisons.csv")
display(paired.drop(columns="note").round(4))
# %% [markdown]
# ## Where classical wins
# Compare RBF SVM with the nested quantum selection procedure, not with the
# largest score among 36 exploratory curves. The Z map has no entangling gates
# and admits efficient classical simulation. A strong Z result does not establish
# a quantum computational advantage. At this sample size, individual test
# differences are coarse (one sample is 1/48 ≈ 2.08 percentage points).
# %%
display(pd.read_csv(ROOT / "results/circles_2q/benchmark.csv").round(4))
display(Image(filename=str(ROOT / "figures/circles_control.png")))
# %% [markdown]
# Circles is a control for ordinary nonlinear structure. It is not a synthetic
# task constructed to favor a quantum kernel. See README for measured outcomes
# and `summary.json` for every split index and selected configuration.
