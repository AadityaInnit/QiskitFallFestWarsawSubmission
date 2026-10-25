# %% [markdown]
# # A disclosed, engineered representation advantage
# This experiment asks whether a circuit-derived representation helps on a
# task explicitly constructed from that representation. It does **not** test
# quantum computational advantage. All features are simulated classically.
#
# A fixed six-qubit circuit produces local X/Y/Z measurements. A fixed linear
# teacher defines labels. Task, seeds, training sizes, search grids and the
# primary comparison were frozen before evaluating fresh data. Pilot data are
# separate. The protocol freeze is local, not an externally registered study.
# %%
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/qkb").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
from report_engineered import validate
spec, execution, scores = validate()
print("Frozen:", spec['frozen_at_utc'])
print("Evaluation began:", execution['started_at_utc'])
print("Protocol SHA256:", execution['protocol_sha256'])
print("Source hashes, sample separation and every saved prediction score verified.")
# %% [markdown]
# ## The circuit representation
# Local X/Y/Z expectations on each of six qubits give 18 features. The teacher
# labels a fixed weighted sum of these features. The RBF projected kernel then
# measures distances between examples in this feature space.
# %%
sys.path.insert(0, str(ROOT / "src"))
from qkb.engineered import make_circuit
print(make_circuit().draw(output="text", fold=100))
# %% [markdown]
# ## Primary comparison, fixed before evaluation
# n=128 versus a classical baseline selected by **training-only** inner CV.
# Intervals reflect ten independently generated datasets for one fixed teacher.
# Other sizes and individual comparisons are secondary and unadjusted.
# %%
summary = pd.read_csv(ROOT / 'results/engineered/summary.csv')
paired = pd.read_csv(ROOT / 'results/engineered/paired_comparisons.csv')
n = spec['primary_comparison']['n_train']
display(summary[summary.n_train == n].sort_values('mean_acc', ascending=False).round(4))
display(paired[(paired.n_train == n) & (paired.comparator == 'Classical raw-input selection')].round(4))
display(Image(filename=str(ROOT / 'figures/engineered_primary_comparison.png')))
# %% [markdown]
# ## Learning curves and the revealing controls
# The direct classical RBF model given the same local-Pauli features must match
# the precomputed projected kernel; this is checked on **every prediction**.
# Logistic regression given these features can directly learn the linear teacher.
# The nonentangling ablation removes CNOTs but retains the input encoding.
# The raw-input Fourier competitor includes frequencies 1–4 and pairwise products.
# %%
display(Image(filename=str(ROOT / 'figures/engineered_learning_curves.png')))
pivot = scores.pivot(index=['seed', 'n_train'], columns='model', values='test_acc')
assert np.array_equal(pivot['Projected quantum kernel'], pivot['Classical same-feature RBF'])
print("Same-feature RBF matches in all", len(pivot), "dataset/training-size cases.")
# %% [markdown]
# ## Limits and reproducibility
# This demonstrates the usefulness of an intentionally matched representation
# against specified baselines. These six-qubit features are classically computed
# here, so the result establishes no speedup or classical hardness. It is one
# teacher and one input distribution, not evidence about ordinary datasets.
# Hyperparameter searches have finite budgets; recorded warnings remain visible.
# %%
warnings = scores.groupby('model').fit_warning_count.sum().astype(int)
display(warnings.rename('recorded_tuning_warnings').to_frame())
display(Markdown("See [full methods and limitations](../ENGINEERED_RESULTS.md), the frozen protocol, and saved predictions."))
# %% [markdown]
# To reproduce the unchanged study from the repo root:
# ```bash
# python scripts/run_engineered.py --stage evaluate --overwrite
# python scripts/report_engineered.py
# python scripts/build_notebooks.py --execute
# ```
# Editing scientific sources invalidates their frozen hashes. A different task
# should be a separately versioned experiment, preserving this result.
