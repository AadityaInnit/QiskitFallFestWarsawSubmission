# %% [markdown]
# # Diagnosing kernel geometry
# These are exploratory, training-set diagnostics, not tests of quantum advantage.
# Z, linear-entanglement ZZ, and all-to-all ZZ are implemented; there is no
# separate IQP construction in this version.
# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/qkb").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from qkb.diagnostics import geometric_difference
# %% [markdown]
# ## Geometric difference: no universal √n ceiling
# [Huang et al. (2021), Eq. 5](https://www.nature.com/articles/s41467-021-22539-9)
# defines $g(C\Vert Q)=\sqrt{\|Q^{1/2}C^{-1}Q^{1/2}\|_2}$ for trace-n kernels.
# We report an explicitly ridge-stabilized variant $g_\lambda$ with
# $C+\lambda I$ in the denominator. A symmetric generalized eigensolver avoids
# explicitly taking a matrix inverse or square root.
#
# The paper's full regularized error bound is not this simple ridge diagnostic.
# For PSD kernels and λ > 0, $g_\lambda\le\sqrt{n/\lambda}$, not √n.
# A nearly singular comparator can therefore produce large values. Large g
# against one RBF comparator does not imply superior SVM accuracy or advantage
# over the best classical model.
# %%
Q = np.eye(2)
C = np.ones((2, 2))
g = geometric_difference(Q, C, reg=1e-3)
print(f"Analytical counterexample: n=2, g={g:.6f}; sqrt(n)={np.sqrt(2):.6f}")
assert np.isclose(g, np.sqrt(1000))
display(Image(filename=str(ROOT / "figures/geometric_sensitivity.png")))
# %% [markdown]
# ## Map, depth, and bandwidth
# Every fixed-map curve tunes C inside inner CV. Selecting the largest curve
# would reuse the outer folds for selection; only the nested benchmark estimates
# the full map-selection procedure. Full-training alignment uses centered labels
# and kernels and is descriptive. Off-diagonal variance alone does not diagnose
# the cause of classification performance.
# %%
sweep = pd.read_csv(ROOT / "results/breast_cancer_4q/qkernel_sweep.csv")
display(sweep.sort_values("cv_acc", ascending=False).head(8).round(4))
display(Image(filename=str(ROOT / "figures/depth_bandwidth.png")))
display(Image(filename=str(ROOT / "figures/diagnostics.png")))
# %% [markdown]
# ## Qubit sweep
# This varies PCA dimensionality and qubit count together (2, 4, 6, 8) at reps=2.
# It shows an association with representation size, not an isolated qubit effect.
# %%
display(Image(filename=str(ROOT / "figures/concentration_qubits.png")))
