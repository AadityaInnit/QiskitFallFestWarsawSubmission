# Submission pack: team AadityaVichare, Track 2

**Deadline: Sunday 25 October 2026, 16:00 UTC (21:30 IST, 17:00 Warsaw).** Send early.

## Email (copy, fill the two links)

**To:** submission@aiqtlab.com
**Subject:** Submission - AadityaVichare - Track 2

Hello,

Please find our entry below.

- **Team name:** AadityaVichare
- **Members:** Aaditya Vichare (solo entry)
- **Track:** 2
- **GitHub:** <PUBLIC REPO URL> (MIT licence, README, four executed notebooks, tests)
- **Demo video (1-3 min):** <VIDEO URL>

Project: "When does a quantum kernel have a chance?" A fair, nested-CV benchmark of quantum-kernel SVMs
against tuned classical models on small real data, plus training-only diagnostics (kernel-target alignment,
geometric difference, concentration) and a fake-backend noise study. We report no quantum advantage on real
data and show where the classical models win; simulator-only, no QPU runs.

Best regards,
Aaditya Vichare

## GitHub repo settings
- Name: `qkernel-bench`; visibility **Public**; description: "Fair benchmark of quantum-kernel SVMs vs classical models, with diagnostics (Qiskit)".
- Topics: `quantum-machine-learning`, `qiskit`, `quantum-kernel`, `benchmark`, `hackathon`.
- Confirm the MIT licence badge shows in the sidebar.

```bash
cd qkernel-bench
git init && git add . && git commit -m "Qiskit Fall Fest entry: AadityaVichare, Track 2"
git branch -M main
git remote add origin https://github.com/<your-username>/qkernel-bench.git
git push -u origin main
```
The `results/` folder is about 5 MB, which is fine for GitHub.

## Fresh-clone test (do this before sending)
```bash
git clone https://github.com/<your-username>/qkernel-bench.git /tmp/qb && cd /tmp/qb
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v    # expect 15 tests OK
```
If a pinned version fails to install on your platform, loosen it in `requirements.txt` and say so in the README.

## 90-second demo script (screen recording of README + figures; read this aloud)
1. **0:00** "I'm Aaditya Vichare, team AadityaVichare, Track 2. The question: should you use a quantum kernel on your data?"
2. **0:12** *(show benchmark figure)* "I compared a quantum-kernel SVM with an RBF SVM, linear, logistic, forest and MLP. Same features, same folds, and the quantum feature map, depth and bandwidth are chosen inside training folds only."
3. **0:32** "Result: a tie on breast cancer and on a circles control. The winning quantum map has no entanglement, so a laptop simulates it exactly. The entangling maps do worse and are more fragile under fake-backend noise."
4. **0:52** *(show diagnostics table)* "The useful part: kernel-target alignment, computed without cross-validation, ranks configurations well on breast cancer. Geometric difference from RBF does not predict wins; larger values meant lower accuracy."
5. **1:10** *(show engineered figure)* "On a task built around a circuit, features win big, but a classical model given the same features matches it. The gain is the representation, not a quantum speedup."
6. **1:22** "Everything is MIT-licensed with tests and executed notebooks. Link in the description. Thank you."

## Rubric mapping (Track 2, 25% each)
| Criterion | Evidence in the repo |
|---|---|
| Originality | Diagnostic framing; showing "different from RBF" does not mean "better"; same-feature control on the engineered task |
| Usefulness and complexity | Nested CV, paired comparisons, frozen protocol, noise study on a fake backend, 15 tests |
| Community benefit | scikit-learn estimator, executed notebooks, documented reproduce steps, MIT licence |
| Presentation | One-minute README summary, results table, honest limits, 90-second video |

## Final checklist
- [ ] Repo public, README renders, hero figure shows
- [ ] Fresh clone installs and tests pass
- [ ] Video link opens while logged out
- [ ] Email sent with team name, member, track 2, repo and video links
