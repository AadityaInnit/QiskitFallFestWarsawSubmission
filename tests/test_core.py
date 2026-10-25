"""Analytical checks for scientific quantities and kernel/data boundary cases."""
import unittest
import numpy as np
from scipy.linalg import sqrtm
from qkb.diagnostics import geometric_difference, target_alignment
from qkb.qkernel import exact_kernel, shot_kernel, feature_map, noisy_kernel
from qkb.data import load_raw, preprocessor


class DiagnosticsTests(unittest.TestCase):
    def test_identity_and_diagonal_generalized_eigenvalue(self):
        self.assertAlmostEqual(geometric_difference(np.eye(3), np.eye(3), 0), 1.)
        q, c = np.diag([2., 1.]), np.diag([.5, 2.5])
        self.assertAlmostEqual(geometric_difference(q, c, 0), 2.)

    def test_no_sqrt_n_ceiling(self):
        q, c = np.eye(2), np.ones((2, 2))
        g = geometric_difference(q, c, 1e-3)
        self.assertAlmostEqual(g, np.sqrt(1000), places=7)
        self.assertGreater(g, np.sqrt(2))

    def test_matches_direct_spectral_formula_and_ridge_monotonicity(self):
        rng = np.random.default_rng(3)
        A, B = rng.normal(size=(7, 7)), rng.normal(size=(7, 7))
        Q, C = A @ A.T, B @ B.T
        Q *= 7 / np.trace(Q); C *= 7 / np.trace(C)
        sq = sqrtm(Q)
        direct = np.sqrt(np.linalg.eigvalsh(sq @ np.linalg.solve(C + .01 * np.eye(7), sq))[-1])
        self.assertAlmostEqual(geometric_difference(Q, C, .01), direct, places=9)
        values = [geometric_difference(Q, C, r) for r in [.001, .01, .1, 1]]
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_reject_invalid_and_singular(self):
        for invalid in [np.array([[1, 2], [2, 1]]), np.zeros((2, 2)), np.array([[1, 0], [1, 1]])]:
            with self.assertRaises(ValueError):
                geometric_difference(invalid, np.eye(2))
        with self.assertRaises(ValueError):
            geometric_difference(np.eye(2), np.ones((2, 2)), 0)

    def test_alignment_centers_imbalanced_labels(self):
        y = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(target_alignment(np.outer(2*y-1, 2*y-1), y), 1.)
        self.assertEqual(target_alignment(np.ones((4, 4)), y), 0.)


class PipelineTests(unittest.TestCase):
    def test_fidelity_psd_and_diagonal(self):
        X = np.random.default_rng(1).uniform(0, np.pi, (8, 3))
        K = exact_kernel(feature_map("ZZ", 3, 2), X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 1., atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-10)

    def test_square_cross_kernel_is_not_symmetrized(self):
        K = np.array([[0., 1.], [0., 0.]])
        np.testing.assert_array_equal(shot_kernel(K, 10, np.random.default_rng(0)), K)

    def test_split_disjoint_deterministic_and_train_only_preprocessing(self):
        X, Xt, y, yt, ids, idst = load_raw()
        self.assertFalse(set(ids) & set(idst))
        np.testing.assert_array_equal(ids, load_raw()[4])
        prep = preprocessor(4).fit(X)
        before = prep.named_steps['standardize'].mean_.copy()
        prep.transform(Xt * 100)
        np.testing.assert_array_equal(before, prep.named_steps['standardize'].mean_)
        self.assertEqual(prep.transform(Xt).shape[1], 4)

    def test_compute_uncompute_orientation_against_exact(self):
        X = np.array([[.2, .3], [.4, .8]])
        fm = feature_map("ZZ", 2, 1)
        exact = exact_kernel(fm, X)
        measured = noisy_kernel(fm, X, shots=8192, seed=2)
        np.testing.assert_allclose(measured, exact, atol=.025)


if __name__ == "__main__":
    unittest.main()
