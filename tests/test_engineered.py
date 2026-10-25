"""Physical and statistical integrity checks for the engineered task."""
import unittest
import numpy as np
from qiskit.quantum_info import Statevector, partial_trace
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from qkb.engineered import (N_QUBITS, make_circuit, projected_features, observables,
                            teacher_weights, labels, draw_inputs, fourier_features)


class EngineeredTests(unittest.TestCase):
    def test_local_features_match_reduced_density_matrices(self):
        X = draw_inputs(7, 2, 10)
        F = projected_features(X)
        self.assertLessEqual(np.abs(F).max(), 1 + 1e-12)
        for i, x in enumerate(X):
            state = Statevector(make_circuit().assign_parameters(x))
            for q in range(N_QUBITS):
                rho = partial_trace(state, [k for k in range(N_QUBITS) if k != q]).data
                bloch = F[i, 3*q:3*q+3]
                reconstructed = np.array([[1+bloch[2], bloch[0]-1j*bloch[1]],
                                          [bloch[0]+1j*bloch[1], 1-bloch[2]]]) / 2
                np.testing.assert_allclose(rho, reconstructed, atol=1e-12)
                self.assertLessEqual(np.linalg.norm(bloch), 1+1e-12)

    def test_projected_distance_matches_one_rdm_distance(self):
        X = draw_inputs(8, 2, 10)
        states = [Statevector(make_circuit().assign_parameters(x)) for x in X]
        F = projected_features(X)
        distance = 0
        for q in range(N_QUBITS):
            trace = [k for k in range(N_QUBITS) if k != q]
            a, b = [partial_trace(state, trace).data for state in states]
            distance += np.linalg.norm(a-b, 'fro')**2
        self.assertAlmostEqual(distance, np.sum((F[0]-F[1])**2)/2, places=12)

    def test_direct_and_precomputed_rbf_are_equivalent(self):
        X = draw_inputs(9, 20, 10)
        F = projected_features(X)
        y = labels(F, 0)
        G = projected_features(draw_inputs(9, 8, 20))
        a = SVC(C=10, gamma=1).fit(F, y)
        b = SVC(C=10, kernel='precomputed').fit(rbf_kernel(F, gamma=1), y)
        np.testing.assert_array_equal(a.predict(G), b.predict(rbf_kernel(G, F, gamma=1)))
        np.testing.assert_allclose(a.decision_function(G), b.decision_function(rbf_kernel(G, F, gamma=1)), atol=1e-10)

    def test_distinct_streams_and_prefix_learning_curve(self):
        X = draw_inputs(10, 32, 10)
        np.testing.assert_array_equal(X, draw_inputs(10, 64, 10)[:32])
        self.assertFalse(np.array_equal(X, draw_inputs(10, 32, 20)))
        self.assertFalse(np.array_equal(X, draw_inputs(11, 32, 10)))
        self.assertFalse(np.array_equal(draw_inputs(101,32,10), draw_inputs(1000,32,10)))

    def test_teacher_and_fourier_features_are_fixed_functions(self):
        X = draw_inputs(12, 4, 10)
        F = projected_features(X)
        np.testing.assert_allclose(projected_features(X[[2,0]]), F[[2,0]], atol=1e-12)
        np.testing.assert_array_equal(labels(F, .1), (F @ teacher_weights() >= .1).astype(int))
        P = fourier_features(X)
        np.testing.assert_array_equal(fourier_features(X[[2,0]]), P[[2,0]])
        self.assertEqual(P.shape, (4, 1224))
        self.assertFalse(np.allclose(F, projected_features(X, entangling=False)))


if __name__ == '__main__':
    unittest.main()


class EstimatorTests(unittest.TestCase):
    def test_sklearn_api_matches_precomputed_svc(self):
        from sklearn.base import clone
        from sklearn.svm import SVC
        from qkb.estimator import QuantumKernelSVC
        from qkb.qkernel import feature_map, exact_kernel
        rng = np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (30, 3)); y = (X[:, 0] > X[:, 1]).astype(int)
        est = clone(QuantumKernelSVC(fmap="ZZ", reps=1, bandwidth=0.5, C=10)).fit(X[:20], y[:20])
        fm = feature_map("ZZ", 3, 1)
        ref = SVC(kernel="precomputed", C=10).fit(exact_kernel(fm, X[:20], bandwidth=0.5), y[:20])
        expect = ref.predict(exact_kernel(fm, X[20:], X[:20], bandwidth=0.5))
        self.assertTrue(np.array_equal(est.predict(X[20:]), expect))
