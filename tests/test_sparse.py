import unittest

import numpy as np

from wireless_ai.sparse_feedback import fit_fixed_sparse, quantize_dequantize, reconstruct


class SparseTests(unittest.TestCase):
    def test_training_fit_and_test_reconstruction_do_not_mutate_state(self):
        train = np.zeros((4, 2, 2), dtype=np.complex64)
        train[:, 1, 0] = np.array([-3, -1, 1, 3]) * (1 + 1j)
        model = fit_fixed_sparse(train, max_coefficients=2)
        self.assertEqual(model['indices'][0], 2)
        snapshot = {key: value.copy() for key, value in model.items()}
        held_out = np.full((2, 2, 2), 100 + 100j, dtype=np.complex64)
        predicted, bits = reconstruct(held_out, model, 8)
        self.assertEqual(bits, 8)
        self.assertEqual(predicted.shape, held_out.shape)
        for key in model:
            np.testing.assert_array_equal(model[key], snapshot[key])
        np.testing.assert_array_equal(predicted[:, 0, 0], model['mean'][0])
        self.assertLess(np.abs(predicted[:, 1, 0]).max(), 5)

    def test_known_quantization_and_full_reconstruction(self):
        values = np.array([-2, -1, -1 / 15, 1 / 15, 1, 2])
        np.testing.assert_allclose(quantize_dequantize(values, np.ones(6)), np.array([-1, -1, -1 / 15, 1 / 15, 1, 1]), atol=1e-7)
        train = np.array([[1 + 1j], [-1 - 1j]], dtype=np.complex64)
        model = fit_fixed_sparse(train, max_coefficients=1)
        predicted, bits = reconstruct(train, model, 8)
        np.testing.assert_allclose(predicted, train, atol=1e-7)
        self.assertEqual(bits, 8)

    def test_invalid_inputs_fail_instead_of_silent_budget_overrun(self):
        train = np.ones((2, 2), dtype=np.complex64)
        model = fit_fixed_sparse(train, max_coefficients=1)
        for budget in (0, -8, 4, 16, 8.0, True):
            with self.assertRaises(ValueError):
                reconstruct(train, model, budget)
        for invalid in (np.ones((0, 2), dtype=np.complex64), np.full((2, 2), np.nan + 1j), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                fit_fixed_sparse(invalid)
        with self.assertRaises(ValueError):
            reconstruct(np.ones((2, 3), dtype=np.complex64), model, 8)


if __name__ == '__main__':
    unittest.main()
