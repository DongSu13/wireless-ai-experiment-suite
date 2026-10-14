import unittest

import numpy as np

from wireless_ai.validation.metrics import (
    beam_gain_per_sample, complex_sgcs_per_sample, feedback_bits,
    illustrative_se_per_sample, nmse_db_from_samples, nmse_per_sample,
    paired_improvement_ci,
)


class MetricTests(unittest.TestCase):
    def test_nmse_and_db_average_linear_errors(self):
        truth = np.array([[1 + 0j, 0j], [2 + 0j, 0j]])
        estimate = np.array([[0j, 0j], [1 + 0j, 0j]])
        np.testing.assert_allclose(nmse_per_sample(truth, estimate), [1.0, 0.25])
        self.assertAlmostEqual(nmse_db_from_samples(np.array([1.0, 0.25])), 10 * np.log10(0.625))

    def test_complex_sgcs_and_beam_gain(self):
        h = np.array([[1 + 0j, 0j], [0j, 1j]])
        np.testing.assert_allclose(complex_sgcs_per_sample(h[:, None, :], h[:, None, :]), [1, 1])
        np.testing.assert_allclose(beam_gain_per_sample(h, h), [1, 1])
        np.testing.assert_allclose(beam_gain_per_sample(h[:1], np.array([[0j, 1 + 0j]])), [0])

    def test_feedback_and_se(self):
        self.assertEqual(feedback_bits(128, 6, 16), 784)
        self.assertAlmostEqual(float(illustrative_se_per_sample(0, np.array([1.0]))[0]), 1.0)

    def test_paired_ci_has_positive_improvement_for_consistent_winner(self):
        mean, low, high = paired_improvement_ci(np.array([0.5] * 20), np.array([1.0] * 20), lower_is_better=True)
        self.assertEqual((mean, low, high), (0.5, 0.5, 0.5))

    def test_zero_power_fails_instead_of_hiding_invalid_samples(self):
        with self.assertRaises(ValueError):
            nmse_per_sample(np.zeros((1, 2)), np.ones((1, 2)))
        with self.assertRaises(ValueError):
            complex_sgcs_per_sample(np.zeros((1, 2), dtype=np.complex64), np.ones((1, 2), dtype=np.complex64))

    def test_phase_rotation_is_invariant_for_complex_similarity(self):
        h = np.array([[1 + 2j, 3 - 1j]])
        np.testing.assert_allclose(complex_sgcs_per_sample(h, h * 1j), [1])
        np.testing.assert_allclose(beam_gain_per_sample(h, h * 1j), [1])

    def test_nonfinite_aggregate_and_fractional_bits_rejected(self):
        for invalid in (np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                nmse_db_from_samples(invalid)
        with self.assertRaises(ValueError):
            feedback_bits(1.5, 4)
        with self.assertRaises(ValueError):
            illustrative_se_per_sample(np.inf, np.ones(1))


if __name__ == "__main__":
    unittest.main()
