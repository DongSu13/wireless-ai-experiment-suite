import unittest

import numpy as np

from wireless_ai.channel import generate_hcsi_down_all


class ChannelTests(unittest.TestCase):
    def test_seed_shape_and_dtype(self):
        a = generate_hcsi_down_all(3, seed=7)
        self.assertEqual(a.shape, (3, 12, 32))
        self.assertEqual(a.dtype, np.complex64)
        np.testing.assert_array_equal(a, generate_hcsi_down_all(3, seed=7))

    def test_los_and_diffuse_have_matching_per_antenna_power(self):
        # For high K, each antenna has unit power before shadowing;
        # normalization by sqrt(n_tx) would incorrectly suppress it.
        for n_tx in (1, 8, 32):
            h = generate_hcsi_down_all(3000, n_tx=n_tx, n_subbands=1,
                                      n_taps=1, spatial_rho=0.2,
                                      k_factor_db=60, shadowing_std_db=0)
            self.assertAlmostEqual(float(np.mean(abs(h) ** 2)), 1.0, delta=0.005)
        diffuse = generate_hcsi_down_all(10000, n_tx=8, n_subbands=1,
                                        n_taps=1, spatial_rho=0.2,
                                        k_factor_db=None, shadowing_std_db=0)
        self.assertAlmostEqual(float(np.mean(abs(diffuse) ** 2)), 1.0, delta=0.03)

    def test_reject_invalid_parameters(self):
        cases = [dict(n_samples=0), dict(n_samples=1.5),
                 dict(n_samples=True), dict(n_samples=1, n_tx=0),
                 dict(n_samples=1, n_taps=2.5), dict(n_samples=1, seed=-1),
                 dict(n_samples=1, spatial_rho=np.nan),
                 dict(n_samples=1, delay_max=-1), dict(n_samples=1, tap_decay=-1),
                 dict(n_samples=1, shadowing_std_db=-1),
                 dict(n_samples=1, spatial_rho=0.1j),
                 dict(n_samples=1, delay_max='bad'), dict(n_samples=1, tap_decay=True),
                 dict(n_samples=1, los_angle_spread=np.inf),
                 dict(n_samples=1, k_factor_db=np.nan)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_hcsi_down_all(**kwargs)


if __name__ == '__main__':
    unittest.main()
