import unittest

import numpy as np

from wireless_ai.validation.manifest import cost2100_ht_to_complex, cost2100_layout_manifest


class ManifestTests(unittest.TestCase):
    def test_cost2100_offset_and_channel_order(self):
        ht = np.full((2, 2048), 0.5, dtype=np.float64)
        ht[0, 0] = 0.75
        ht[0, 1024] = 0.25
        result = cost2100_ht_to_complex(ht)
        self.assertEqual(result.shape, (2, 32, 32))
        self.assertEqual(result.dtype, np.complex64)
        self.assertEqual(result[0, 0, 0], 0.25 - 0.25j)
        self.assertEqual(result[1, 0, 0], 0j)
        self.assertEqual(ht[0, 0], 0.75)

    def test_wrong_shape_fails_and_manifest_names_domain(self):
        with self.assertRaises(ValueError):
            cost2100_ht_to_complex(np.zeros((2, 1024)))
        self.assertIn("angle-delay", cost2100_layout_manifest()["domain"])


if __name__ == "__main__":
    unittest.main()
