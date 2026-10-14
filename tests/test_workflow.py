import tempfile
import unittest
from pathlib import Path

import numpy as np

from wireless_ai.workflow import make_splits, main, parser


class WorkflowTests(unittest.TestCase):
    def test_separate_reproducible_splits(self):
        a = make_splits(4, 2, 3, 42)
        b = make_splits(4, 2, 3, 42)
        self.assertEqual([len(x) for x in a], [4, 2, 3])
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0][:2], a[1]))
        self.assertFalse(np.array_equal(a[1], a[2][:2]))

    def test_invalid_counts_rejected(self):
        with self.assertRaises(SystemExit):
            parser().parse_args(['--train-samples', '0'])
        with self.assertRaises(SystemExit):
            parser().parse_args(['--epochs', '-1'])

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'summary.json'
            output.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                main(['--out', str(output)])
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep')

    def test_tiny_numpy_cli_finite(self):
        result = main(['--task', 'csi-numpy', '--train-samples', '4',
                       '--val-samples', '2', '--test-samples', '2', '--epochs', '1'])
        self.assertEqual(result['evaluation_split'], 'test')
        self.assertTrue(np.isfinite(result['nmse_db']))
        self.assertIn('latent_dim', result)
        self.assertNotIn('feedback_bits', result)


if __name__ == '__main__':
    unittest.main()
