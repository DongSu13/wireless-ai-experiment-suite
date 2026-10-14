import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from wireless_ai.validation.data_audit import inspect_mat, main


class DataAuditTests(unittest.TestCase):
    def test_reports_complex_mat_shape_without_changing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.mat"
            savemat(str(path), {"H": np.ones((3, 12, 32), dtype=np.complex64)})
            before = path.read_bytes()
            result = inspect_mat(path, hash_content=True)
            self.assertEqual(result["container"], "mat_v5")
            self.assertEqual(result["variables"][0]["shape"], [3, 12, 32])
            self.assertEqual(result["variables"][0]["dtype"], "single")
            self.assertEqual(result["path"], "sample.mat")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(result["sha256"]), 64)

    def test_cli_refuses_existing_output_before_inspecting_input(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'audit.json'
            output.write_text('keep this', encoding='utf-8')
            error = io.StringIO()
            with patch('sys.argv', ['audit', '--out', str(output), 'missing-input.mat']):
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    main()
            self.assertIn('output already exists', error.getvalue())
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep this')

    def test_cli_stdout_uses_only_output_basename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.mat'
            output = Path(folder) / 'audit.json'
            savemat(str(path), {'H': np.ones((2, 3))})
            stdout = io.StringIO()
            with patch('sys.argv', ['audit', '--out', str(output), str(path)]):
                with contextlib.redirect_stdout(stdout):
                    main()
            self.assertNotIn(folder, stdout.getvalue())
            self.assertIn('audit.json', stdout.getvalue())

    def test_hdf5_metadata_reports_shape_without_values(self):
        import h5py

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.mat'
            with h5py.File(str(path), 'w') as file:
                file.create_dataset('nested/H', data=np.ones((2, 3)))
            before = path.read_bytes()
            result = inspect_mat(path)
            self.assertEqual(result['container'], 'hdf5_mat')
            self.assertEqual(result['variables'], [{'name': 'nested/H', 'shape': [2, 3], 'dtype': 'float64'}])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
