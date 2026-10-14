import tempfile
import unittest
from pathlib import Path

import numpy as np

from wireless_ai.numpy_csi import CSIFCAE, Adam
from wireless_ai.numpy_beam import BeamformerMLP


class NumpyModelTests(unittest.TestCase):
    def test_manual_gradients_finite_difference(self):
        rng = np.random.RandomState(12)
        for model in (CSIFCAE(4, 5, latent_dim=2, seed=4), BeamformerMLP(4, 5, seed=4)):
            with self.subTest(model=type(model).__name__):
                # Float64 and positive biases keep finite differences away from ReLU kinks.
                for key, value in model.params_dict().items():
                    setattr(model, key, value.astype(np.float64))
                model.b1[:] = 2
                if isinstance(model, BeamformerMLP):
                    model.b2[:] = 2
                x = rng.normal(scale=0.05, size=(2, 4))
                y, cache = model.forward(x)
                upstream = rng.normal(size=y.shape)
                gradients = model.backward(cache, upstream)
                epsilon = 1e-5
                for key, param in model.params_dict().items():
                    self.assertTrue(np.isfinite(gradients[key]).all())
                    for index in np.ndindex(param.shape):
                        original = param[index]
                        param[index] = original + epsilon
                        plus = np.sum(model.forward(x)[0] * upstream)
                        param[index] = original - epsilon
                        minus = np.sum(model.forward(x)[0] * upstream)
                        param[index] = original
                        self.assertAlmostEqual(float(gradients[key][index]),
                                               float((plus - minus) / (2 * epsilon)), delta=2e-6)

    def test_npz_roundtrip_and_adam(self):
        x = np.random.RandomState(2).normal(size=(2, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            model = CSIFCAE(4, 5, latent_dim=2)
            y, cache = model.forward(x)
            path = str(Path(directory) / 'csi.npz')
            model.save_npz(path, best_cos=0.3)
            restored, latent_dim, score = CSIFCAE.load_npz(path)
            self.assertEqual(latent_dim, 2)
            self.assertAlmostEqual(score, 0.3)
            np.testing.assert_array_equal(y, restored.forward(x)[0])
            original = model.W3.copy()
            Adam(model.params_dict()).step(model.backward(cache, (y - x) / y.size))
            self.assertFalse(np.array_equal(model.W3, original))
            beam = BeamformerMLP(4, 5, seed=4)
            output, _ = beam.forward(x)
            np.testing.assert_allclose((output ** 2).sum(1), np.ones(2), atol=1e-5)
            path = str(Path(directory) / 'beam.npz')
            beam.save_npz(path, best_gain=0.7)
            restored_beam, score = BeamformerMLP.load_npz(path)
            np.testing.assert_array_equal(output, restored_beam.forward(x)[0])
            self.assertAlmostEqual(score, 0.7)

    def test_small_beam_normalization_gradient(self):
        model = BeamformerMLP(4, 5, seed=4)
        for key, value in model.params_dict().items():
            setattr(model, key, value.astype(np.float64))
        model.W3[:] = 0
        model.b3[:] = 1e-4
        x = np.ones((1, 4))
        _, cache = model.forward(x)
        upstream = np.arange(1, 5).reshape(1, 4)
        analytic = model.backward(cache, upstream)['b3']
        for index in range(4):
            model.b3[index] += 1e-8
            plus = np.sum(model.forward(x)[0] * upstream)
            model.b3[index] -= 2e-8
            minus = np.sum(model.forward(x)[0] * upstream)
            model.b3[index] += 1e-8
            self.assertAlmostEqual(float(analytic[index]), float((plus - minus) / 2e-8), delta=0.005)

    def test_shapes_and_stable_sigmoid(self):
        with self.assertRaises(ValueError):
            CSIFCAE(4, 5, latent_dim=0)
        with self.assertRaises(ValueError):
            BeamformerMLP(0, 5)
        model = CSIFCAE(4, 5, latent_dim=2)
        with self.assertRaises(ValueError):
            model.forward(np.zeros((2, 3)))
        model.b2[:] = -1e5
        with np.errstate(over='raise', invalid='raise'):
            y, _ = model.forward(np.zeros((2, 4)))
        self.assertTrue(np.isfinite(y).all())


if __name__ == '__main__':
    unittest.main()
