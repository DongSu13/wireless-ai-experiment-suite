import unittest

import torch

from wireless_ai.csi_models import CSI_ARCH_CHOICES, build_csi_model
from wireless_ai.beam_models import BEAM_ARCH_CHOICES, build_beam_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_all_csi_architectures_forward_backward(self):
        x = torch.randn(2, 2, 12, 32)
        for arch in CSI_ARCH_CHOICES:
            with self.subTest(arch=arch):
                model = build_csi_model(arch, latent_dim=5)
                reconstruction, latent = model(x)
                self.assertEqual(tuple(reconstruction.shape), tuple(x.shape))
                self.assertEqual(tuple(latent.shape), (2, 5))
                self.assertTrue(torch.all((latent > 0) & (latent < 1)))
                (reconstruction - x).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.eval()
                with torch.no_grad():
                    _, latent_eval = model(x)
                    torch.testing.assert_close(model.encode(x), latent_eval)
                    torch.testing.assert_close(model.decode(latent_eval), model(x)[0])
                self.assertTrue(torch.all((latent_eval > 0) & (latent_eval < 1)))

    def test_all_beam_architectures_forward_backward(self):
        x = torch.randn(2, 2, 32)
        for arch in BEAM_ARCH_CHOICES:
            with self.subTest(arch=arch):
                model = build_beam_model(arch)
                beam = model(x)
                self.assertEqual(tuple(beam.shape), tuple(x.shape))
                torch.testing.assert_close(beam.square().sum((1, 2)), torch.ones(2))
                (beam - x).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_mrt_phase_convention(self):
        # y = h^H w x + n: MRT uses h, not conj(h).
        h = torch.tensor([[1 + 2j, 3 - 4j]])
        w = h / h.abs().square().sum(1, keepdim=True).sqrt()
        gain = (h.conj() * w).sum(1).abs().square()
        torch.testing.assert_close(gain, h.abs().square().sum(1))
        wrong = (h.conj() * w.conj()).sum(1).abs().square()
        self.assertLess(float(wrong), float(gain))

    def test_invalid_registry_arguments_and_input_shapes(self):
        for arch in CSI_ARCH_CHOICES:
            with self.subTest(csi=arch):
                with self.assertRaises(ValueError):
                    build_csi_model(arch, latent_dim=0)
                with self.assertRaises(ValueError):
                    build_csi_model(arch)(torch.zeros(1, 2, 11, 32))
        for arch in BEAM_ARCH_CHOICES:
            with self.subTest(beam=arch):
                with self.assertRaises(ValueError):
                    build_beam_model(arch, n_ant=0)
                with self.assertRaises(ValueError):
                    build_beam_model(arch)(torch.zeros(1, 2, 31))
        with self.assertRaises(ValueError):
            build_csi_model('unknown')
        with self.assertRaises(ValueError):
            build_beam_model('cnn')


if __name__ == '__main__':
    unittest.main()
