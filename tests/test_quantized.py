import unittest

import torch

from wireless_ai.quantized_csi import QuantizedCsiAutoencoder, batch_nmse, four_bit_ste


class QuantizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_quantizer_has_sixteen_levels_and_finite_gradients(self):
        z = torch.linspace(-10, 10, 1000, requires_grad=True)
        q = four_bit_ste(z)
        self.assertEqual(torch.unique(q.detach().round(decimals=6)).numel(), 16)
        torch.testing.assert_close((q + 1) * 7.5, ((q + 1) * 7.5).round(), atol=2e-6, rtol=0)
        q.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_payload_quantized_in_train_and_eval_and_roundtrip(self):
        torch.manual_seed(0)
        model = QuantizedCsiAutoencoder(128)
        x = torch.randn(2, 2, 32, 32)
        for mode in (True, False):
            model.train(mode)
            code = model.encode(x)
            self.assertEqual(tuple(code.shape), (2, 32))
            torch.testing.assert_close((code + 1) * 7.5, ((code + 1) * 7.5).round(), atol=2e-6, rtol=0)
            prediction = model.decode(code)
            self.assertEqual(prediction.shape, x.shape)
            loss = batch_nmse(prediction, x).mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            model.zero_grad(set_to_none=True)

    def test_invalid_budget_and_shape_rejected(self):
        for budget in (0, -4, 3, 4.0, True):
            with self.assertRaises(ValueError):
                QuantizedCsiAutoencoder(budget)
        with self.assertRaises(ValueError):
            QuantizedCsiAutoencoder(4)(torch.ones(1, 2, 16, 16))

    def test_nmse_zero_power_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            batch_nmse(torch.ones(1, 2), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            batch_nmse(torch.full((1, 2), float('nan')), torch.ones(1, 2))


if __name__ == '__main__':
    unittest.main()
