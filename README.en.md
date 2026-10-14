# Wireless AI Experiment Suite

[中文](README.md) | [English](README.en.md)

This project studies CSI reconstruction, quantized feedback and single-user beam prediction on synthetic multipath channels. It provides interchangeable neural baselines, fixed sparse feedback, and consistent interfaces for data splits, training and evaluation.

## Research questions

The experiments address three related questions:

1. **CSI representation and reconstruction:** How do convolutional, recurrent and attention-based encoders affect reconstruction error and spatial-direction retention at the same input layout and latent dimension?
2. **Finite-feedback reconstruction:** How do a four-bit quantized autoencoder and train-fitted fixed sparse feedback compare at the same nominal feedback budget?
3. **Single-user beam prediction:** How well can different networks learn the complex-channel-to-MRT mapping and preserve normalized beam gain?

These are currently separate experiments. Beam predictors receive one subband of the true synthetic channel; reconstructed CSI is not yet connected to their inputs.

## Method overview

```text
Correlated multipath MISO channel → independent train / validation / test samples
                                    ├─ continuous encoding → CSI → NMSE / SGCS
                                    ├─ four-bit latent quantization → CSI → bits / NMSE
                                    ├─ fixed sparse feedback → CSI → bits / NMSE
                                    └─ single-subband channel → beam → normalized gain
```

The generator includes exponential spatial correlation, multipath delays and power profiles, optional Rician fading and shadowing. CSI models cover CNN, residual CNN, LSTM, GRU, RNN, TCN and Transformer encoders. Beam prediction uses MLP, residual MLP and the corresponding sequence-model families. NumPy implementations expose manual backpropagation and Adam for gradient checks and implementation comparisons.

Quantized reconstruction uses 16-level real latents and a straight-through training gradient. Sparse feedback fits its mean, fixed coefficient locations and quantizer scales only on training samples. [REFERENCES.md](REFERENCES.md) credits the underlying architectures and algorithms. The contribution is wireless-task implementation, experiment interfaces and model integration; the underlying network families are not proposed as new methods.

## Experimental design

| Experiment | Input representation | Controlled variables | Test metrics |
| --- | --- | --- | --- |
| Continuous CSI reconstruction | 12 subbands × 32 antennas, real/imaginary channels | Layout, latent dimension, data seed and training budget | NMSE, complex SGCS |
| Quantized / sparse feedback | 32 subbands × 32 antennas | Data seed, equal nominal feedback budget and train-only fitting | Feedback bits, NMSE, complex SGCS |
| Beam prediction | First-subband channel over 32 antennas | Input samples, training budget and MRT targets | Normalized beam gain |

Training, validation and test samples use independent random streams. Neural models update parameters on training samples, select an epoch by validation MSE, restore that state and then compute test metrics. Sparse parameters are fitted on training samples only. Paired bootstrap utilities support comparisons on corresponding test samples.

Continuous latent dimensions are not feedback bits. Quantized latents count four bits per real value; sparse feedback counts eight per complex coefficient, assuming the fixed mean, locations and scales are shared offline. The two CSI groups have different input sizes, so their scores cannot be compared directly across groups. See the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for metric definitions, budget assumptions and layout contracts.

## Code structure

| Path | Purpose |
| --- | --- |
| `wireless_ai/channel.py` | Correlated multipath channel generation |
| `wireless_ai/csi_models.py` | CSI encoding, decoding and architecture selection |
| `wireless_ai/beam_models.py` | Single-user beam prediction and architecture selection |
| `wireless_ai/quantized_csi.py` | Four-bit quantized CNN autoencoder |
| `wireless_ai/sparse_feedback.py` | Fixed sparse feedback and reconstruction |
| `wireless_ai/numpy_csi.py`, `numpy_beam.py` | Manual gradients, Adam and parameter save/load |
| `wireless_ai/workflow.py` | Synthetic experiments, validation selection and test evaluation |
| `wireless_ai/validation/` | Metrics, bootstrap, HT conversion and MAT metadata checks |
| `tests/` | Forward/backward, gradient, quantization-budget and convention tests |

## Setup and experiments

Use Python 3.9 from the repository directory:

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -B -X utf8 -m unittest discover -s tests -v
```

Start with the default small CPU runs to check the environment:

```powershell
py -3.9 -B -X utf8 -m wireless_ai --task csi --arch gru --latent-dim 16
py -3.9 -B -X utf8 -m wireless_ai --task beam --arch mlp
py -3.9 -B -X utf8 -m wireless_ai --task quantized --feedback-bits 128
py -3.9 -B -X utf8 -m wireless_ai --task sparse --feedback-bits 128
```

NumPy alternatives use `--task csi-numpy` and `--task beam-numpy`. Configure sample counts, epochs and seed with `--train-samples`, `--val-samples`, `--test-samples`, `--epochs` and `--seed`; epochs affect neural models only. Optional `--out` saves metrics to a new JSON file and refuses existing targets. Generated outputs and model weights are not distributed with the repository.

## Current research status

The listed models and experiment interfaces are implemented. Thirty-five synthetic tests cover forward/backward execution, NumPy gradients, quantization and feedback budgets, metric definitions and output safety. Six task entry points have passed small CPU checks. These checks validate implementation behavior; formal comparisons of convergence, cross-scenario generalization or method superiority are not provided yet.

The simulator is a simplified, uncalibrated statistical model. Experiments cover synthetic frequency-domain CSI and single-user beam alignment, without RF capture, hardware deployment or a complete protocol link. Performance conclusions require sufficient training, multiple random seeds and independent scenario evaluation.

## References and licensing

The CSI feedback task draws on the [CsiNet paper](https://ieeexplore.ieee.org/document/8322184/) and [authors' code](https://github.com/sydney222/Python_CsiNet). Complete MRT, architecture, quantization and statistical references are in [REFERENCES.md](REFERENCES.md). Models are local task implementations and adaptations; cite the respective original work when using its methods.

The repository's code and original project documentation are licensed under the [MIT License](LICENSE), allowing use, modification and redistribution, including commercial use, with the copyright and permission notice retained. Third-party dependencies, cited papers, external datasets and model weights retain their respective terms and are outside this MIT grant. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source and dependency information.
