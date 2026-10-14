# Wireless AI Experiment Suite

[中文](README.md)

An educational suite integrating locally developed synthetic-channel experiments, CSI reconstruction, single-user beam prediction, quantized feedback comparisons and numerical validation. The contribution is implementation and experiment integration, interfaces, manual gradients, consistent conventions and tests. The project does not claim invention of the underlying architectures, exact paper reproduction or published benchmark performance.

This repository is a private release candidate. Publication requires separate authorization. Only code, synthetic tests and documentation are included; real data, weights, internal reports and personal material are excluded.

## Modules

| Module | Purpose |
| --- | --- |
| `channel.py` | Simplified correlated multipath MISO frequency-domain channel with optional Rician component and shadowing |
| `csi_models.py` | CNN, residual CNN, LSTM, GRU, RNN, TCN and Transformer continuous-latent autoencoders |
| `beam_models.py` | MLP, residual MLP, LSTM, GRU, RNN, TCN and Transformer beam predictors |
| `numpy_csi.py`, `numpy_beam.py` | Manual backpropagation, Adam, inference and NPZ parameter interfaces |
| `quantized_csi.py` | CNN autoencoder with 16-level latent quantization during both training and evaluation |
| `sparse_feedback.py` | Train-fitted fixed coefficient locations, with 4 bits each for real and imaginary residuals |
| `validation/` | NMSE, complex SGCS, beam alignment, feedback budgets, paired bootstrap and MAT metadata checks |

## Quick start

From this directory, using Python 3.9:

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -B -X utf8 -m unittest discover -s tests -v
py -3.9 -B -X utf8 -m wireless_ai --task csi-numpy
py -3.9 -B -X utf8 -m wireless_ai --task beam-numpy
py -3.9 -B -X utf8 -m wireless_ai --task csi --arch gru --latent-dim 16
py -3.9 -B -X utf8 -m wireless_ai --task beam --arch mlp
py -3.9 -B -X utf8 -m wireless_ai --task quantized --feedback-bits 128
py -3.9 -B -X utf8 -m wireless_ai --task sparse --feedback-bits 128
```

The CPU CLI defaults to 32/8/8 train/validation/test samples and two epochs: a workflow check, not a meaningful benchmark. Splits use separate random streams. Neural models select an epoch by validation MSE and report test metrics only afterward. CSI and beam learning are separate tasks; beam input is the true synthetic channel's first subband. Reconstruction is not connected to beam prediction.

An optional `--out` writes a new JSON file outside the repository and refuses overwrites. No weights are saved by default. Model classes support forward inference; NumPy classes also save/load NPZ parameters. Load only trusted files matching the documented model interface.

## Conventions and limits

- The simulator is simplified and uncalibrated; it is not a 3GPP/QuaDRiGa implementation. CSI shape is `[sample,12,32]`. Quantized/sparse CLI examples use `[sample,32,32]`, still frequency-domain subbands/antennas, not COST2100 angle-delay data.
- Rician K is the expected LoS/scattered power ratio before shadowing. Both components have unit expected per-antenna power.
- Continuous `latent_dim` is a floating-point dimension, not a bit count. Quantized latents use four nominal bits per real value; sparse feedback uses eight per complex coefficient. There is no packed wire codec, entropy coding or protocol overhead accounting.
- Sparse mean, indices and scales are fitted only on training data and shared offline. Their refresh/transmission cost is excluded from per-sample budgets. This is not a standards-compliant Type-II method.
- With `y=hᴴwx+n`, MRT is `w=h/||h||`. Normalized beam gain measures direction alignment, not throughput or multiuser optimization.
- NMSE aggregates as `10log10(mean(linear per-sample NMSE))`. Complex SGCS averages squared complex cosines within each sample, with antennas last. Choose the domain and axes explicitly. Zero-power vectors cannot be scored.
- Paired bootstrap assumes independently resampleable pairs; time/trajectory dependence needs grouped or block resampling.
- The CsiNet-style `HT` converter expects `[sample,2048]`, splits real/imaginary values and subtracts 0.5. Its `[sample,angle,delay]` output is not directly frequency-domain SGCS input. MAT reports default to basenames, but filenames and variable names still need review before sharing.
- Synthetic tests do not validate real-world wireless performance, exact GPU reproducibility or reproduction of paper results.

## Attribution and rights

[REFERENCES.md](REFERENCES.md) maps methods to conceptual papers and relevant author code; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) explains provenance and boundaries. Generic architecture citations do not imply exact reproduction of the authors' tasks.

No complete third-party model repositories, data or pretrained weights are bundled or downloaded. Obtain public data separately from its authors and check applicable terms. Nonpublic data is excluded. Citation and educational use statements do not replace licensing permission. No blanket open-source license has been granted to all code; rights must be confirmed before public release.
