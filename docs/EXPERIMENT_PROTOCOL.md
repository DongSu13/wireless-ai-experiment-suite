# Experiment Protocol

This document specifies the implemented data, evaluation and feedback conventions.

## Synthetic channel and splits

The simulator returns complex64 `[sample, subband, antenna]` arrays. Defaults are 12 subbands, 32 antennas, 8 taps, exponential spatial correlation 0.82, normalized maximum delay 1.0, tap decay 0.55, Rician K of 6 dB, angle spread pi/3 and shadowing standard deviation 0.8 dB. Quantized/sparse CLI tasks use 32 subbands. These are frequency-domain arrays; square shape does not imply angle-delay preprocessing.

LoS and scattered components have equal expected per-antenna power before mixing. K specifies their expected power ratio before sample-level shadowing. The model is not calibrated to a deployment or a 3GPP/QuaDRiGa channel profile.

The CLI defaults to 32/8/8 training/validation/test samples, two epochs and seed zero. These defaults are smoke experiments. Three independently spawned random streams generate the splits. Within the same layout and seed, changing an architecture does not change its input samples. Direct library users must ensure their own split isolation.

Neural training uses full-batch CPU updates with Adam at learning rate 0.001 and selects the epoch with lowest validation MSE. PyTorch gradients are clipped at norm five; NumPy Adam includes a gradient-coupled weight penalty. Backends are alternative implementations, not an assertion of identical optimization. Match all relevant hyperparameters before treating them as controlled performance comparisons.

## Task definitions

- **Continuous reconstruction:** input `[N,2,12,32]`, with real and imaginary channels; sigmoid latents are continuous values. `latent_dim` counts coordinates, not encoded bits. Registry models return `(reconstruction, latent)` and provide encoding/decoding interfaces.
- **Quantized reconstruction:** input `[N,2,32,32]`; bounded latents are rounded to 16 fixed levels in training and evaluation. Four nominal bits are counted per latent value. Training uses a straight-through surrogate gradient, not the derivative of hard rounding.
- **Fixed sparse reconstruction:** fit the mean, variance-ranked coefficient positions and 0.999-quantile real/imaginary residual scales on training data. Reconstruct from selected coefficients with four bits per real component. The shared state is fixed during evaluation.
- **Beam prediction:** input `[N,2,32]` from the first subband. With `y=h^H w x+n`, supervised MRT is `w=h/norm(h)`. Learned outputs use epsilon-stabilized L2 normalization; zero raw output remains zero. No constant-modulus or RF-chain constraint is imposed.

Quantized and sparse feedback use the same square layout and can be evaluated on matching samples at a common nominal budget. Continuous reconstruction uses a different layout and has no transmission-bit definition. The CLI does not compose reconstruction and beam prediction into a communication link. CSI sequence models use the subband axis and beam sequence models use the antenna axis, not a temporal forecasting history.

## Metrics

| Metric | Definition / interpretation |
| --- | --- |
| Per-sample NMSE | Squared reconstruction-error norm divided by squared truth norm, across all non-sample axes |
| Aggregate NMSE in dB | `10 * log10(mean(linear per-sample NMSE))`, not the mean of sample dB values |
| Complex SGCS | Squared complex cosine along the last antenna axis, averaged over subbands per sample; invariant to global phase rotation |
| Normalized beam gain | Squared magnitude of the complex channel/beam inner product divided by their powers; alignment rather than throughput |
| Illustrative spectral efficiency | Single-user `log2(1 + SNR * alignment)`; excludes interference, pilot, coding and feedback overhead |

Zero-power truth or similarity vectors have undefined scores and are rejected. Aggregate dB NMSE yields negative infinity when all errors are zero; the CLI's strict JSON output rejects nonfinite values. SGCS here is squared, unlike the original CsiNet unsquared rho metric.

Paired bootstrap resamples corresponding scores on the same independently resampleable samples and returns a percentile 95% interval. Time, user or trajectory dependence requires grouped/block resampling; intervals do not correct for multiple comparisons. The CLI reports test averages; bootstrap is available through the evaluation API.

## Feedback accounting

Quantized budgets are positive multiples of four. Sparse budgets are positive multiples of eight and cannot request more coefficients than the fitted state contains. Both methods represent quantized levels in numerical arrays; neither implements a packed wire codec, headers or entropy coding.

Sparse mean, positions and scales are shared offline and excluded from per-sample feedback. Initialization, refresh and scene-change signaling costs must be accounted for in system-level comparisons. This baseline is not a standardized Type-II codebook or quantized PCA method.

## Optional data compatibility and outputs

The HT converter supports the CsiNet convention `[N,2048]`: first 1024 real and last 1024 imaginary values, centered by subtracting 0.5 and reshaped to `[N,32,32]`. The result is truncated angle-delay CSI, not raw OFDM CSI. Choose the domain and antenna axis before scoring SGCS. Public acquisition instructions are linked from [REFERENCES.md](../REFERENCES.md); data and download automation are not bundled.

MAT inspection reads metadata without loading full arrays. HDF5 dimensions are storage axes, with no automatic MATLAB transpose. Hashing reads files in chunks. Reports contain basenames and variable names, which should be reviewed before sharing.

CLI JSON output is optional and never overwrites an existing file. Keep generated outputs outside tracked source. NumPy parameter loading disables pickle and validates shapes/finiteness; its separate save methods can overwrite caller-selected files. The CLI does not save model weights.

## Verification scope

Synthetic tests check numerical contracts, quantization, interfaces and gradients. They do not establish convergence, deployment generalization, calibrated wireless performance, GPU bitwise reproducibility or published paper results. Formal comparisons should report training settings, multiple seeds, independent scenarios and uncertainty alongside metrics.
