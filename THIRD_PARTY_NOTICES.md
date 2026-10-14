# Third-Party Notices and Provenance

## Source selection

This repository is a selected and adapted collection of existing local wireless experiment utilities. It includes the synthetic channel generator, CSI and beam architecture comparisons, NumPy learning implementations, quantized CSI and fixed sparse feedback components, evaluation utilities, and synthetic regression tests. Package integration, terminology corrections and numerical checks were added for this standalone collection.

The selected local source files do not declare a standalone source-code license or provide a complete author-rights record. Their presence in a local workspace, or the absence of a copyright header, does not establish sole authorship or redistribution rights. No blanket MIT, Apache or other source-code license is added to this candidate. Confirm ownership and any applicable institutional or third-party restrictions before public release or licensing. Keeping a repository private is not itself a copyright clearance.

No complete upstream implementation, dataset, trained model, paper PDF or third-party repository is bundled. In particular, externally authored foundation-model repositories and their assets are outside this collection. Where third-party notices exist, they must remain with the affected material; a learning-only disclaimer does not replace license obligations.

Local source-to-candidate records and the privacy review are maintained outside this repository. Repository history starts from the selected candidate and does not import earlier research history or private workspace records.

## CsiNet convention reference

- C.-K. Wen, W.-T. Shih and S. Jin, *Deep Learning for Massive MIMO CSI Feedback*, IEEE Wireless Communications Letters, 2018. [Paper](https://ieeexplore.ieee.org/document/8322184/).
- Authors' implementation: [sydney222/Python_CsiNet](https://github.com/sydney222/Python_CsiNet).
- Layout convention: [CS-CsiNet_onlytest.py](https://github.com/sydney222/Python_CsiNet/blob/master/CS-CsiNet_onlytest.py), real/imaginary reconstruction with subtraction of `0.5`.

This is a methodology and layout citation. The authors' implementation and assets are not vendored. GitHub did not identify a repository-wide license during the local review on 2026-10-14. Public availability and a citation do not grant redistribution or relicensing permission. The local CSI architectures and synthetic experiments must not be presented as the original CsiNet implementation or its published results.

## Architecture and numerical methods

[REFERENCES.md](REFERENCES.md) maps each architecture family and numerical method to its source, including residual networks, recurrent models, dilated convolutions, Transformers, MRT, Adam, uniform quantization and straight-through estimation. These references credit concepts; they are not a claim that the code is the authors' official implementation, or that the papers' experimental results have been reproduced.

## Separately installed dependencies

Dependencies are installed separately and are not vendored. Their own distributions contain the applicable licenses and notices:

| Dependency | Upstream license information | Use |
| --- | --- | --- |
| NumPy | [License](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Numerical arrays, synthetic channel generation and manual gradient implementations. |
| SciPy | [License](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | MAT metadata reading and synthetic MAT fixtures. |
| h5py | [License](https://github.com/h5py/h5py/blob/master/LICENSE) | HDF5 metadata inspection; the underlying HDF5 distribution has its own notices. |
| PyTorch | [License](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Neural network layers, automatic differentiation and optimization. |

No citation or dependency implies affiliation with, endorsement by, or permission from its authors or maintainers.
