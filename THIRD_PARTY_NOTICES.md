# Sources and Third-Party Notices

The repository integrates local channel simulation, neural task models, NumPy learning implementations and numerical evaluation utilities. [REFERENCES.md](REFERENCES.md) credits their methodological background and describes each implementation's relationship to its references. Conceptual references do not imply exact reproduction, affiliation or endorsement.

## CSI feedback and layout

C.-K. Wen, W.-T. Shih and S. Jin, *Deep Learning for Massive MIMO CSI Feedback*, IEEE Wireless Communications Letters, 2018: [paper](https://ieeexplore.ieee.org/document/8322184/) and [authors' implementation](https://github.com/sydney222/Python_CsiNet).

The optional HT conversion follows the real/imaginary layout and subtraction of 0.5 shown in [CS-CsiNet_onlytest.py](https://github.com/sydney222/Python_CsiNet/blob/master/CS-CsiNet_onlytest.py). Networks here are local task implementations rather than the original CsiNet network. The authors' source, datasets and weights are not bundled. GitHub did not identify a repository-wide CsiNet license during verification; its public availability does not authorize redistribution or relicensing.

## Dependencies

Dependencies are installed separately. Their own distributions retain the applicable licenses and notices.

| Dependency | Upstream license information | Use |
| --- | --- | --- |
| NumPy | [License](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Arrays, channel generation and manual gradients |
| SciPy | [License](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | MAT metadata and synthetic fixtures |
| h5py | [License](https://github.com/h5py/h5py/blob/master/LICENSE) | HDF5 metadata inspection; HDF5 has separate notices |
| PyTorch | [License](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Neural layers, differentiation and optimization |

## Source-code licensing

No unified source-code license is specified. Third-party code, data and weights remain subject to their respective terms. Citation and research or educational use do not replace permission to redistribute protected material. Retain any applicable copyright and license notices with the affected material.
