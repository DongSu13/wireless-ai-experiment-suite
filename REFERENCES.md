# Method and Architecture References

These references credit the methods and architecture families used in the experiments. They do not imply that the implementations, hyperparameters, datasets or reported results reproduce those papers. No affiliation or endorsement is implied.

## Implemented experiment components

| Component | Reference | Scope in this repository |
| --- | --- | --- |
| CSI autoencoder task | C.-K. Wen, W.-T. Shih and S. Jin, *Deep Learning for Massive MIMO CSI Feedback*, IEEE Wireless Communications Letters, 2018. [Paper](https://ieeexplore.ieee.org/document/8322184/), [authors' code](https://github.com/sydney222/Python_CsiNet). | Context for learning a compressed CSI representation. The models here are local architecture comparisons, not the published CsiNet network or a reproduction of its results. |
| CNN encoder | Y. LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 1998. [Paper](https://doi.org/10.1109/5.726791). | Convolutional feature extraction; different task and architecture. |
| Residual CNN / residual MLP | K. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016. [Paper](https://arxiv.org/abs/1512.03385). | Residual connections; these compact networks are not a named ResNet variant. |
| MLP / simple RNN and backpropagation | D. E. Rumelhart, G. E. Hinton and R. J. Williams, *Learning Representations by Back-Propagating Errors*, Nature, 1986. [Paper](https://doi.org/10.1038/323533a0); PyTorch [RNN documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html). | Fully connected models and a tanh recurrent baseline. NumPy implementations explicitly calculate gradients. |
| LSTM | S. Hochreiter and J. Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997. [Paper](https://doi.org/10.1162/neco.1997.9.8.1735). | Uses PyTorch's LSTM layer in a pooled bidirectional encoder. |
| GRU | K. Cho et al., *Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation*, EMNLP, 2014. [Paper](https://arxiv.org/abs/1406.1078). | Uses PyTorch's GRU layer in a pooled bidirectional encoder. |
| Dilated convolution / TCN family | S. Bai, J. Z. Kolter and V. Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018. [Paper](https://arxiv.org/abs/1803.01271). | Short stacks of residual dilated convolutions with symmetric padding. They are noncausal and are not an exact implementation of the paper's causal TCN. |
| Transformer encoder | A. Vaswani et al., *Attention Is All You Need*, NeurIPS, 2017. [Paper](https://arxiv.org/abs/1706.03762). | Uses PyTorch Transformer encoder layers, sinusoidal positions and pooling; no original encoder-decoder translation model. |
| Group normalization | Y. Wu and K. He, *Group Normalization*, ECCV, 2018. [Paper](https://arxiv.org/abs/1803.08494). | Group normalization layers in the convolutional models. |
| Batch normalization | S. Ioffe and C. Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML, 2015. [Paper](https://arxiv.org/abs/1502.03167). | Batch normalization in the beam dilated-convolution model. |
| Maximum-ratio transmission (MRT) | T. K. Y. Lo, *Maximum Ratio Transmission*, IEEE Transactions on Communications, 1999. [Paper](https://doi.org/10.1109/26.795811). | With the convention `y = h^H w x + n`, the supervised target is `w = h / norm(h)`. Normalized beam gain is an alignment metric, not measured throughput. |
| Adam | D. P. Kingma and J. Ba, *Adam: A Method for Stochastic Optimization*, ICLR, 2015. [Paper](https://arxiv.org/abs/1412.6980). | PyTorch Adam and manual NumPy Adam. The NumPy weight penalty is added to gradients; it is not decoupled AdamW. |
| Uniform scalar quantization | R. M. Gray and D. L. Neuhoff, *Quantization*, IEEE Transactions on Information Theory, 1998. [Paper](https://doi.org/10.1109/18.720541). | Fixed 16-level quantization for the quantized CSI and sparse-coefficient experiments. Continuous latent dimensions in the other models are not feedback bits. |
| Straight-through estimator | Y. Bengio, N. Leonard and A. Courville, *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*, 2013. [Paper](https://arxiv.org/abs/1308.3432). | The quantized autoencoder uses hard levels in its forward pass and a surrogate gradient through its bounded latent. This does not establish optimality or an exact codec. |
| Bootstrap confidence intervals | B. Efron, *Bootstrap Methods: Another Look at the Jackknife*, The Annals of Statistics, 1979. [Paper](https://doi.org/10.1214/aos/1176344552). | Resampling paired, independently resampleable evaluation samples. Dependence across users, trajectories or captures requires a different resampling design. |
| Simplified channel simulation | D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*, Cambridge University Press, 2005. [Authors' book page](https://web.stanford.edu/~dntse/wireless_book.html). | General background for multipath fading, array channels and beamforming. The correlated Rayleigh/Rician simulator here is an educational local model, not a calibrated 3GPP or QuaDRiGa implementation. |

The CSI sequence models operate along subband indices, and the beam sequence models operate along antenna indices. These axes do not make either task a temporal forecasting experiment. Real and imaginary values are the input channels; antenna tokens are not amplitude/phase pairs.

The fixed sparse baseline fits its mean, coefficient locations and scalar ranges on training samples. Its shared offline parameters are assumed known to both ends. It is neither a standardized Type-II codebook nor a quantized PCA implementation. Nominal sample payload accounting excludes offline model distribution and transport overhead; no packed bitstream implementation is included.

## Optional layout compatibility

The validation utility can convert the public CsiNet `HT` layout `[sample, 2048]` by reshaping into real/imaginary `32 x 32` blocks and subtracting `0.5`. Its convention reference is the authors' [CS-CsiNet_onlytest.py](https://github.com/sydney222/Python_CsiNet/blob/master/CS-CsiNet_onlytest.py). That layout describes preprocessed angle-delay CSI; it is not the simulator's raw subband-domain data. The synthetic square-layout demonstrations do not emulate the COST2100 dataset or its preprocessing distribution.

The validation SGCS is squared complex cosine similarity, including complex phase invariance. It must not be equated to the authors' unsquared `rho` metric. Public dataset acquisition instructions can be found in the authors' [data preparation section](https://github.com/sydney222/Python_CsiNet#step2-data-preparation); this repository includes no dataset or download automation.

Citations describe intellectual context. They do not grant permission to copy or redistribute another project's source, datasets or weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licensing boundaries.
