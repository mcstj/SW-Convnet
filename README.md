# SW-Convnet
scattering wavelet convolutional network for image classification

# Introduction
This study introduces an Enhanced Scattering Wavelet Convolutional Neural Network (SW-ConvNet), which integrates a wavelet scattering network into a classical deep convolutional network (CNN) framework to augment feature extraction. This integration facilitates the fusion of spatial-domain and frequency-domain features. The scattering coefficients are iteratively computed via the Scattering Wavelet Transform (SCWT) and subsequent modulus operations. The feature vectors derived from these scattering coefficients are capable of capturing intricate frequency-domain information. Through theoretical analysis of the invariance and deformation sensitivity bounds of the SCWT, we demonstrate that SCWT enables the extraction of richer and more discriminative features for signal classification.

We trained the model for 90/120 epochs using the Stochastic Gradient Descent (SGD) algorithm, with a batch size of 150~256. Among the models, the initial learning rate of CNN and SW-ConvNet was set to 0.1; the learning rate was decayed by multiplying it by 0.1 every 30 epochs.

# Requirements
Python 3.12
timm 1.0.7
PyWavelets 1.6.0

The classification results of the original CNNs were used as the baseline, and these results were obtained from the official PyTorch 1.11+cuda11.3, platform  is ubuntu+python 3.8, the dependent software libraries include Kymatio 0.3 \cite{Kymatio}, Pywavelets 1.4.1\cite{Lee}.

# How to use
You can import scattering and use it in your CNN
```markdown
```python
from kymatio.torch import Scattering2D
scattering1 = ScatteringTorch2D(J=J, shape=(224, 224), L=L_ang, max_order=max_order,backend='torch')
scattering1 = nn.Sequential(scattering1, nn.Flatten(1, 2))
self.s1=scattering1
```
