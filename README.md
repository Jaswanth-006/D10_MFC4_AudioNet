# Environmental Sound Classification Using Residual Convolutional Neural Networks and Bidirectional Pseudo-Inverse Learning

**Team-10**
22MAT220 – Mathematics for Computing IV

---

## Member Details

| Name               | Roll No          | Email                                                       |
| ------------------ | ---------------- | ----------------------------------------------------------- |
| Aparna Bharani     | CB.SC.U4AIE24304 | [cb.sc.u4aie24304@cb.students.amrita.edu](mailto:cb.sc.u4aie24304@cb.students.amrita.edu)   |
| Jaswanth Saravanan | CB.SC.U4AIE24324 | [cb.sc.u4aie24324@cb.students.amrita.edu](mailto:cb.sc.u4aie24324@cb.students.amrita.edu)   |
| Parkavi R          | CB.SC.U4AIE24338 | [cb.sc.u4aie24338@cb.students.amrita.edu](mailto:cb.sc.u4aie24338@cb.students.amrita.edu)   |
| Rajashree T        | CB.SC.U4AIE24346 | [cb.sc.u4aie24346@cb.students.amrita.edu](mailto:cb.sc.u4aie24346@cb.students.amrita.edu)   |

---

## 1. Objective

To design and implement an end-to-end deep learning system that classifies environmental sounds (such as sirens, rain, engines, and animal sounds) from raw audio signals by combining signal processing techniques with a Residual Convolutional Neural Network (ResNet), and mathematically comparing it against a gradient-free Bidirectional Pseudo-Inverse Learning (BiPIL) scheme.

---

## 2. Motivation / Why the Project is Interesting

Environmental audio classification plays a key role in smart cities, surveillance systems, assistive technologies, and edge AI devices. Audio signals are non-stationary and unstructured, making them mathematically challenging to model. This project is interesting because it integrates Fourier analysis and linear algebra with modern deep learning techniques, adapts image-based CNN architectures for audio data, and contrasts iterative optimization (gradient descent) with analytical closed-form learning (Moore-Penrose pseudo-inverse).

---

## 3. Introduction

Environmental Sound Classification (ESC) focuses on identifying non-speech audio events occurring in everyday environments. Unlike speech recognition, environmental sounds are highly diverse, irregular, and often contaminated by background noise, making classification a challenging task.

Recent advances in deep learning have significantly improved audio recognition performance. Convolutional Neural Networks (CNNs), particularly residual architectures introduced in [1], enable deeper networks through shortcut connections that mitigate vanishing gradient issues. However, increasing model depth raises computational complexity and may lead to overfitting, especially for small-scale datasets.

Alternatively, gradient-free learning approaches such as BiPIL [3] eliminate backpropagation and iterative gradient updates. These methods reduce hyperparameter tuning complexity and offer architectural flexibility, making them promising for moderate-sized datasets.

---

## 4. Dataset Description

The experiments in this work utilize the ESC-50 dataset [2], a widely used benchmark dataset for environmental sound classification.

**Table 1: ESC-50 Dataset Characteristics**
| Property | Value |
| :--- | :--- |
| Total audio clips | 2000 |
| Number of classes | 50 |
| Samples per class | 40 |
| Audio duration | 5 seconds |
| Sampling rate | 44.1 kHz |
| Audio format | WAV |

**Table 2: ESC-50 Category Groups**
| Category Group | Examples |
| :--- | :--- |
| Animals | Dog bark, rooster, frog |
| Natural soundscapes | Rain, sea waves, wind |
| Human non-speech | Coughing, sneezing, clapping |
| Interior sounds | Door knock, washing machine |
| Urban noises | Car horn, siren, engine |

The ESC-50 dataset is balanced. However, due to its relatively small size compared to large-scale datasets, it poses challenges for training deep neural networks and is therefore well suited for evaluating both deep learning models and alternative learning strategies.

---

## 5. Methodology

### 5.1 System Overview
The proposed framework compares two fundamentally different learning paradigms:
1. **A gradient-based Convolutional Neural Network (CNN)**, scaled up to deep Residual Networks (ResNets).
2. **A gradient-free Bidirectional Pseudo-Inverse Learning (BiPIL) network**, which encompasses both forward and backward training processes using analytical matrix operations.

![Overview](./docs/images/Screenshot%202026-02-28%20193529.png)
*Figure 1: Overview of the audio classification pipeline.*

### 5.2 CNN and ResNet Architecture

#### 5.2.1 Signal Transformation (Time → Frequency Domain)
Raw audio signals $s(t)$ do not possess spatial structure suitable for convolutional neural networks. Therefore, Short-Time Fourier Transform (STFT) is applied.

$$X(k,m) = \sum_{n=0}^{N-1} x[n]w[n-m]e^{-j2\pi kn/N}$$

To better represent human auditory perception, the frequency axis is converted to the Mel scale:

$$Mel(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

![Example of a Mel Spectrogram](./docs/images/Screenshot%202026-03-06%20122524.png)
*Figure 2: Example of a Mel Spectrogram. The resulting representation is treated as a 2D image input.*

#### 5.2.2 Convolutional Feature Extraction
Instead of full matrix multiplication, a CNN slides learnable filters (kernels) across the input to extract localized acoustic patterns. 

$$Y(i,j) = \sum_m \sum_n X(i-m,j-n)K(m,n)$$

![CNN Feature Extraction](./docs/images/Screenshot%202026-02-28%20193444.png)
*Figure 3: Hierarchical feature extraction through multiple Conv2D layers.*

#### 5.2.3 Deep Residual Networks (ResNet) & The Vanishing Gradient
As networks become deeper, they encounter the degradation problem. In a plain CNN, the output after one block is $H(x) = Wx$. During backpropagation, the gradient becomes:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot W$$
If $W$ has eigenvalues less than 1, repeated multiplication across many layers causes gradients to shrink exponentially (vanishing gradient problem).

**Numerical Toy Example:**
Let Input $x = 1$, Weight $W = 0.1$.
* Plain CNN output after 5 layers: $0.1^5 = 0.00001$ (gradient nearly vanishes).
* Residual CNN output after 5 layers: $1 + 5 \times 0.1 \approx 1.5$ (signal preserved).

![Residual Block](./docs/images/Screenshot%202026-02-28%20193545.png)
*Figure 4: A Residual Block with the identity shortcut connection.*

In a Residual Block, the mapping is reformulated as $y = \mathcal{F}(x,\{W_i\}) + x$. Because the identity matrix $I$ is added, the gradient always has a direct path ($\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot (W + I)$), ensuring stable gradient flow.

#### 5.2.4 Classification
Following feature extraction, Global Average Pooling reduces spatial dimensions, followed by a fully connected layer with Softmax activation to classify input audio into one of 50 environmental sound categories.

### 5.3 Bidirectional Pseudo-Inverse (BiPIL)

#### 5.3.1 Network Structure
![Matrix dimensional flow](./docs/images/Screenshot%202026-03-11%20230542.png)
*Figure 5: Matrix dimensional flow of the BiPIL architecture.*

The BiPIL network processes all training samples simultaneously in matrix form. A typical implementation includes:
* Input: 2048-dimensional vector
* Hidden Layer 1: 25 neurons
* Hidden Layer 2: 50 neurons
* Output Layer: 50 neurons 

#### 5.3.2 Forward Phase and Backward Phase
Given an input $X$, hidden representations are mapped sequentially:
$$H_1 = \phi(W_1 X)$$
$$H_2 = \phi(W_2 H_1)$$
The optimal output weights $W_o$ are computed analytically using the Moore-Penrose inverse:
$$W_o = Y H_2^{+}$$

Label information is then propagated backward to analytically update the connection weights:
$$H_2^{b} = W_o^{+} Y$$
$$H_1^{b} = W_2^{+} \phi^{-1}(H_2^{b})$$
Forward weights are recomputed:
$$W_1 = H_1^{b} X^{+}$$

---

## 6. Computational Complexity Analysis

**Table 3: Computational Complexity Comparison**

| Method | Operation | Time Complexity |
| :--- | :--- | :--- |
| CNN Layer | Convolution operation | $O(N \cdot k^2 \cdot C_{in} \cdot C_{out})$ |
| BiPIL Training | Pseudo-inverse matrix computation | $O(n^3)$ |

Convolution operations scale with the feature map size and channel dimensions, while the BiPIL algorithm involves cubic complexity due to matrix inversion. Although BiPIL eliminates iterative backpropagation, the pseudo-inverse computation may become computationally expensive for large matrices.

---

## 7. Toy Example of BiPIL Learning

Consider a simplified dataset with an Input dimension of 2, 1 hidden layer with 2 neurons, and an Output dimension of 1.

Let the training input matrix and target output be:
$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad Y = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

Assume a random weight matrix $W_1$:
$$W_1 = \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.7 \end{bmatrix}$$

The hidden representation becomes $H = \phi(W_1 X)$. The output weight is computed analytically as:
$$W_o = Y H^{+}$$
Unlike gradient descent, no iterative updates are required. The solution is obtained in closed form using matrix operations.

---

## 8. Results & Discussion

The implemented ResNet successfully learned discriminative features from Mel spectrograms. Initial overfitting due to the small size of the ESC-50 dataset was mitigated using Mixup augmentation and label smoothing. Training on Modal serverless GPUs significantly reduced training time, and feature map visualizations demonstrated meaningful spectral pattern learning across layers. 

Meanwhile, BiPIL offered rapid analytical training that significantly reduced hyperparameter tuning time.

### 8.1 Effect of Hyperparameters on BiPIL Accuracy

**Table 4: Effect of Hidden Layer Architecture on Model Accuracy**
| Hidden Layer Architecture | Train Accuracy | Test Accuracy | Time took for training |
| :--- | :--- | :--- | :--- |
| (50, 25) | 60.00% | 30.00% | 4.00 secs |
| (25, 50) | 100.00% | 50.00% | 2.20 secs |
| (50, 100, 50) | 100.00% | 50.00% | 2.57 secs |
| (200, 500, 1000) | 100.00% | 50.00% | 4.48 secs |

**Table 5: Effect of Activation Functions and Train-Test Split**
* **Activation Functions:** Sigmoid, Tanh, and Cos all achieved 100% Train / 50.00% Test accuracy in ~2.5 seconds.
* **Train-Test Split:** A 50:50 split yielded 100% Train / 50.00% Test accuracy, outperforming 40:60, 70:30, and 80:20 splits.

### 8.2 Training and Validation Performance (ResNet)

![ResNet-26 Accuracy](./docs/images/resnet26_accuracy.jpeg)
*Figure 6: Validation accuracy curve for ResNet-26 during training.*

![ResNet-26 Loss](./docs/images/resnet26_loss.jpeg)
*Figure 7: Training loss curve for ResNet-26 showing convergence across epochs.*

![ResNet-34 Accuracy](./docs/images/resnet34_accuracy.jpeg)
*Figure 8: Validation accuracy curve for ResNet-34 during training.*

![ResNet-34 Loss](./docs/images/resnet34_loss.jpeg)
*Figure 9: Training loss curve for ResNet-34 showing gradual reduction across epochs.*

### 8.3 Final Model Comparison

**Table 6: Comparison of Different ResNet Architectures (GPU)**
| Model | Layers | Channels | Epochs | Acc (GPU) | Time (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-26 | 26 | [64,128,256,512] | 100 | 83.00% | 20.8 minutes |
| ResNet-34 | 34 | [64,128,256,512] | 100 | 84.25% | 20 minutes |
| ResNet-39 | 39 | [64,128,256,512] | 100 | 83.00% | 23 minutes |

**Table 7: Overall Model Performance Comparison**
| Model | Training Accuracy |
| :--- | :--- |
| ResNet-34 | 84.25% |
| ResNet-39 | 83.00% |
| ResNet-26 | 83.00% |
| BiPIL | 100.00% |

---

## 9. Future Plans

* Real-time environmental sound classification using streaming microphone input.
* Edge deployment through INT8 quantization for low-power devices such as Raspberry Pi.
* Performance comparison with Audio Spectrogram Transformers (AST).
* Deployment in smart city noise monitoring and emergency detection systems.

---

## 10. References

1. He et al., Deep Residual Learning for Image Recognition, 2016. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
2. Piczak, ESC-50: Dataset for Environmental Sound Classification, 2015. [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
3. Wang, K., et al. "Bi-PIL: Bidirectional Gradient-Free Learning Scheme for Multilayer Neural Networks." *IEEE Transactions*, 2025.
4. Hershey et al., CNN Architectures for Large-Scale Audio Classification, 2017. [https://arxiv.org/abs/1609.09430](https://arxiv.org/abs/1609.09430)
5. Librosa Documentation. [https://librosa.org/doc/latest/index.html](https://librosa.org/doc/latest/index.html)
6. PyTorch Audio. [https://pytorch.org/audio/stable/index.html](https://pytorch.org/audio/stable/index.html)

---

## Content and Folder Structure

```bash
├── code/
|   ├── audio-cnn-visualisation/
|       └── src/app/              
│   ├── main.py            
│   ├── model.py            
│   ├── requirements.txt
|   ├── train.py            
│
├── doc/                   
│   ├── MCF4_0th_review.pdf
│   ├── MFC4_1st_review.pdf
│   ├── base paper.pdf
│   ├── theory.excalidraw
│ 
└── README.md