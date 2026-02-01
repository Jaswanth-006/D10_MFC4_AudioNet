# D10_MFC4_Environmental Audio Classification via Residual CNN

## Project Title

**Environmental Audio Classification via Residual Convolutional Neural Networks**
22MAT220 – Mathematics for Computing IV

---

## Member Details

| Name               | Roll No          | Email                                                                                       |
| ------------------ | ---------------- | ------------------------------------------------------------------------------------------- |
| Aparna Bharani     | CB.SC.U4AIE24304 | [cb.sc.u4aie24304@cb.students.amrita.edu](mailto:cb.sc.u4aie24304@cb.students.amrita.edu)   |
| Jaswanth Saravanan | CB.SC.U4AIE24324 | [cb.sc.u4aie24324@cb.students.amrita.edu](mailto:cb.sc.u4aie24324@cb.students.amrita.edu)   |
| Parkavi R          | CB.SC.U4AIE24338 | [cb.sc.u4aie24338@cb.students.amrita.edu](mailto:cb.sc.u4aie24338@cb.students.amrita.edu)   |
| Rajashree T        | CB.SC.U4AIE24346 | [cb.sc.u4aie24346@cb.students.amrita.edu](mailto:cb.sc.u4aie24346@cb.students.amrita.edu)   |

---

## Objective

To design and implement an end-to-end deep learning system that classifies environmental sounds (such as sirens, rain, engines, and animal sounds) from raw audio signals by combining signal processing techniques with a Residual Convolutional Neural Network (ResNet).

---

## Motivation / Why the Project is Interesting

Environmental audio classification plays a key role in smart cities, surveillance systems, assistive technologies, and edge AI devices. Audio signals are non-stationary and unstructured, making them mathematically challenging to model. This project is interesting because it integrates Fourier analysis and linear algebra with modern deep learning techniques, adapts image-based CNN architectures for audio data, and demonstrates how residual connections improve deep network training.

---

## Methodology

### 1. Signal Transformation (Time → Frequency Domain)

Raw audio signals s(t) do not possess spatial structure suitable for convolutional neural networks. Therefore, Short-Time Fourier Transform (STFT) is applied followed by Mel-scale filtering to approximate human auditory perception.

                                             s(t) → STFT → Mel Spectrogram (Time × Frequency)

The resulting Mel spectrogram is treated as a 2D image input to the neural network.

### 2. Feature Extraction using Residual CNN

A custom ResNet architecture is implemented using PyTorch. Convolutional layers extract local spectral patterns, while residual (skip) connections ensure stable gradient flow.

Residual block formulation:

                           y = F(x) + x

where x is the input and F(x) is the residual mapping.

#### Toy Example (Mathematical Demonstration)

Consider a simple deep network composed of multiple layers trying to learn a mapping H(x).

In a plain CNN, the output after one block is:

                                             H(x) = Wx

During backpropagation, the gradient becomes:

                                            ∂L/∂x = ∂L/∂H · W

If W has eigenvalues less than 1, repeated multiplication across many layers causes gradients to shrink exponentially, leading to the vanishing gradient problem.

In a Residual Block, the mapping is reformulated as:

                                                   H(x) = F(x) + x

where F(x) = Wx.

Now the gradient becomes:

                        ∂L/∂x = ∂L/∂H · (W + I)

Because the identity matrix I is added, the gradient always has a direct path, ensuring stable gradient flow.

##### Numerical Toy Example

Let:

* Input x = 1
* Weight W = 0.1

Plain CNN output after one layer:
H(x) = 0.1

After 5 layers:
0.1⁵ = 0.00001 (gradient nearly vanishes)

Residual CNN output after one block:
H(x) = 0.1 + 1 = 1.1

After 5 layers:
1 + 5 × 0.1 ≈ 1.5 (signal preserved)

This demonstrates mathematically how residual connections prevent vanishing gradients and enable deeper networks to train effectively.

### 3. Classification

Global Average Pooling reduces spatial dimensions, followed by a fully connected layer with Softmax activation to classify input audio into one of 50 environmental sound categories.

---

## Results & Discussion

The implemented ResNet successfully learned discriminative features from Mel spectrograms. Initial overfitting due to the small size of the ESC-50 dataset was mitigated using Mixup augmentation and label smoothing. Training on Modal serverless GPUs significantly reduced training time, and feature map visualizations demonstrated meaningful spectral pattern learning across layers.

---

## Future Plans

* Real-time environmental sound classification using streaming microphone input.
* Edge deployment through INT8 quantization for low-power devices such as Raspberry Pi.
* Performance comparison with Audio Spectrogram Transformers (AST).
* Deployment in smart city noise monitoring and emergency detection systems.

---

## References

1. He et al., Deep Residual Learning for Image Recognition, 2016
   [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

2. Hershey et al., CNN Architectures for Large-Scale Audio Classification, 2017
   [https://arxiv.org/abs/1609.09430](https://arxiv.org/abs/1609.09430)

3. Piczak, ESC-50: Dataset for Environmental Sound Classification, 2015
   [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

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
```
