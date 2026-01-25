# Environmental Audio Classification via Residual CNN
### 22MAT220 Mathematics for Computing 4 - Team 10

This project implements an end-to-end Deep Learning system to classify environmental sounds (e.g., sirens, rain, engines) from raw audio waveforms. It bridges signal processing theory with practical AI engineering by building a **Residual Convolutional Neural Network (ResNet)** from scratch in PyTorch, deployed on serverless GPU infrastructure.

## 👥 Team Members
* **Aparna Bharani** [CB.SC.U4AIE24304]
* **Jaswanth Saravanan** [CB.SC.U4AIE24324]
* **Parkavi R** [CB.SC.U4AIE24338]
* **Rajashree T** [CB.SC.U4AIE24346]

## 📄 Base & Reference Papers
The architectural design and feature extraction methodology are based on the following core research:
1.  **"CNN Architectures for Large-Scale Audio Classification"** (Hershey et al., Google, 2017)
    * *Relevance:* Establishes the effectiveness of adapting image-based CNN architectures (like ResNet) for audio classification using Mel-spectrograms.
2.  **"Deep Residual Learning for Image Recognition"** (He et al., 2016)
    * *Relevance:* Provides the mathematical foundation for skip connections ($y = F(x) + x$) to solve the vanishing gradient problem in deep networks.
3.  **ESC-50: Dataset for Environmental Sound Classification** (Piczak, 2015)
    * *Relevance:* The benchmark dataset used for training and validation.

## 📝 Project Outline
The project follows a mathematically grounded pipeline to transform unstructured audio signals into structured classifications:

1.  **Signal Transformation (Time $\to$ Frequency Domain):**
    * Raw audio signals $s(t)$ are non-stationary and lack spatial structure.
    * We apply Short-Time Fourier Transform (STFT) and map the result to the **Mel Scale** to mimic human auditory perception.
    * **Output:** A 2D Mel Spectrogram (Time $\times$ Frequency) that serves as the "image" input for the network.

2.  **Feature Extraction (Residual CNN):**
    * A custom **ResNet** architecture extracts hierarchical features.
    * **Convolutional Layers** detect local patterns (edges/textures in the spectrogram).
    * **Residual Blocks** allow the network to learn identity mappings, ensuring signal stability and gradient flow across deep layers.

3.  **Classification & Inference:**
    * Global Average Pooling reduces spatial dimensions.
    * A linear classifier maps features to 50 discrete probabilities (Softmax).
    * The model is deployed via **Modal** (serverless) and visualized using a **Next.js** frontend.

## 🔄 Updates (Current Status)
* **✅ Architecture Complete:** Successfully implemented a custom ResNet with 4 residual layers in PyTorch.
* **✅ Pipeline Integrated:** Automated the Raw Audio $\to$ Mel Spectrogram conversion within the training loop.
* **✅ Cloud Training:** Migrated training workload to **Modal** (Serverless GPUs), reducing training time significantly.
* **✅ Visualization UI:** Developed a Next.js dashboard that visualizes internal feature maps (activation heatmaps) to interpret *what* the model is learning.

## ⚠️ Challenges & Issues Faced
1.  **Overfitting on Small Data:**
    * *Issue:* The ESC-50 dataset is relatively small (2,000 samples), leading the model to memorize noise rather than features.
    * *Solution:* Implemented **Mixup Augmentation** (blending two audio clips linearly) and **Label Smoothing** to force the model to generalize better.
2.  **Vanishing Gradients:**
    * *Issue:* Initial plain CNN architectures struggled to converge as depth increased.
    * *Solution:* Introduced **Skip Connections (Residual Blocks)** to allow gradients to flow through the network unimpeded.
3.  **Spectrogram Resolution Trade-offs:**
    * *Issue:* Balancing time vs. frequency resolution in the STFT parameters.
    * *Solution:* Tuned hop length and window size to optimize for environmental sounds (which often have distinct spectral footprints).

## 🔮 Future Plans
1.  **Real-Time Inference:** Optimize the pipeline to classify streaming audio from a microphone in real-time.
2.  **Edge Deployment:** Quantize the model (Int8) to run on low-power edge devices (e.g., Raspberry Pi) for field deployment.
3.  **Transformer Architecture:** Compare the current ResNet performance against an Audio Spectrogram Transformer (AST) to analyze the efficiency of attention mechanisms vs. convolution.

---

## 🌟 Features
* **Custom CNN Architecture**: A ResNet-inspired convolutional neural network built from scratch in PyTorch.
* **Audio Processing**: Converts raw audio (WAV) to Mel Spectrograms for image-based classification.
* **Advanced Training Techniques**: Implements Mixup data augmentation, Label Smoothing, and One Cycle Learning Rate scheduling.
* **Serverless Infrastructure**: Uses [Modal](https://modal.com/) to train on high-end GPUs (H100/A10G) and host the inference API without local hardware.
* **Interactive Frontend**: A Next.js dashboard to upload audio files, view predictions, and explore:
    * Input Mel Spectrograms.
    * Raw Audio Waveforms.
    * Layer-by-layer feature map visualizations (Heatmaps).

## 🛠️ Tech Stack
### Machine Learning & Backend
* **Python 3.12**
* **PyTorch**: Deep learning framework.
* **Torchaudio**: Audio transformations (Spectrograms).
* **Librosa**: Audio loading and resampling.
* **Pandas & NumPy**: Data handling.
* **Modal**: Serverless cloud platform for training and API deployment.
* **FastAPI**: Used internally by Modal for the web endpoint.

### Frontend
* **Next.js (App Router)**: React framework.
* **TypeScript**: Type safety.
* **Tailwind CSS**: Styling.
* **shadcn/ui**: UI components.

## 📂 Project Structure
```bash
├── model.py            # PyTorch model architecture (ResNet blocks, forward pass)
├── train.py            # Training script (runs on Modal serverless GPU)
├── main.py             # Inference API endpoint (Modal web function)
├── requirements.txt    # Python dependencies
├── README.md           # Project Documentation
└── audio-cnn-viz/      # Next.js Frontend application
    ├── src/app/        # Page logic and UI components
    └── ...
