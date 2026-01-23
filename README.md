

# Audio Classification via Residual CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) from scratch to classify audio files using **PyTorch**. It includes a full-stack application with a **Next.js** frontend to upload audio, run inference, and visualize the internal feature maps of the CNN layers to understand how the model "sees" sound.

The project leverages **Modal** for serverless GPU training and deployment.

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
* **shadcn/ui**: UI components (Buttons, Cards, Progress bars).

## 📂 Project Structure

```
├── model.py            # PyTorch model architecture (ResNet blocks, forward pass)
├── train.py            # Training script (runs on Modal serverless GPU)
├── main.py             # Inference API endpoint (Modal web function)
├── requirements.txt    # Python dependencies
├── README.md
└── audio-cnn-viz/      # Next.js Frontend application
    ├── src/app/        # Page logic and UI components
    └── ...

```

## 🚀 Getting Started

### Prerequisites

* **Python 3.11+** installed locally.
* **Node.js & npm** installed.
* A **Modal** account (for running code in the cloud).

### 1. Backend Setup

1. **Clone the repository** (or create the files as per the tutorial).
2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Modal**:
```bash
pip install modal

```


4. **Setup Modal**:
```bash
modal setup

```


Follow the authentication steps in your browser to link your terminal to your Modal account.

### 2. Training the Model

The training script automatically downloads the **ESC-50 dataset**, processes it, and trains the model on a cloud GPU.

1. **Run the training script**:
```bash
modal run train.py

```


* This will provision a container, download the dataset, and start training for 100 epochs.
* You can monitor progress via the terminal progress bar or the Modal dashboard.
* **TensorBoard** logs are saved to a Modal Volume (`ESC-50-data`) and can be downloaded for analysis.



### 3. Deploying the Inference API

Once training is complete, the best model is saved to the shared volume. You can now deploy the API that the frontend will consume.

1. **Deploy the endpoint**:
```bash
modal deploy main.py

```


2. **Copy the URL**: Modal will output a web endpoint URL (e.g., `https://your-username--audio-cnn-inference.modal.run`). You will need this for the frontend.

### 4. Frontend Setup

1. **Navigate to the frontend directory** (create a Next.js app if you haven't):
```bash
npx create-next-app@latest audio-cnn-viz
# Select: TypeScript, Tailwind, ESLint, App Router
cd audio-cnn-viz

```


2. **Install dependencies**:
```bash
npm install
# Install shadcn/ui components used (button, card, progress, etc.)
npx shadcn-ui@latest init
npx shadcn-ui@latest add button card progress badge

```


3. **Update API URL**:
Open `src/app/page.tsx` and find the `fetch` call. Replace the URL with the **Modal deployment URL** you obtained in step 3.
4. **Run the development server**:
```bash
npm run dev

```


5. **Open the App**: Visit `http://localhost:3000` in your browser.

## 📊 How to Use

1. **Open the Web Interface**.
2. **Upload an Audio File**: Select a `.wav` file (e.g., from the ESC-50 dataset like a dog bark, clapping, or thunderstorm).
3. **View Results**:
* **Predictions**: See the top 3 classified categories with confidence scores.
* **Visualizations**: Scroll down to see the raw waveform, the input Mel Spectrogram, and the activation heatmaps for every convolutional layer in the network.



## 🧠 Model Details

The model architecture is defined in `model.py` and consists of:

* **Initial Convolution**: 7x7 kernel to capture broad features.
* **Residual Blocks**: 4 Layers of stacked residual blocks (ResNet style) with 3x3 convolutions, Batch Normalization, and ReLU activation.
* **Shortcuts**: Skip connections to prevent vanishing gradients.
* **Global Average Pooling & Linear Head**: For final classification into 50 classes.

## 📚 Dataset

This project uses the **ESC-50** (Environmental Sound Classification) dataset, which contains 2,000 labeled audio recordings (5 seconds each) organized into 50 semantic classes.

