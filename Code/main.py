import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# Import your model
from model import AudioCNN

# Global variables for model storage
MODEL_PATH = Path("./models/best_model.pth")
ml_models = {}

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {MODEL_PATH} to {device}...")
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Please run train.py first.")
        yield
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    
    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    ml_models["model"] = model
    ml_models["classes"] = classes
    ml_models["processor"] = AudioProcessor()
    ml_models["device"] = device
    
    print("Model loaded successfully!")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Setup CORS to allow Next.js (usually localhost:3000) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    audio_data: str

@app.post("/inference")
async def inference(request: InferenceRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    model = ml_models["model"]
    classes = ml_models["classes"]
    processor = ml_models["processor"]
    device = ml_models["device"]

    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Preprocessing
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=44100)

        spectrogram = processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(device)

        # Inference
        with torch.no_grad():
            output, feature_maps = model(spectrogram, return_feature_maps=True)
            
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

            predictions = [
                {"class": classes[idx.item()], "confidence": prob.item()}
                for prob, idx in zip(top3_probs, top3_indicies)
            ]

            # Process visualizations
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            # Waveform decimation for visualization
            max_samples = 8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

            return {
                "predictions": predictions,
                "visualization": viz_data,
                "input_spectrogram": {
                    "shape": list(clean_spectrogram.shape),
                    "values": clean_spectrogram.tolist()
                },
                "waveform": {
                    "values": waveform_data.tolist(),
                    "sample_rate": waveform_sample_rate,
                    "duration": len(audio_data) / waveform_sample_rate
                }
            }

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)