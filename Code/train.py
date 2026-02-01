import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
import soundfile as sf
import shutil
import urllib.request
import zipfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import your model
from model import AudioCNN

# --- Configuration ---
# 1. We will try to use your local path first.
#    If it's broken, we will fix it in the 'data' folder.
LOCAL_SOURCE_PATH = Path(r"C:\Users\parka\OneDrive\Desktop(1)\Environmental-Audio-Project\D10_MFC4_AudioNet\ESC-50")
DATA_ROOT = Path("./data") # Fallback folder for downloads
MODEL_DIR = Path("./models")
LOG_DIR = Path("./runs/v2_experiment_spectrogram") 
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("⚠️ GPU NOT detected. Training will be slow on CPU.")
        return torch.device('cpu')

def validate_and_fix_dataset(local_path, fallback_root):
    """
    Checks if the dataset at local_path is valid (2000 files).
    If invalid, it downloads a fresh copy to fallback_root.
    """
    # 1. Check Local Path
    if local_path.exists():
        audio_files = list((local_path / "audio").glob("*.wav"))
        nested_audio = list((local_path / "ESC-50-master" / "audio").glob("*.wav"))
        
        # Check main folder
        if len(audio_files) == 2000:
            return local_path
        # Check nested folder
        if len(nested_audio) == 2000:
            return local_path / "ESC-50-master"
            
        print(f"⚠️ Local dataset corrupted: Found {len(audio_files) + len(nested_audio)} files (Expected 2000).")
    
    # 2. If we reach here, local is missing or broken. Check Fallback.
    fallback_path = fallback_root / "ESC-50-master"
    if fallback_path.exists():
        count = len(list((fallback_path / "audio").glob("*.wav")))
        if count == 2000:
            print(f"✅ Found valid cached dataset at {fallback_path}")
            return fallback_path
        else:
             print("⚠️ Cached dataset also corrupted. Re-downloading...")
             shutil.rmtree(fallback_root, ignore_errors=True)

    # 3. Download and Extract
    print(f"⬇️ Downloading fresh ESC-50 dataset to {fallback_root}...")
    fallback_root.mkdir(exist_ok=True, parents=True)
    zip_path = fallback_root / "esc50.zip"
    
    # Progress bar hook
    def progress_hook(t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None: t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return update_to

    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=progress_hook(t))
            
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fallback_root)
        os.remove(zip_path)
        
        # Verify again
        new_path = fallback_root / "ESC-50-master"
        count = len(list((new_path / "audio").glob("*.wav")))
        if count == 2000:
            print("✅ Download & Extraction successful.")
            return new_path
        else:
            raise RuntimeError(f"Download failed. Extracted {count} files.")
            
    except Exception as e:
        print(f"❌ Error during download/extraction: {e}")
        return None

# --- Dataset Class ---
class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']
        
        # Robust read
        if not audio_path.exists():
             raise FileNotFoundError(f"❌ Missing file: {audio_path}")

        # Use soundfile (Fix for Windows/Torchcodec)
        audio_np, sample_rate = sf.read(str(audio_path), dtype='float32')
        waveform = torch.from_numpy(audio_np).float()
        
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']

def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train():
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    device = check_gpu()

    # 1. Validation & Repair Step
    dataset_path = validate_and_fix_dataset(LOCAL_SOURCE_PATH, DATA_ROOT)
    
    if dataset_path is None:
        print("❌ CRITICAL: Could not prepare dataset.")
        return

    meta_path = dataset_path / "meta" / "esc50.csv"
    print(f"📂 Using verified dataset at: {dataset_path}")

    # 2. Transforms
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )
    val_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB()
    )

    # 3. Data Loaders
    train_dataset = ESC50Dataset(data_dir=dataset_path, metadata_file=meta_path, split="train", transform=train_transform)
    val_dataset = ESC50Dataset(data_dir=dataset_path, metadata_file=meta_path, split="test", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Set pin_memory=True for GPU training
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Model Setup
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.002, epochs=num_epochs, steps_per_epoch=len(train_dataloader), pct_start=0.1)
    
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    best_accuracy = 0.0
    min_loss = float('inf')
    writer.add_text('Config', 'Model: ResNet-18 (2D CNN) | Input: Mel-Spectrogram | Latent Dim: 512')

    # 5. Training Loop
    print(f"Starting training on {device}...")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if avg_epoch_loss < min_loss: min_loss = avg_epoch_loss

        model.eval()
        correct = 0; total = 0; val_loss = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)
        epoch_duration = time.time() - epoch_start_time

        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('Performance/Time_Per_Epoch', epoch_duration, epoch)
        
        print(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%, Time: {epoch_duration:.2f}s')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = MODEL_DIR / 'best_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, save_path)
            print(f'New best model saved to {save_path}')

    total_duration = time.time() - total_start_time
    writer.add_text('Results', f'Best Acc: {best_accuracy:.2f}% | Min Loss: {min_loss:.4f} | Total Time: {total_duration:.2f}s')
    writer.close()
    
    print("--------------------------------------------------")
    print(f"Training Completed on {device}.")
    print(f"Total Time: {total_duration:.2f} seconds")
    print(f"Minimum Loss Achieved: {min_loss:.4f}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print("--------------------------------------------------")

if __name__ == "__main__":
    train()