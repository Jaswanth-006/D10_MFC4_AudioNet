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
import shutil
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import AudioCNN


LOCAL_SOURCE_PATH = Path(os.getenv("ESC50_PATH", "./Dataset/ESC-50-master"))
DATA_ROOT = Path("./data")
MODEL_DIR = Path("./models")
LOG_DIR = Path("./runs/ResNet_39_Experiment")
IMG_DIR = Path("./spectrogram_samples")

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def save_spectrogram_batch(spectrograms, labels, classes, epoch, batch_idx, prefix="train"):

    IMG_DIR.mkdir(exist_ok=True)

    num_imgs = min(len(spectrograms), 8)

    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 4))

    if num_imgs == 1:
        axes = [axes]

    for i in range(num_imgs):

        spec = spectrograms[i].squeeze().cpu().numpy()

        label_id = labels[i].item()

        label_name = classes[label_id]

        ax = axes[i]

        ax.imshow(spec, origin='lower', aspect='auto', cmap='viridis')

        ax.set_title(f"{label_name}\n({prefix})")

        ax.axis('off')

    plt.tight_layout()

    save_path = IMG_DIR / f"epoch_{epoch+1}_batch_{batch_idx}_{prefix}.png"

    plt.savefig(save_path)

    plt.close(fig)


def check_gpu():

    if torch.cuda.is_available():

        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

        return torch.device('cuda')

    else:

        print("GPU NOT detected")

        return torch.device('cpu')


def validate_and_fix_dataset(local_path, fallback_root):

    if local_path.exists():

        audio_files = list((local_path / "audio").glob("*.wav"))

        nested_audio = list((local_path / "ESC-50-master" / "audio").glob("*.wav"))

        if len(audio_files) == 2000:
            return local_path

        if len(nested_audio) == 2000:
            return local_path / "ESC-50-master"

    fallback_path = fallback_root / "ESC-50-master"

    if fallback_path.exists() and len(list((fallback_path / "audio").glob("*.wav"))) == 2000:

        return fallback_path

    print("Downloading ESC50 dataset")

    fallback_root.mkdir(exist_ok=True, parents=True)

    zip_path = fallback_root / "esc50.zip"

    urllib.request.urlretrieve(ESC50_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall(fallback_root)

    os.remove(zip_path)

    return fallback_root / "ESC-50-master"


class ESC50Dataset(Dataset):

    def __init__(self, data_dir, metadata_file, split="train", transform=None):

        self.data_dir = Path(data_dir)

        self.metadata = pd.read_csv(metadata_file)

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

        if not audio_path.exists():

            raise FileNotFoundError(f"Missing file: {audio_path}")

        # FIX: Use torchaudio instead of soundfile
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:

            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:

            spectrogram = self.transform(waveform)

        else:

            spectrogram = waveform

        return spectrogram, row['label']


def mixup_data(x, y):

    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)

    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train():

    MODEL_DIR.mkdir(exist_ok=True)

    LOG_DIR.mkdir(exist_ok=True, parents=True)

    device = check_gpu()

    dataset_path = validate_and_fix_dataset(LOCAL_SOURCE_PATH, DATA_ROOT)

    meta_path = dataset_path / "meta" / "esc50.csv"


    train_transform = nn.Sequential(

        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128),

        T.AmplitudeToDB(),

        T.FrequencyMasking(30),

        T.TimeMasking(80)

    )


    val_transform = nn.Sequential(

        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128),

        T.AmplitudeToDB()

    )


    train_dataset = ESC50Dataset(dataset_path, meta_path, "train", train_transform)

    val_dataset = ESC50Dataset(dataset_path, meta_path, "test", val_transform)


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    model = AudioCNN(num_classes=len(train_dataset.classes)).to(device)


    num_epochs = 100

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(optimizer, max_lr=0.002, epochs=num_epochs, steps_per_epoch=len(train_dataloader))


    writer = SummaryWriter(log_dir=str(LOG_DIR))

    best_accuracy = 0.0


    print("Starting training...")


    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')


        for batch_idx, (data, target) in enumerate(progress_bar):

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


        # Validation

        model.eval()

        correct = 0

        total = 0


        with torch.no_grad():

            for data, target in test_dataloader:

                data, target = data.to(device), target.to(device)

                outputs = model(data)

                _, predicted = torch.max(outputs.data, 1)

                total += target.size(0)

                correct += (predicted == target).sum().item()


        accuracy = 100 * correct / total


        
        # Calculate the average training loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        print(f'Epoch {epoch+1} Accuracy: {accuracy:.2f}% | Train Loss: {avg_epoch_loss:.4f}')

        # ---> ADD THESE LINES TO SEND DATA TO TENSORBOARD <---
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        # ------------------------------------------------------


        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

            print("New best model saved")


    writer.close()


if __name__ == "__main__":

    train()