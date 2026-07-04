import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import cv2
import json
import time
from glob import glob
import math

# Set matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# ============================================
# 🎯 CHANGED FOLDER NAME FOR RESULTS
# ============================================
results_folder = 'new_swinir_results'

# Create directories
os.makedirs(f'{results_folder}/error_maps', exist_ok=True)
os.makedirs(f'{results_folder}/comparisons', exist_ok=True)
os.makedirs(f'{results_folder}/metrics', exist_ok=True)
os.makedirs(f'{results_folder}/models', exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# 1. Dataset Loader
# ==========================
class M3FDDataset(Dataset):
    def __init__(self, image_dir, scale=4, crop_size=128):
        self.image_dir = image_dir
        self.scale = scale
        self.crop_size = crop_size
        self.image_files = sorted(
            glob(os.path.join(image_dir, "*.jpg")) +
            glob(os.path.join(image_dir, "*.png")) +
            glob(os.path.join(image_dir, "*.bmp"))
        )
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            img = img[top:top+self.crop_size, left:left+self.crop_size]

        hr = torch.FloatTensor(img / 255.0).unsqueeze(0)
        lr_img = cv2.resize(img, (self.crop_size // self.scale, self.crop_size // self.scale), interpolation=cv2.INTER_CUBIC)
        lr = torch.FloatTensor(lr_img / 255.0).unsqueeze(0)
        return lr, hr, os.path.basename(img_path)

# ==========================
# 2. SwinIR Model
# ==========================
class SwinIR(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, upscale=4):
        super(SwinIR, self).__init__()
        self.upscale = upscale
        self.embed_dim = embed_dim

        # Patch embedding
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[nn.Conv2d(embed_dim, embed_dim, 3, 1, 1) for _ in range(6)])  # Simplified for IR
        self.conv_before_upsample = nn.Conv2d(embed_dim, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale**2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blocks(x)
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x

# ==========================
# 3. Metrics & Visualization
# ==========================
def calculate_metrics(sr, hr):
    sr_np = sr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    psnr_val = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    win_size = min(7, min(hr_np.shape))
    if win_size % 2 == 0:
        win_size -= 1
    ssim_val = structural_similarity(hr_np, sr_np, data_range=1.0, win_size=win_size)
    return psnr_val, ssim_val

def create_error_map(hr, sr, title, filename):
    hr_np = hr.squeeze().cpu().numpy()
    sr_np = sr.squeeze().cpu().numpy()
    error = np.abs(hr_np - sr_np)
    plt.figure(figsize=(6,5))
    plt.imshow(error, cmap='hot')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_comparison_figure(lr, sr, hr, filename, sample_name, psnr, ssim):
    lr_np = lr.squeeze().cpu().numpy()
    sr_np = sr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    error = np.abs(hr_np - sr_np)
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes[0,0].imshow(lr_np, cmap='gray'); axes[0,0].set_title("LR Input"); axes[0,0].axis('off')
    axes[0,1].imshow(sr_np, cmap='gray'); axes[0,1].set_title(f"SR Output\nPSNR:{psnr:.2f}, SSIM:{ssim:.4f}"); axes[0,1].axis('off')
    axes[0,2].imshow(hr_np, cmap='gray'); axes[0,2].set_title("HR GT"); axes[0,2].axis('off')
    axes[1,0].imshow(error, cmap='hot'); axes[1,0].set_title("Error Map (Hot)"); axes[1,0].axis('off'); plt.colorbar(axes[1,0].images[0], ax=axes[1,0])
    axes[1,1].imshow(error, cmap='viridis'); axes[1,1].set_title("Error Map (Viridis)"); axes[1,1].axis('off'); plt.colorbar(axes[1,1].images[0], ax=axes[1,1])
    axes[1,2].imshow(error, cmap='plasma'); axes[1,2].set_title("Error Map (Plasma)"); axes[1,2].axis('off'); plt.colorbar(axes[1,2].images[0], ax=axes[1,2])
    plt.suptitle(f"SwinIR Results: {sample_name}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================
# 4. Training & Evaluation
# ==========================
def train_swinir(model, train_loader, val_loader, epochs=50, lr=1e-4):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    train_losses, val_psnr, val_ssim = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for lr_imgs, hr_imgs, _ in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lr_imgs, hr_imgs, img_names in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                for j in range(sr_imgs.shape[0]):
                    psnr_val, ssim_val = calculate_metrics(sr_imgs[j:j+1], hr_imgs[j:j+1])
                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
        val_psnr.append(np.mean(psnr_list))
        val_ssim.append(np.mean(ssim_list))
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} - Loss:{avg_loss:.4f} PSNR:{val_psnr[-1]:.2f} SSIM:{val_ssim[-1]:.4f}")

        if epoch % 10 == 0:
            # ============================================
            # 🎯 CHANGED RESULTS FOLDER PATH
            # ============================================
            torch.save(model.state_dict(), f"{results_folder}/models/swinir_epoch_{epoch}.pth")

    # ============================================
    # 🎯 CHANGED RESULTS FOLDER PATH
    # ============================================
    torch.save(model.state_dict(), f"{results_folder}/models/swinir_final.pth")
    return model

def evaluate_swinir(model, test_loader):
    model.eval()
    metrics_list = {}
    with torch.no_grad():
        for lr_imgs, hr_imgs, img_names in test_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            for j in range(sr_imgs.shape[0]):
                psnr_val, ssim_val = calculate_metrics(sr_imgs[j:j+1], hr_imgs[j:j+1])
                metrics_list[img_names[j]] = {"PSNR": psnr_val, "SSIM": ssim_val}
                # ============================================
                # 🎯 CHANGED RESULTS FOLDER PATHS
                # ============================================
                create_error_map(hr_imgs[j:j+1], sr_imgs[j:j+1],
                                 title=f"{img_names[j]} Error Map",
                                 filename=f"{results_folder}/error_maps/{img_names[j]}_error.png")
                create_comparison_figure(lr_imgs[j:j+1], sr_imgs[j:j+1], hr_imgs[j:j+1],
                                         filename=f"{results_folder}/comparisons/{img_names[j]}_compare.png",
                                         sample_name=img_names[j],
                                         psnr=psnr_val, ssim=ssim_val)
    # ============================================
    # 🎯 CHANGED RESULTS FOLDER PATH
    # ============================================
    with open(f"{results_folder}/metrics/metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=4)
    print("Evaluation finished. Metrics saved.")

# ==========================
# 5. Run Everything
# ==========================
# ============================================
# 🎯 CHANGED DATASET PATH FROM "Ir" TO "Irn"
# ============================================
dataset_path = "Irn"  # <- Changed from "Ir" to "Irn"
dataset = M3FDDataset(dataset_path, crop_size=128, scale=4)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

model = SwinIR().to(device)

start_time = time.time()
trained_model = train_swinir(model, train_loader, val_loader, epochs=100)
print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes")

evaluate_swinir(trained_model, test_loader)