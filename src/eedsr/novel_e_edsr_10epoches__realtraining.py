import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import os
import cv2
from glob import glob

# Set matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Clear GPU memory
torch.cuda.empty_cache()

# ============================================
# 1. Dataset Loader for M3FD Infrared Images
# ============================================
class M3FDDataset(Dataset):
    def __init__(self, image_dir, scale=4, crop_size=512):
        self.image_dir = image_dir
        self.scale = scale
        self.crop_size = crop_size
        
        # Get list of image files
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")) + 
                                glob(os.path.join(image_dir, "*.png")) +
                                glob(os.path.join(image_dir, "*.bmp")))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Center crop or resize
        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            img = img[top:top+self.crop_size, left:left+self.crop_size]
        
        # Convert to tensor and normalize to [0, 1]
        hr = torch.FloatTensor(img / 255.0).unsqueeze(0)
        
        # Create LR image by downscaling (without upscaling back)
        lr_img = cv2.resize(img, (self.crop_size // self.scale, self.crop_size // self.scale), 
                           interpolation=cv2.INTER_CUBIC)
        lr = torch.FloatTensor(lr_img / 255.0).unsqueeze(0)
        
        return lr, hr

# ============================================
# 2. EEDSR+ Model (Fixed Output Size)
# ============================================
class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale


class EEDSR(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=64):
        super(EEDSR, self).__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)

        self.resblocks = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.conv_mid = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.resblocks(x)
        res = self.conv_mid(res)
        x = x + res
        x = self.upsample(x)
        return x


# ============================================
# 3. Loss Function (Edge + Perceptual + L1)
# ============================================
class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.2, lambda3=0.01):
        super(EdgeAwareLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3

        # Fixed VGG19 loading for older torchvision versions
        try:
            # Try new weights API (torchvision >= 0.13)
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:8].eval()
        except AttributeError:
            # Fallback to old API (torchvision < 0.13)
            vgg = models.vgg19(pretrained=True).features[:8].eval()
            
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def edge_loss(self, sr, hr):
        def sobel(x):
            gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            grad_x = F.conv2d(x, gx, padding=1)
            grad_y = F.conv2d(x, gy, padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return F.l1_loss(sobel(sr), sobel(hr))

    def perceptual_loss(self, sr, hr):
        # Repeat single channel to 3 channels for VGG
        sr_3ch = sr.repeat(1, 3, 1, 1)
        hr_3ch = hr.repeat(1, 3, 1, 1)
        
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
        
        sr_3ch = (sr_3ch - mean) / std
        hr_3ch = (hr_3ch - mean) / std
        
        sr_vgg = self.vgg(sr_3ch)
        hr_vgg = self.vgg(hr_3ch)
        return F.l1_loss(sr_vgg, hr_vgg)

    def forward(self, sr, hr):
        l1 = self.l1(sr, hr)
        edge = self.edge_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)
        return self.lambda1*l1 + self.lambda2*edge + self.lambda3*perceptual


# ============================================
# 4. Training + Evaluation
# ============================================
def calculate_metrics(sr, hr):
    sr_np = sr.squeeze().cpu().detach().numpy()
    hr_np = hr.squeeze().cpu().detach().numpy()
    psnr_val = psnr_metric(hr_np, sr_np, data_range=1.0)
    
    # Dynamically adjust window size for SSIM based on image size
    min_dim = min(hr_np.shape)
    win_size = min(7, min_dim)
    if win_size % 2 == 0:  # Ensure window size is odd
        win_size -= 1
    win_size = max(win_size, 3)  # Ensure at least 3
    
    ssim_val = ssim_metric(hr_np, sr_np, data_range=1.0, win_size=win_size)
    return psnr_val, ssim_val

def show_results(lr, sr, hr, epoch):
    lr_np = lr.squeeze().cpu().numpy()
    sr_np = sr.squeeze().cpu().detach().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    error_map = np.abs(hr_np - sr_np)

    fig, axs = plt.subplots(1,4, figsize=(16,4))
    axs[0].imshow(lr_np, cmap='gray'); axs[0].set_title("LR")
    axs[1].imshow(sr_np, cmap='gray'); axs[1].set_title("SR (Ours)")
    axs[2].imshow(hr_np, cmap='gray'); axs[2].set_title("HR GT")
    im = axs[3].imshow(error_map, cmap='jet'); axs[3].set_title("Error Map")
    fig.colorbar(im, ax=axs[3])
    plt.suptitle(f"Epoch {epoch} Results")
    plt.savefig(f"results_epoch_{epoch}.png")
    plt.close()


def train_model(train_loader, test_loader, scale=4, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = EEDSR(scale=scale).to(device)
    criterion = EdgeAwareLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Training history
    train_losses = []
    val_psnr = []
    val_ssim = []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            sr = model(lr)
            
            # Ensure output matches target size
            if sr.shape != hr.shape:
                # Resize output to match target
                sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
            
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Print before deleting
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {current_loss:.4f}")
            
            # Clear memory after each batch
            del sr, loss
            if i % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for i, (lr, hr) in enumerate(test_loader):
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                
                # Ensure output matches target size
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
                
                # Calculate metrics for each image in the batch
                for j in range(sr.shape[0]):
                    psnr_val, ssim_val = calculate_metrics(sr[j:j+1], hr[j:j+1])
                    psnr_vals.append(psnr_val)
                    ssim_vals.append(ssim_val)
                
                # Show results for first image in first batch only
                if i == 0 and epoch % 2 == 0:  # Show every 2 epochs
                    show_results(lr[0], sr[0], hr[0], epoch)
                    
                # Clear memory
                del sr
                if i % 5 == 0:
                    torch.cuda.empty_cache()

        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        val_psnr.append(avg_psnr)
        val_ssim.append(avg_ssim)
        
        scheduler.step()

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
        
        # Save model checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"EEDSR_epoch_{epoch}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "EEDSR_final.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnr, label="PSNR")
    plt.plot(val_ssim, label="SSIM")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()
    
    return model, train_losses, val_psnr, val_ssim


# ============================================
# 5. Run Training with M3FD Dataset
# ============================================
if __name__ == "__main__":
    # Set your M3FD dataset path here
    dataset_path = "Irn"  # Update this to your M3FD infrared images path
    
    # Create dataset
    dataset = M3FDDataset(dataset_path, scale=4, crop_size=512)
    
    # Split into train and test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Train for 10 epochs
    model, train_losses, val_psnr, val_ssim = train_model(
        train_loader, test_loader, scale=4, epochs=10
    )
    
    print("Training completed!")
    print(f"Final PSNR: {val_psnr[-1]:.4f}, Final SSIM: {val_ssim[-1]:.4f}")