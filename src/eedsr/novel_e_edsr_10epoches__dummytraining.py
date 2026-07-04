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

# ============================================
# 1. Dataset Loader (Update with your dataset paths)
# ============================================
class IRDataset(Dataset):
    def __init__(self, lr_images, hr_images, transform=None):
        self.lr_images = lr_images
        self.hr_images = hr_images
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = self.lr_images[idx]
        hr = self.hr_images[idx]

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr


# ============================================
# 2. EEDSR+ Model
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
    def __init__(self, scale=2, n_resblocks=16, n_feats=64):
        super(EEDSR, self).__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)

        self.resblocks = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.conv_mid = nn.Conv2d(n_feats, n_feats, 3, padding=1)

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


def train_model(train_loader, test_loader, scale=2, epochs=10):
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
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for lr, hr in test_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                
                # Calculate metrics for each image in the batch
                for i in range(sr.shape[0]):
                    psnr_val, ssim_val = calculate_metrics(sr[i:i+1], hr[i:i+1])
                    psnr_vals.append(psnr_val)
                    ssim_vals.append(ssim_val)
                
                # Show results for first image in first batch only
                if len(psnr_vals) == 1:
                    show_results(lr[0], sr[0], hr[0], epoch)

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


# ============================================
# 5. Run Training (Example)
# ============================================
if __name__ == "__main__":
    # TODO: Replace these with your real IR dataset tensors
    # Example: use torch tensors (N,1,H,W) normalized to [0,1]
    dummy_lr = [torch.rand(1, 64, 64) for _ in range(20)]
    dummy_hr = [F.interpolate(x.unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False).squeeze(0) for x in dummy_lr]

    transform = T.Compose([T.Normalize(0.5, 0.5)])  # update as needed

    train_ds = IRDataset(dummy_lr[:15], dummy_hr[:15], transform=None)
    test_ds = IRDataset(dummy_lr[15:], dummy_hr[15:], transform=None)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    train_model(train_loader, test_loader, scale=2, epochs=10)