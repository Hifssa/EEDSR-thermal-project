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
import json
import time
from datetime import datetime

# Set matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Clear GPU memory
torch.cuda.empty_cache()

# Create directories for saving results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/error_maps', exist_ok=True)
os.makedirs('results/comparisons', exist_ok=True)
os.makedirs('results/graphs', exist_ok=True)

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
        
        return lr, hr, img_path

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
# 3. Baseline Models for Comparison
# ============================================
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class EDSR(nn.Module):
    def __init__(self, num_blocks=8, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.final_conv = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x += residual
        x = self.final_conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

# ============================================
# 4. Loss Function (Edge + Perceptual + L1)
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
# 5. Training + Evaluation Functions
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

def create_error_map(hr, sr, filename, title="Error Map"):
    """Create and save an error map visualization"""
    hr_np = hr.squeeze().cpu().numpy() if torch.is_tensor(hr) else hr
    sr_np = sr.squeeze().cpu().numpy() if torch.is_tensor(sr) else sr
    error_map = np.abs(hr_np - sr_np)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_figure(lr, sr, hr, model_name, epoch, psnr, ssim, filename):
    """Create and save a comprehensive comparison figure"""
    lr_np = lr.squeeze().cpu().numpy() if torch.is_tensor(lr) else lr
    sr_np = sr.squeeze().cpu().numpy() if torch.is_tensor(sr) else sr
    hr_np = hr.squeeze().cpu().numpy() if torch.is_tensor(hr) else hr
    error_map = np.abs(hr_np - sr_np)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # LR image
    axes[0, 0].imshow(lr_np, cmap='gray')
    axes[0, 0].set_title("Low Resolution Input")
    axes[0, 0].axis('off')
    
    # SR image
    axes[0, 1].imshow(sr_np, cmap='gray')
    axes[0, 1].set_title(f"{model_name} Output\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    axes[0, 1].axis('off')
    
    # HR image
    axes[1, 0].imshow(hr_np, cmap='gray')
    axes[1, 0].set_title("High Resolution (Ground Truth)")
    axes[1, 0].axis('off')
    
    # Error map
    im = axes[1, 1].imshow(error_map, cmap='hot')
    axes[1, 1].set_title("Error Map")
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{model_name} - Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(train_losses, val_psnr, val_ssim, model_name, filename):
    """Plot and save training history"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title(f"{model_name} - Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Plot validation PSNR
    ax2.plot(val_psnr)
    ax2.set_title(f"{model_name} - Validation PSNR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True)
    
    # Plot validation SSIM
    ax3.plot(val_ssim)
    ax3.set_title(f"{model_name} - Validation SSIM")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SSIM")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(metrics_dict, filename):
    """Create comparison plots between all models"""
    # PSNR comparison
    plt.figure(figsize=(10, 6))
    for model_name, metrics in metrics_dict.items():
        if 'val_psnr' in metrics:
            plt.plot(metrics['val_psnr'], label=model_name)
    plt.title("PSNR Comparison Across Models")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename.replace('.png', '_psnr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SSIM comparison
    plt.figure(figsize=(10, 6))
    for model_name, metrics in metrics_dict.items():
        if 'val_ssim' in metrics:
            plt.plot(metrics['val_ssim'], label=model_name)
    plt.title("SSIM Comparison Across Models")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename.replace('.png', '_ssim.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss comparison
    plt.figure(figsize=(10, 6))
    for model_name, metrics in metrics_dict.items():
        if 'train_losses' in metrics:
            plt.plot(metrics['train_losses'], label=model_name)
    plt.title("Training Loss Comparison Across Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename.replace('.png', '_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# 6. Model Training Function
# ============================================
def train_model(model, model_name, train_loader, test_loader, criterion, optimizer, 
                scheduler, device, epochs=100, save_interval=10):
    
    # Training history
    train_losses = []
    val_psnr = []
    val_ssim = []
    learning_rates = []
    
    # Best model tracking
    best_psnr = 0
    best_model_path = ""
    
    # Results dictionary to store all metrics
    results = {
        'model_name': model_name,
        'start_time': datetime.now().isoformat(),
        'train_losses': [],
        'val_psnr': [],
        'val_ssim': [],
        'learning_rates': [],
        'best_psnr': 0,
        'best_epoch': 0
    }
    
    print(f"Training {model_name} for {epochs} epochs...")
    
    for epoch in range(1, epochs+1):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for i, (lr, hr, _) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            sr = model(lr)
            
            # Ensure output matches target size
            if sr.shape != hr.shape:
                sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
            
            loss = criterion(sr, hr)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Clear memory
            del sr, loss
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Record training loss
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        results['train_losses'].append(avg_loss)
        
        # Validation phase
        model.eval()
        epoch_psnr, epoch_ssim = [], []
        
        with torch.no_grad():
            for i, (lr, hr, _) in enumerate(test_loader):
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
                
                # Calculate metrics for each image in the batch
                for j in range(sr.shape[0]):
                    psnr_val, ssim_val = calculate_metrics(sr[j:j+1], hr[j:j+1])
                    epoch_psnr.append(psnr_val)
                    epoch_ssim.append(ssim_val)
                
                # Save sample results at intervals
                if i == 0 and (epoch % save_interval == 0 or epoch == 1 or epoch == epochs):
                    sample_idx = 0  # Use first sample in batch
                    save_comparison_figure(
                        lr[sample_idx], sr[sample_idx], hr[sample_idx], 
                        model_name, epoch, 
                        np.mean(epoch_psnr), np.mean(epoch_ssim),
                        f"results/comparisons/{model_name}_epoch_{epoch}.png"
                    )
                    create_error_map(
                        hr[sample_idx], sr[sample_idx],
                        f"results/error_maps/{model_name}_epoch_{epoch}.png",
                        f"{model_name} Error Map - Epoch {epoch}"
                    )
                
                # Clear memory
                del sr
        
        # Record validation metrics
        avg_psnr = np.mean(epoch_psnr)
        avg_ssim = np.mean(epoch_ssim)
        val_psnr.append(avg_psnr)
        val_ssim.append(avg_ssim)
        results['val_psnr'].append(avg_psnr)
        results['val_ssim'].append(avg_ssim)
        
        # Record learning rate
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        results['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            results['best_psnr'] = best_psnr
            results['best_epoch'] = best_epoch
            best_model_path = f"models/{model_name}_best.pth"
            torch.save(model.state_dict(), best_model_path)
        
        # Save checkpoint at intervals
        if epoch % save_interval == 0:
            checkpoint_path = f"models/{model_name}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
        
        print(f"{model_name} Epoch {epoch}/{epochs}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LR={current_lr:.2e}")
    
    # Save final model
    final_model_path = f"models/{model_name}_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Update results
    results['end_time'] = datetime.now().isoformat()
    results['final_model_path'] = final_model_path
    results['best_model_path'] = best_model_path
    
    # Plot training history
    plot_training_history(train_losses, val_psnr, val_ssim, model_name, 
                         f"results/graphs/{model_name}_training_history.png")
    
    print(f"{model_name} training completed!")
    print(f"Best PSNR: {best_psnr:.4f} at epoch {best_epoch}")
    
    return results

# ============================================
# 7. Main Execution
# ============================================
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set your M3FD dataset path here
    dataset_path = "Ir"  # Update this to your M3FD infrared images path
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Dictionary to store all results
    all_results = {}
    
    # Train EEDSR model
    eedsr_model = EEDSR(scale=4).to(device)
    eedsr_criterion = EdgeAwareLoss().to(device)
    eedsr_optimizer = torch.optim.Adam(eedsr_model.parameters(), lr=1e-4)
    eedsr_scheduler = torch.optim.lr_scheduler.StepLR(eedsr_optimizer, step_size=30, gamma=0.5)
    
    eedsr_results = train_model(
        eedsr_model, "EEDSR", train_loader, test_loader, 
        eedsr_criterion, eedsr_optimizer, eedsr_scheduler, 
        device, epochs=100, save_interval=10
    )
    all_results["EEDSR"] = eedsr_results
    
    # Train SRCNN model for comparison
    srcnn_model = SRCNN().to(device)
    srcnn_criterion = nn.MSELoss()
    srcnn_optimizer = torch.optim.Adam(srcnn_model.parameters(), lr=1e-4)
    srcnn_scheduler = torch.optim.lr_scheduler.StepLR(srcnn_optimizer, step_size=30, gamma=0.5)
    
    srcnn_results = train_model(
        srcnn_model, "SRCNN", train_loader, test_loader, 
        srcnn_criterion, srcnn_optimizer, srcnn_scheduler, 
        device, epochs=100, save_interval=10
    )
    all_results["SRCNN"] = srcnn_results
    
    # Train EDSR model for comparison
    edsr_model = EDSR().to(device)
    edsr_criterion = nn.MSELoss()
    edsr_optimizer = torch.optim.Adam(edsr_model.parameters(), lr=1e-4)
    edsr_scheduler = torch.optim.lr_scheduler.StepLR(edsr_optimizer, step_size=30, gamma=0.5)
    
    edsr_results = train_model(
        edsr_model, "EDSR", train_loader, test_loader, 
        edsr_criterion, edsr_optimizer, edsr_scheduler, 
        device, epochs=100, save_interval=10
    )
    all_results["EDSR"] = edsr_results
    
    # Create model comparison plots
    plot_model_comparison(all_results, "results/graphs/model_comparison.png")
    
    # Save all results to a JSON file
    with open("results/training_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    print("All training completed! Results saved to 'results/' directory.")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for model_name, results in all_results.items():
        best_psnr = results.get('best_psnr', 0)
        best_epoch = results.get('best_epoch', 0)
        final_psnr = results.get('val_psnr', [0])[-1]
        final_ssim = results.get('val_ssim', [0])[-1]
        print(f"{model_name}: Best PSNR {best_psnr:.4f} (Epoch {best_epoch}), Final PSNR {final_psnr:.4f}, Final SSIM {final_ssim:.4f}")
    print("="*60)