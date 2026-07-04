import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import cv2
import json
from glob import glob
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.cluster import KMeans
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set matplotlib parameters for publication-quality figures00
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create directories for results
output_dir = 'analysis_results_02'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/error_maps', exist_ok=True)
os.makedirs(f'{output_dir}/comparisons', exist_ok=True)
os.makedirs(f'{output_dir}/metrics', exist_ok=True)
os.makedirs(f'{output_dir}/diverse_samples', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# 1. Enhanced Dataset Loader with Diverse Sampling
# ============================================
class M3FDDataset:
    def __init__(self, image_dir, scale=4, crop_size=512):
        self.image_dir = image_dir
        self.scale = scale
        self.crop_size = crop_size
        
        # Get list of image files and sort them numerically
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")) + 
                                 glob(os.path.join(image_dir, "*.png")) +
                                 glob(os.path.join(image_dir, "*.bmp")))
        
        # Sort numerically by filename (assuming they're named as numbers)
        self.image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Extract and analyze image features for diversity sampling
        self.image_features = self.extract_image_features()
        
    def extract_image_features(self):
        """Extract features from images to help with diverse sampling"""
        features = []
        
        for img_path in self.image_files[:500]:  # Sample first 500 for speed
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Calculate various image features that might correlate with weather conditions
                avg_intensity = np.mean(img)
                intensity_std = np.std(img)
                entropy = self.calculate_entropy(img)
                
                # Resize to common size for histogram comparison
                img_resized = cv2.resize(img, (64, 64))
                hist = cv2.calcHist([img_resized], [0], None, [16], [0, 256]).flatten()
                
                features.append([avg_intensity, intensity_std, entropy] + list(hist))
        
        return np.array(features) if features else None

    def calculate_entropy(self, image):
        """Calculate image entropy - higher entropy might indicate more complex scenes"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def get_diverse_indices(self, n_samples=10):
        """Select diverse samples using clustering"""
        if self.image_features is None or len(self.image_features) < n_samples:
            # Fallback to uniform sampling if we don't have enough features
            step = max(1, len(self.image_files) // n_samples)
            return list(range(0, len(self.image_files), step))[:n_samples]
        
        # Use K-means to cluster images
        n_clusters = min(n_samples, len(self.image_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.image_features)
        
        # Select one sample from each cluster
        diverse_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select the image closest to the cluster center
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(self.image_features[cluster_indices] - center, axis=1)
                diverse_indices.append(cluster_indices[np.argmin(distances)])
        
        # If we need more samples, add random ones
        if len(diverse_indices) < n_samples:
            additional_indices = random.sample(
                [i for i in range(len(self.image_files)) if i not in diverse_indices],
                n_samples - len(diverse_indices)
            )
            diverse_indices.extend(additional_indices)
        
        return diverse_indices

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
        
        # Create LR image by downscaling
        lr_img = cv2.resize(img, (self.crop_size // self.scale, self.crop_size // self.scale), 
                           interpolation=cv2.INTER_CUBIC)
        lr = torch.FloatTensor(lr_img / 255.0).unsqueeze(0)
        
        return lr, hr, os.path.basename(img_path)

# ============================================
# 2. Model Definitions (Same as before)
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

# ============================================
# 3. Analysis Functions
# ============================================
def load_model(model_class, model_path, device):
    """Load a pre-trained model"""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def calculate_metrics(sr, hr):
    """Calculate PSNR and SSIM between SR and HR images"""
    sr_np = sr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    
    psnr_val = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    
    # Ensure window size is appropriate for the image size
    min_side = min(hr_np.shape)
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    
    ssim_val = structural_similarity(hr_np, sr_np, data_range=1.0, win_size=win_size)
    return psnr_val, ssim_val

def create_error_map(hr, sr, title, filename):
    """Create and save an error map visualization"""
    hr_np = hr.squeeze().cpu().numpy()
    sr_np = sr.squeeze().cpu().numpy()
    error = np.abs(hr_np - sr_np)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(error, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_figure(lr, sr_dict, hr, filename, sample_name):
    """Create a comprehensive comparison figure for all models"""
    num_models = len(sr_dict)
    fig, axes = plt.subplots(2, num_models + 1, figsize=(5*(num_models+1), 10))
    
    # Display LR image
    lr_np = lr.squeeze().cpu().numpy()
    axes[0, 0].imshow(lr_np, cmap='gray')
    axes[0, 0].set_title("Low Resolution Input")
    axes[0, 0].axis('off')
    
    # Display HR image
    hr_np = hr.squeeze().cpu().numpy()
    axes[1, 0].imshow(hr_np, cmap='gray')
    axes[1, 0].set_title("High Resolution (Ground Truth)")
    axes[1, 0].axis('off')
    
    # Display SR results and error maps for each model
    for i, (model_name, sr) in enumerate(sr_dict.items(), 1):
        sr_np = sr.squeeze().cpu().numpy()
        psnr, ssim = calculate_metrics(sr, hr)
        
        # SR output
        axes[0, i].imshow(sr_np, cmap='gray')
        axes[0, i].set_title(f"{model_name}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}")
        axes[0, i].axis('off')
        
        # Error map
        error = np.abs(hr_np - sr_np)
        im = axes[1, i].imshow(error, cmap='hot')
        axes[1, i].set_title(f"{model_name} Error Map")
        axes[1, i].axis('off')
        
        # Add colorbar for error map
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Super-Resolution Results for {sample_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_models(models_dict, data_loader, device, num_samples=5):
    """Evaluate all models on the test dataset"""
    results = {model_name: {"psnr": [], "ssim": []} for model_name in models_dict.keys()}
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, (lr, hr, img_name) in enumerate(data_loader):
            lr, hr = lr.to(device), hr.to(device)
            sr_dict = {}
            
            # Get SR outputs from all models
            for model_name, model in models_dict.items():
                sr = model(lr)
                # Ensure output matches target size
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
                
                sr_dict[model_name] = sr
                
                # Calculate metrics
                psnr, ssim = calculate_metrics(sr, hr)
                results[model_name]["psnr"].append(psnr)
                results[model_name]["ssim"].append(ssim)
            
            # Save sample results for visualization
            if batch_idx < num_samples:
                sample_results.append({
                    "lr": lr,
                    "hr": hr,
                    "sr_dict": sr_dict,
                    "img_name": img_name[0]
                })
    
    # Calculate average metrics
    avg_results = {}
    for model_name, metrics in results.items():
        avg_results[model_name] = {
            "psnr": np.mean(metrics["psnr"]),
            "ssim": np.mean(metrics["ssim"]),
            "psnr_std": np.std(metrics["psnr"]),
            "ssim_std": np.std(metrics["ssim"]),
            "count": len(metrics["psnr"])
        }
    
    return avg_results, results, sample_results

def plot_metrics_comparison(avg_results, filename, title_suffix=""):
    """Create a bar chart comparing metrics across models"""
    models = list(avg_results.keys())
    psnr_vals = [avg_results[m]["psnr"] for m in models]
    ssim_vals = [avg_results[m]["ssim"] for m in models]
    psnr_err = [avg_results[m]["psnr_std"] for m in models]
    ssim_err = [avg_results[m]["ssim_std"] for m in models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR comparison
    x_pos = np.arange(len(models))
    bars = ax1.bar(x_pos, psnr_vals, yerr=psnr_err, align='center', alpha=0.7, capsize=10)
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_title(f'PSNR Comparison {title_suffix}')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{psnr_vals[i]:.2f}', ha='center', va='bottom')
    
    # SSIM comparison
    bars = ax2.bar(x_pos, ssim_vals, yerr=ssim_err, align='center', alpha=0.7, capsize=10)
    ax2.set_ylabel('SSIM')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_title(f'SSIM Comparison {title_suffix}')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{ssim_vals[i]:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# 4. Main Analysis Script
# ============================================
def main():
    # Set your dataset path and model paths
    dataset_path = "Ir"  # Update this to your dataset path
    model_paths = {
        "SRCNN": "srcnn_model_50epochs.pth",  # Update with your actual paths
        "EDSR": "edsr_model_50epochs.pth",
        "EEDSR": "EEDSR_final.pth"
    }
    
    # Create dataset
    dataset = M3FDDataset(dataset_path, scale=4, crop_size=512)
    
    # Get diverse sample indices (using 800-step sampling as requested)
    step_size = 800
    diverse_indices = list(range(0, len(dataset), step_size))
    
    # If we don't have enough samples, add some random ones
    if len(diverse_indices) < 5:
        additional_indices = random.sample(
            [i for i in range(len(dataset)) if i not in diverse_indices],
            5 - len(diverse_indices)
        )
        diverse_indices.extend(additional_indices)
    
    print(f"Selected {len(diverse_indices)} diverse samples: {diverse_indices}")
    
    # Create a subset with these diverse samples
    diverse_subset = Subset(dataset, diverse_indices)
    data_loader = DataLoader(diverse_subset, batch_size=1, shuffle=False)
    
    # Load models
    models_dict = {}
    for model_name, model_path in model_paths.items():
        if model_name == "SRCNN":
            models_dict[model_name] = load_model(SRCNN, model_path, device)
        elif model_name == "EDSR":
            models_dict[model_name] = load_model(EDSR, model_path, device)
        elif model_name == "EEDSR":
            models_dict[model_name] = load_model(EEDSR, model_path, device)
        print(f"Loaded {model_name} model from {model_path}")
    
    # Evaluate models on diverse samples
    print("Evaluating models on diverse samples...")
    avg_results, all_results, sample_results = evaluate_models(
        models_dict, data_loader, device, num_samples=len(diverse_indices)
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Diverse Samples)")
    print("="*60)
    for model_name, metrics in avg_results.items():
        print(f"{model_name}: PSNR = {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB, "
              f"SSIM = {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Save metrics to JSON file
    with open(f'{output_dir}/metrics/evaluation_results.json', 'w') as f:
        json.dump({
            "diverse_samples": avg_results,
            "sample_indices": diverse_indices
        }, f, indent=4)
    
    # Create metrics comparison plot
    plot_metrics_comparison(avg_results, f'{output_dir}/metrics/model_comparison_diverse.png', "(Diverse Samples)")
    
    # Create visualizations for sample images
    print("\nCreating visualizations for diverse sample images...")
    for i, sample in enumerate(sample_results):
        lr, hr, sr_dict, img_name = sample["lr"], sample["hr"], sample["sr_dict"], sample["img_name"]
        
        # Create comparison figure
        create_comparison_figure(
            lr, sr_dict, hr, 
            f'{output_dir}/comparisons/diverse_sample_{i+1}_{img_name}.png',
            f'Diverse Sample {i+1} ({img_name})'
        )
        
        # Create individual error maps
        for model_name, sr in sr_dict.items():
            create_error_map(
                hr, sr,
                f"{model_name} Error Map - {img_name}",
                f"{output_dir}/error_maps/{model_name}_diverse_sample_{i+1}_{img_name}.png"
            )
    
    print("\nAnalysis complete! Results saved in the 'analysis_results_02' directory.")
    print("Files created:")
    print(f"- {output_dir}/metrics/evaluation_results.json (detailed metrics)")
    print(f"- {output_dir}/metrics/model_comparison_diverse.png (metrics comparison chart)")
    print(f"- {output_dir}/comparisons/ (comparison figures for diverse samples)")
    print(f"- {output_dir}/error_maps/ (individual error maps)")

if __name__ == "__main__":
    main()