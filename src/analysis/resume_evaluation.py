import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import gc
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set matplotlib cache directory to a writable location
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Function to find the least used GPU
def get_available_gpu():
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        free = total - allocated
        
        # Consider GPU available if it has more than 20GB free
        if free > 20:
            available_gpus.append((i, free))
    
    # Sort by available memory (descending)
    available_gpus.sort(key=lambda x: x[1], reverse=True)
    
    if available_gpus:
        return available_gpus[0][0]  # Return GPU with most free memory
    else:
        return 0  # Default to first GPU if none meet criteria

# Set CUDA device to the most available GPU
available_gpu = get_available_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpu)
print(f"Using GPU: {available_gpu}")

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

class InfraredDataset(Dataset):
    def __init__(self, folder, scale=4, crop_size=512):
        self.all_files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.files = [f for f in self.all_files if cv2.imread(f, cv2.IMREAD_GRAYSCALE) is not None]
        self.scale = scale
        self.crop_size = crop_size
        self.transform = ToTensor()

        # Check for invalid images
        for f in self.all_files:
            if f not in self.files:
                print(f"Skipping invalid image: {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            idx = (idx + 1) % len(self.files)
            img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        
        # Center crop or resize
        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            img = img[top:top+self.crop_size, left:left+self.crop_size]
        
        hr = self.transform(img)
        
        # Downsample for LR
        lr_img = cv2.resize(img, (self.crop_size // self.scale, self.crop_size // self.scale), 
                           interpolation=cv2.INTER_CUBIC)
        lr_img = cv2.resize(lr_img, (self.crop_size, self.crop_size), 
                           interpolation=cv2.INTER_CUBIC)
        lr = self.transform(lr_img)
        
        return lr, hr

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

# Fixed EDSR Model - removed upscaling to match target size
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
        # Final convolution to get the output to the right number of channels
        self.final_conv = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x += residual
        x = self.final_conv(x)
        return x

# Evaluation function
def evaluate_model(model, dataloader, model_name="Model"):
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            
            # Convert tensors to numpy arrays
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            
            for i in range(sr_np.shape[0]):
                sr_img = np.squeeze(sr_np[i])
                hr_img = np.squeeze(hr_np[i])
                
                # Calculate metrics
                psnr = peak_signal_noise_ratio(hr_img, sr_img, data_range=1)
                ssim = structural_similarity(hr_img, sr_img, data_range=1)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"{model_name} - Average PSNR: {avg_psnr:.2f}")
    print(f"{model_name} - Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim, psnr_values, ssim_values

# Statistical validation function
def statistically_compare_models(model_a, model_b, dataloader, runs=5):
    psnr_a_list, ssim_a_list = [], []
    psnr_b_list, ssim_b_list = [], []

    for run in range(runs):
        print(f"Run {run+1}/{runs}")
        psnr_a, ssim_a, _, _ = evaluate_model(model_a, dataloader, "SRCNN")
        psnr_b, ssim_b, _, _ = evaluate_model(model_b, dataloader, "EDSR")

        psnr_a_list.append(psnr_a)
        ssim_a_list.append(ssim_a)
        psnr_b_list.append(psnr_b)
        ssim_b_list.append(ssim_b)

    # Calculate averages and standard deviations
    print(f"\nSRCNN PSNR: {np.mean(psnr_a_list):.2f} ± {np.std(psnr_a_list):.2f}")
    print(f"SRCNN SSIM: {np.mean(ssim_a_list):.4f} ± {np.std(ssim_a_list):.4f}")
    print(f"EDSR  PSNR: {np.mean(psnr_b_list):.2f} ± {np.std(psnr_b_list):.2f}")
    print(f"EDSR  SSIM: {np.mean(ssim_b_list):.4f} ± {np.std(ssim_b_list):.4f}")
    
    return (psnr_a_list, ssim_a_list, psnr_b_list, ssim_b_list)

# Error mapping function
def create_error_map(hr, sr, title, filename=None):
    """Create a heatmap of the absolute error between HR and SR images"""
    error = np.abs(hr - sr)
    plt.figure(figsize=(10, 8))
    plt.imshow(error, cmap='hot')
    plt.colorbar()
    plt.title(f"Error Map: {title}")
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Create dataset and dataloader
folder = 'Ir'
dataset = InfraredDataset(folder, scale=4, crop_size=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize models
srcnn_model = SRCNN().to(device)
edsr_model = EDSR(num_blocks=8, channels=64).to(device)

# Load pre-trained models
srcnn_model_path = 'srcnn_model_50epochs.pth'
edsr_model_path = 'edsr_model_50epochs.pth'

if os.path.exists(srcnn_model_path):
    print(f"Loading pre-trained SRCNN model from {srcnn_model_path}")
    srcnn_model.load_state_dict(torch.load(srcnn_model_path, map_location=device))
    srcnn_model.eval()
    print("SRCNN model loaded successfully!")
else:
    print("SRCNN model not found. Please train it first.")
    exit()

if os.path.exists(edsr_model_path):
    print(f"Loading pre-trained EDSR model from {edsr_model_path}")
    edsr_model.load_state_dict(torch.load(edsr_model_path, map_location=device))
    edsr_model.eval()
    print("EDSR model loaded successfully!")
else:
    print("EDSR model not found. Please train it first.")
    exit()

# Load training losses if available
srcnn_losses = []
edsr_losses = []
if os.path.exists('model_evaluation_results.json'):
    try:
        with open('model_evaluation_results.json', 'r') as f:
            results = json.load(f)
            srcnn_losses = results.get("SRCNN", {}).get("Training_losses", [])
            edsr_losses = results.get("EDSR", {}).get("Training_losses", [])
    except:
        print("Could not load training losses from previous run")

# Evaluate models
print("\nEvaluating models...")
srcnn_psnr, srcnn_ssim, srcnn_psnr_vals, srcnn_ssim_vals = evaluate_model(srcnn_model, dataloader, "SRCNN")
edsr_psnr, edsr_ssim, edsr_psnr_vals, edsr_ssim_vals = evaluate_model(edsr_model, dataloader, "EDSR")

# Statistical validation
print("\nRunning statistical validation (5 runs each)...")
psnr_srcnn, ssim_srcnn, psnr_edsr, ssim_edsr = statistically_compare_models(srcnn_model, edsr_model, dataloader, runs=5)

# Save evaluation results to a JSON file
results = {
    "SRCNN": {
        "PSNR": float(srcnn_psnr),
        "SSIM": float(srcnn_ssim),
        "PSNR_values": [float(v) for v in srcnn_psnr_vals],
        "SSIM_values": [float(v) for v in srcnn_ssim_vals],
        "Training_losses": [float(v) for v in srcnn_losses]
    },
    "EDSR": {
        "PSNR": float(edsr_psnr),
        "SSIM": float(edsr_ssim),
        "PSNR_values": [float(v) for v in edsr_psnr_vals],
        "SSIM_values": [float(v) for v in edsr_ssim_vals],
        "Training_losses": [float(v) for v in edsr_losses]
    },
    "Statistical_validation": {
        "SRCNN_PSNR_mean_std": [float(np.mean(psnr_srcnn)), float(np.std(psnr_srcnn))],
        "SRCNN_SSIM_mean_std": [float(np.mean(ssim_srcnn)), float(np.std(ssim_srcnn))],
        "EDSR_PSNR_mean_std": [float(np.mean(psnr_edsr)), float(np.std(psnr_edsr))],
        "EDSR_SSIM_mean_std": [float(np.mean(ssim_edsr)), float(np.std(ssim_edsr))]
    }
}

with open('model_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Evaluation results saved to 'model_evaluation_results.json'")

# Visual comparison function
def visualize_comparison(srcnn_model, edsr_model, dataset, num_samples=3):
    for sample_idx in range(num_samples):
        # Get a sample
        lr, hr = dataset[sample_idx]
        lr = lr.unsqueeze(0).to(device)
        
        # Model predictions
        with torch.no_grad():
            srcnn_model.eval()
            edsr_model.eval()
            
            srcnn_output = srcnn_model(lr).cpu().squeeze().numpy()
            edsr_output = edsr_model(lr).cpu().squeeze().numpy()
        
        hr_img = hr.squeeze().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display images
        axes[0, 0].imshow(lr.cpu().squeeze(), cmap='gray')
        axes[0, 0].set_title('Low Resolution Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(srcnn_output, cmap='gray')
        axes[0, 1].set_title(f'SRCNN Output (PSNR: {peak_signal_noise_ratio(hr_img, srcnn_output, data_range=1):.2f})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(edsr_output, cmap='gray')
        axes[0, 2].set_title(f'EDSR Output (PSNR: {peak_signal_noise_ratio(hr_img, edsr_output, data_range=1):.2f})')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(hr_img, cmap='gray')
        axes[1, 0].set_title('High Resolution (Ground Truth)')
        axes[1, 0].axis('off')
        
        # Error maps
        axes[1, 1].imshow(np.abs(hr_img - srcnn_output), cmap='hot')
        axes[1, 1].set_title('SRCNN Error Map')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(np.abs(hr_img - edsr_output), cmap='hot')
        axes[1, 2].set_title('EDSR Error Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'model_comparison_sample_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual error maps
        create_error_map(hr_img, srcnn_output, f"SRCNN - Sample {sample_idx+1}", f"srcnn_error_map_{sample_idx+1}.png")
        create_error_map(hr_img, edsr_output, f"EDSR - Sample {sample_idx+1}", f"edsr_error_map_{sample_idx+1}.png")

# Create visual comparison
print("\nCreating visual comparisons...")
visualize_comparison(srcnn_model, edsr_model, dataset, num_samples=3)

# Print summary
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
print(f"SRCNN Final PSNR: {srcnn_psnr:.2f}, SSIM: {srcnn_ssim:.4f}")
print(f"EDSR Final PSNR: {edsr_psnr:.2f}, SSIM: {edsr_ssim:.4f}")
print(f"PSNR Improvement: {edsr_psnr - srcnn_psnr:.2f} dB")
print(f"SSIM Improvement: {edsr_ssim - srcnn_ssim:.4f}")
print("="*50)

print("\nAll operations completed successfully!")
print("Files created/updated:")
print("- model_evaluation_results.json (evaluation metrics)")
print("- model_comparison_sample_*.png (visual comparisons)")
print("- *_error_map_*.png (error maps)")