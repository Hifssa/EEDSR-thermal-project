import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import gc
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

# Create dataset and dataloader with smaller batch size
folder = 'Irn'
dataset = InfraredDataset(folder, scale=4, crop_size=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Reduced batch size

# Load your pre-trained SRCNN model
print("Loading pre-trained SRCNN model...")
srcnn_model = SRCNN().to(device)
srcnn_model.load_state_dict(torch.load('srcnn_infrared_model.pth'))
print("SRCNN Model loaded successfully!")

# Evaluate SRCNN model
print("\nEvaluating SRCNN model:")
srcnn_psnr, srcnn_ssim, srcnn_psnr_vals, srcnn_ssim_vals = evaluate_model(srcnn_model, dataloader, "SRCNN")

# Train EDSR model with memory optimizations
print("\nTraining EDSR model with memory optimizations...")
edsr_model = EDSR(num_blocks=8, channels=64).to(device)  # Removed scale parameter
criterion = nn.MSELoss()
optimizer = optim.Adam(edsr_model.parameters(), lr=1e-4)

# Use gradient accumulation to handle smaller batch sizes
accumulation_steps = 4  # Effective batch size = 4 * 4 = 16
num_epochs = 10

for epoch in range(num_epochs):
    edsr_model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (lr, hr) in enumerate(dataloader):
        lr, hr = lr.to(device), hr.to(device)
        
        sr = edsr_model(lr)
        loss = criterion(sr, hr) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear memory
        
        total_loss += loss.item() * accumulation_steps
        
        if i % 10 == 0:
            print(f"EDSR Epoch {epoch+1}/{num_epochs}, Batch {i}, Loss: {loss.item() * accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"EDSR Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the trained EDSR model
torch.save(edsr_model.state_dict(), 'edsr_infrared_model.pth')
print("EDSR Model saved as 'edsr_infrared_model.pth'")

# Evaluate EDSR model
print("\nEvaluating EDSR model:")
edsr_psnr, edsr_ssim, edsr_psnr_vals, edsr_ssim_vals = evaluate_model(edsr_model, dataloader, "EDSR")

# Save evaluation results to a JSON file
results = {
    "SRCNN": {
        "PSNR": float(srcnn_psnr),
        "SSIM": float(srcnn_ssim),
        "PSNR_values": [float(v) for v in srcnn_psnr_vals],
        "SSIM_values": [float(v) for v in srcnn_ssim_vals]
    },
    "EDSR": {
        "PSNR": float(edsr_psnr),
        "SSIM": float(edsr_ssim),
        "PSNR_values": [float(v) for v in edsr_psnr_vals],
        "SSIM_values": [float(v) for v in edsr_ssim_vals]
    }
}

with open('model_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Evaluation results saved to 'model_evaluation_results.json'")

# Visual comparison function
def visualize_comparison(srcnn_model, edsr_model, dataset):
    # Get a sample
    lr, hr = dataset[0]
    lr = lr.unsqueeze(0).to(device)
    
    # Model predictions
    with torch.no_grad():
        srcnn_model.eval()
        edsr_model.eval()
        
        srcnn_output = srcnn_model(lr).cpu().squeeze().numpy()
        edsr_output = edsr_model(lr).cpu().squeeze().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Display images
    axes[0, 0].imshow(lr.cpu().squeeze(), cmap='gray')
    axes[0, 0].set_title('Low Resolution Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hr.squeeze(), cmap='gray')
    axes[0, 1].set_title('High Resolution (Ground Truth)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(srcnn_output, cmap='gray')
    axes[1, 0].set_title(f'SRCNN Output (PSNR: {srcnn_psnr:.2f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edsr_output, cmap='gray')
    axes[1, 1].set_title(f'EDSR Output (PSNR: {edsr_psnr:.2f})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create visual comparison
print("\nCreating visual comparison...")
visualize_comparison(srcnn_model, edsr_model, dataset)

print("\nAll operations completed successfully!")
print("Files created:")
print("- edsr_infrared_model.pth (EDSR model weights)")
print("- model_evaluation_results.json (evaluation metrics)")
print("- model_comparison.png (visual comparison)")