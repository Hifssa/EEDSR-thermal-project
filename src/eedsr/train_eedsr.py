import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import os
import cv2
from glob import glob
import csv

# Set matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
torch.cuda.empty_cache()

# ====================================================
# 1. Dataset Loader
# ====================================================
class M3FDDataset(Dataset):
    def __init__(self, image_dir, scale=4, crop_size=512):
        self.image_dir = image_dir
        self.scale = scale
        self.crop_size = crop_size
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")) +
                                  glob(os.path.join(image_dir, "*.png")) +
                                  glob(os.path.join(image_dir, "*.bmp")))
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
        lr_img = cv2.resize(img, (self.crop_size // self.scale, self.crop_size // self.scale),
                            interpolation=cv2.INTER_CUBIC)
        lr = torch.FloatTensor(lr_img / 255.0).unsqueeze(0)
        return lr, hr

# ====================================================
# 2. Model
# ====================================================
class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1):
        super().__init__()
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
        super().__init__()
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
        return self.upsample(x)

# ====================================================
# 3. Loss
# ====================================================
class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.2, lambda3=0.01):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3

        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:8].eval()
        except AttributeError:
            vgg = models.vgg19(pretrained=True).features[:8].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg

    def edge_loss(self, sr, hr):
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=sr.device).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=sr.device).view(1,1,3,3)
        def sobel(x): return torch.sqrt(F.conv2d(x,gx,padding=1)**2 + F.conv2d(x,gy,padding=1)**2 + 1e-6)
        return F.l1_loss(sobel(sr), sobel(hr))

    def perceptual_loss(self, sr, hr):
        sr, hr = sr.repeat(1,3,1,1), hr.repeat(1,3,1,1)
        mean = torch.tensor([0.485,0.456,0.406], device=sr.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=sr.device).view(1,3,1,1)
        sr, hr = (sr-mean)/std, (hr-mean)/std
        return F.l1_loss(self.vgg(sr), self.vgg(hr))

    def forward(self, sr, hr):
        return (self.lambda1*self.l1(sr,hr) +
                self.lambda2*self.edge_loss(sr,hr) +
                self.lambda3*self.perceptual_loss(sr,hr))

# ====================================================
# 4. Training + Evaluation
# ====================================================
def calculate_metrics(sr, hr):
    sr_np, hr_np = sr.squeeze().cpu().numpy(), hr.squeeze().cpu().numpy()
    psnr_val = psnr_metric(hr_np, sr_np, data_range=1.0)
    min_dim = min(hr_np.shape)
    win_size = max(3, min(7, min_dim - (min_dim+1)%2))
    ssim_val = ssim_metric(hr_np, sr_np, data_range=1.0, win_size=win_size)
    return psnr_val, ssim_val

def show_results(lr, sr, hr, epoch, save_dir):
    lr_np, sr_np, hr_np = lr.squeeze().cpu().numpy(), sr.squeeze().cpu().numpy(), hr.squeeze().cpu().numpy()
    error_map = np.abs(hr_np - sr_np)
    fig, axs = plt.subplots(1,4,figsize=(16,4))
    axs[0].imshow(lr_np,cmap='gray'); axs[0].set_title("LR")
    axs[1].imshow(sr_np,cmap='gray'); axs[1].set_title("SR")
    axs[2].imshow(hr_np,cmap='gray'); axs[2].set_title("HR")
    im=axs[3].imshow(error_map,cmap='hot'); axs[3].set_title("Error Map")
    fig.colorbar(im, ax=axs[3])
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(os.path.join(save_dir, f"results_epoch_{epoch}.png"))
    plt.close()

def train_model(train_loader, test_loader, save_dir, scale=4, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, criterion = EEDSR(scale=scale).to(device), EdgeAwareLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "metrics.csv")
    with open(log_path,"w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["Epoch","Loss","PSNR","SSIM"])

    train_losses,val_psnr,val_ssim=[],[],[]

    for epoch in range(1,epochs+1):
        model.train(); epoch_loss=0
        for lr,hr in train_loader:
            lr,hr=lr.to(device),hr.to(device)
            sr=model(lr)
            if sr.shape!=hr.shape: sr=F.interpolate(sr,size=hr.shape[2:],mode="bicubic",align_corners=False)
            loss=criterion(sr,hr)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss+=loss.item()
        avg_loss=epoch_loss/len(train_loader); train_losses.append(avg_loss)

        model.eval(); psnr_vals,ssim_vals=[],[]
        with torch.no_grad():
            for i,(lr,hr) in enumerate(test_loader):
                lr,hr=lr.to(device),hr.to(device)
                sr=model(lr)
                if sr.shape!=hr.shape: sr=F.interpolate(sr,size=hr.shape[2:],mode="bicubic",align_corners=False)
                for j in range(sr.shape[0]):
                    p,s=calculate_metrics(sr[j:j+1],hr[j:j+1])
                    psnr_vals.append(p); ssim_vals.append(s)
                if i==0 and epoch%10==0: show_results(lr[0],sr[0],hr[0],epoch,save_dir)
        avg_psnr,avg_ssim=np.mean(psnr_vals),np.mean(ssim_vals)
        val_psnr.append(avg_psnr); val_ssim.append(avg_ssim)
        scheduler.step()

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
        with open(log_path,"a",newline="") as f: csv.writer(f).writerow([epoch,avg_loss,avg_psnr,avg_ssim])
        if epoch%10==0: torch.save(model.state_dict(), os.path.join(save_dir,f"EEDSR_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir,"EEDSR_final.pth"))
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(train_losses); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2); plt.plot(val_psnr,label="PSNR"); plt.plot(val_ssim,label="SSIM")
    plt.title("Validation Metrics"); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(os.path.join(save_dir,"training_history.png")); plt.close()
    return model

# ====================================================
# 5. Main
# ====================================================
if __name__=="__main__":
    dataset_path="Ir"   # <-- update with your dataset folder
    save_dir="results"
    dataset=M3FDDataset(dataset_path,scale=4,crop_size=512)
    train_size=int(0.8*len(dataset)); test_size=len(dataset)-train_size
    train_ds,test_ds=torch.utils.data.random_split(dataset,[train_size,test_size],
                                                   generator=torch.Generator().manual_seed(42))
    train_loader=DataLoader(train_ds,batch_size=4,shuffle=True,num_workers=2)
    test_loader=DataLoader(test_ds,batch_size=4,shuffle=False,num_workers=2)
    train_model(train_loader,test_loader,save_dir,scale=4,epochs=100)
