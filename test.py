# ===== Standard imports =====
import argparse
import os
from glob import glob
import nibabel as nib
import numpy as np
import random
from tqdm.auto import tqdm

# ===== Torch and related =====
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToPILImage, Pad, CenterCrop, ToTensor, Resize
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path
#from utils import ConvBlock2d, SimpleEncoder

# ==== ARGPARSE ====
parser = argparse.ArgumentParser(description="Run artifact scoring on a NIfTI image")
parser.add_argument('--nifti-path', type=str, required=True, help='Path to NIfTI file')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
args = parser.parse_args()

# ==== CONFIG ====
MODEL_PATH = "model_a5fb8712.pth"
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
NIFTI_PATH = args.nifti_path

# ==== DEFINE TRANSFORM ====
default_transform = Compose([
    ToPILImage(),
    Pad(400),
    CenterCrop((224, 224)),
])
# ==== MODEL ===
# Network
class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock2d(in_ch, 16, 16),
            ConvBlock2d(16, 64, 64),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )
        #self.activation = nn.Sigmoid()
        #self.activation = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, out_ch)
        #x = self.activation(x)
        x = torch.abs(x)
        return x

# ==== LOAD MODEL ====
model = SimpleEncoder(in_ch=1, out_ch=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["encoder"])
model = model.to(DEVICE)
model.eval()

# ==== LOAD NIFTI ====
nii = nib.load(NIFTI_PATH)
vol = nii.get_fdata()  # shape: (H, W, S)
vol = np.nan_to_num(vol)

# Normalize volume (optional but common)
vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)

# ==== SELECT MIDDLE 60% OF SLICES ====
num_slices = vol.shape[2]
start = int(num_slices * 0.2)
end = int(num_slices * 0.8)
mid_slices = vol[:, :, start:end]

# ==== RUN THROUGH MODEL ====
scores = []

with torch.no_grad():
    for i in range(mid_slices.shape[2]):
        slice_img = mid_slices[:, :, i]  # (H, W)

        # Convert to uint8 PIL image for transforms
        slice_img_uint8 = (slice_img * 255).astype(np.uint8)
        transformed = default_transform(slice_img_uint8)  # Apply transform

        # Convert to torch tensor, shape [1, 1, H, W]
        input_tensor = pil_to_tensor(transformed).float().unsqueeze(0).to(DEVICE)

        output = model(input_tensor)
        score = output.item() if output.numel() == 1 else output.mean().item()
        scores.append(score)

# ==== AVERAGE SCORE ====
avg_score = np.mean(scores)
print(f"Average score for {os.path.basename(NIFTI_PATH)}: {avg_score:.4f}")

# ==== SAVE OUTPUT ====
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"{Path(NIFTI_PATH).stem}_artifact_score.txt"

with open(output_file, "w") as f:
    f.write(f"{NIFTI_PATH}, avg_score, {avg_score:.4f}\n")

print(f"Saved average score to: {output_file}")
