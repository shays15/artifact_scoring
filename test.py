# ===== Standard imports =====
import os
from glob import glob
import nibabel as nib
import numpy as np
import random
from tqdm.auto import tqdm

# ===== Torch and related =====
import torch
from torchvision.transforms import Compose, ToPILImage, Pad, CenterCrop, ToTensor, Resize
from pathlib import Path
from utils import ConvBlock2d, SimpleEncoder

# ==== CONFIG ====
NIFTI_PATH = args
MODEL_PATH = "/iacl/pg23/savannah/projects/harmonization/eta_encoder_sav/triplet_loss_1d_margin_unregistered_treatms_sites_20250125/checkpoints/models/model_epoch_5.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== DEFINE TRANSFORM ====
default_transform = Compose([
    ToPILImage(),
    Pad(400),
    CenterCrop((224, 224)),
])

# ==== LOAD MODEL ====
model = SimpleEncoder(in_ch=1, out_ch=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
