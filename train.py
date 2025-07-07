# ===== Standard imports =====
import os
from glob import glob
import nibabel as nib
import numpy as np
import random
from tqdm.auto import tqdm

# ===== Torch and related =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToPILImage, Pad, CenterCrop, ToTensor, Resize
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


# ===== Project-specific =====
from utils import ArtifactTransform, AddBackgroundNoise, ConvBlock2d, SimpleEncoder, ContrastiveMRIDataset

from PIL import Image
import numpy as np
import torchio as tio
from torchio.data.subject import Subject
from torchio.transforms import Lambda
from torchio.transforms.augmentation import RandomTransform
from torchio.typing import TypeRangeFloat
from torchio.data.io import nib_to_sitk
from radifox.utils.degrade.degrade import WINDOW_OPTIONS, fwhm_needed, select_kernel
from radifox.utils.resize.pytorch import resize
import SimpleITK as sitk
import os

artifact_transform = ArtifactTransform()

default_transform = Compose([
    ToPILImage(),
    AddBackgroundNoise(snr=30),  # Add noise with SNR = 20
    Pad(400),
    CenterCrop((224, 224))
])

# Define dataset parameters
mode = 'train'
GPUID = 0
encoder_out_dim = 1
pretrained_weight_fpath = None
lr = 1e-3
device = torch.device(f"cuda:{GPUID}")

# Define dataset parameters
dataset_dirs = ['/iacl/pg23/savannah/data_1mm/slices_norm/treatms-0100/']

# Create train and validation datasets
train_dataset = ContrastiveMRIDataset(dataset_dirs, mode='train')
val_dataset = ContrastiveMRIDataset(dataset_dirs, mode='valid')

# Check the size of each dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

def save_progress(artifact_encoder, img_a, img_b, img_with_artifacts, save_dir, epoch, batch_idx, mode, target_size=(224, 224)):
    def resize_or_pad(img, target_size):
        # Resize or pad each tensor to the target size
        _, _, h, w = img.size()
        if h != target_size[0] or w != target_size[1]:
            # Pad or resize to the target size
            img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        return img

    # Resize or pad the images to the target size
    img_a = resize_or_pad(img_a[:4], target_size)
    img_b = resize_or_pad(img_b[:4], target_size)
    img_with_artifacts = resize_or_pad(img_with_artifacts[:4], target_size)

    # Create a grid of images for visualization
    img_grid = make_grid(
        torch.cat([img_a, img_b, img_with_artifacts], dim=0),  # Take first 4 from each for visualization
        nrow=4,  # Arrange them in rows
        normalize=True
    )

    # Save the grid as a PNG file
    save_dir_images = os.path.join(save_dir, "images")
    os.makedirs(save_dir_images, exist_ok=True)
    img_path = os.path.join(save_dir_images, f"{mode}_epoch_{epoch}_iter_{batch_idx}.png")
    save_image(img_grid, img_path)

    # Evaluate scores for the saved images
    scores_a = artifact_encoder(img_a).view(-1).tolist()
    scores_b = artifact_encoder(img_b).view(-1).tolist()
    scores_artifacts = artifact_encoder(img_with_artifacts).view(-1).tolist()

    # Save scores to a TXT file
    save_dir_scores = os.path.join(save_dir, "scores")
    os.makedirs(save_dir_scores, exist_ok=True)
    scores_path = os.path.join(save_dir_scores, f"{mode}_scores_epoch_{epoch}.txt")
    with open(scores_path, "a") as f:
        f.write(f"Iteration {batch_idx}\n")
        f.write(f"Scores img_a: {scores_a}\n")
        f.write(f"Scores img_b: {scores_b}\n")
        f.write(f"Scores img_with_artifacts: {scores_artifacts}\n")
        f.write("\n")

exp_name = 'exp_name'
GPUID = 1
encoder_out_dim = 1
pretrained_weight_fpath = None
lr = 1e-3

device = torch.device(f"cuda:{GPUID}")

artifact_encoder = SimpleEncoder(in_ch=1, out_ch=encoder_out_dim).to(device)

if pretrained_weight_fpath:
    checkpoint = torch.load(pretrained_weight_fpath, map_location=device)
    artifact_encoder.load_state_dict(checkpoint["encoder"])

opt = torch.optim.Adam(artifact_encoder.parameters(), lr=lr, weight_decay=1e-5)
writer = SummaryWriter(log_dir=f"./{exp_name}/runs")

batch_size = 1
train_ds = ContrastiveMRIDataset(dataset_dirs, "train")
val_ds = ContrastiveMRIDataset(dataset_dirs, "valid")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
epochs = 15
save_dir = f"./{exp_name}/checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, epochs + 1):
    # Training phase
    artifact_encoder.train()
    mode = "train"
    avg_epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Training")):
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        img_with_artifacts = batch["img_with_artifacts"].to(device)
        artifact_severity = batch["artifact_severity"].to(device)

        # Forward pass through the encoder
        features_a = artifact_encoder(img_a)
        features_b = artifact_encoder(img_b)
        features_artifacts = artifact_encoder(img_with_artifacts)

        loss_good_near_zero = (torch.sqrt(features_a ** 2) + torch.sqrt(features_b ** 2)).mean()
        triplet_loss = nn.TripletMarginLoss(margin=artifact_severity.mean().item())

        contrastive_loss = triplet_loss(features_a, features_b, features_artifacts).mean()
        loss = contrastive_loss + (loss_good_near_zero*10)

        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_epoch_loss += loss.item()

        writer.add_scalar("Train Loss/L2", loss_good_near_zero.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar("Train Loss/Triplet", contrastive_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar("Train Loss/Total", loss.item(), epoch * len(train_loader) + batch_idx)

        # Save images and scores every few iterations
        if batch_idx % 50 == 0:
            save_progress(artifact_encoder, img_a, img_b, img_with_artifacts, save_dir, epoch, batch_idx, mode)

    print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss / len(train_loader)}")

    # Validation phase
    artifact_encoder.eval()
    mode = "val"
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} - Validation")):
            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            img_with_artifacts = batch["img_with_artifacts"].to(device)

            # Forward pass through the encoder
            features_a = artifact_encoder(img_a)
            features_b = artifact_encoder(img_b)
            features_artifacts = artifact_encoder(img_with_artifacts)

            loss_good_near_zero = (torch.sqrt(features_a ** 2) + torch.sqrt(features_b ** 2)).mean()
            # triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
            triplet_loss = nn.TripletMarginLoss(margin=artifact_severity.mean().item())

            contrastive_loss = triplet_loss(features_a, features_b, features_artifacts).mean()
            loss = contrastive_loss + (loss_good_near_zero*10)

            val_loss += loss.item()
            writer.add_scalar("Val Loss/L2", loss_good_near_zero.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Val Loss/Triplet", contrastive_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Val Loss/Total", loss.item(), epoch * len(train_loader) + batch_idx)

            # Save images and scores every so often
            if batch_idx % 50 == 0:
                save_progress(artifact_encoder, img_a, img_b, img_with_artifacts, save_dir, epoch, batch_idx, mode)

        print(f"Epoch {epoch}: Val Loss = {val_loss / len(train_loader)}")

    # Save model checkpoint after each epoch
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        "encoder": artifact_encoder.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
    }, checkpoint_path)
    print(f"Model saved at {checkpoint_path}")
