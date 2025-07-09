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

from PIL import Image
import numpy as np
import torchio as tio
from torchio.data.subject import Subject
from torchio.transforms import Lambda
from torchio.transforms.augmentation import RandomTransform
from torchio.typing import TypeRangeFloat
from torchio.data.io import nib_to_sitk
import SimpleITK as sitk
import os

# ==== ARGPARSE ====
parser = argparse.ArgumentParser(description="Run artifact scoring on a NIfTI image")
parser.add_argument('--dataset', type=str, required=True, help='Path to training directory')
parser.add_argument('--exp_name', type=str, required=True, help='Output directory name')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
args = parser.parse_args()

class ArtifactTransform:
    def __init__(self):
        """
        Initialize the artifact transform. Artifact transformations and their severity 
        scores will be dynamically generated within the class.
        """
        pass

    def _get_random_noise(self):
        std = random.uniform(0.005, 0.2)
        severity = (std - 0.005) / (0.2 - 0.005)
        params = {"std": std}
        return tio.RandomNoise(std=(std, std)), severity, "RandomNoise", params

    def _get_random_ghosting(self):
        num_ghosts = random.randint(2, 10)
        intensity = random.uniform(0.2, 1.5)
        severity = ((intensity - 0.2) + num_ghosts/10) / ((1.5 - 0.2)+1)
        params = {"num_ghosts": num_ghosts, "intensity": intensity}
        return tio.RandomGhosting(num_ghosts=(num_ghosts, num_ghosts), intensity=(intensity, intensity)), severity, "RandomGhosting", params

    def _get_random_bias_field(self):
        coefficients = random.uniform(0.01, 0.3)
        severity = (coefficients - 0.01) / (0.3 - 0.01)
        params = {"coefficients": coefficients}
        return tio.RandomBiasField(coefficients=(coefficients, coefficients)), severity, "RandomBiasField", params

    def _get_random_downsampling(self):
        scale = random.uniform(1, 4)
        severity = (scale - 1) / (4 - 1)
        params = {"scale": scale}
        return tio.Resample((scale, scale, 1), image_interpolation='linear'), severity, "Downsampling", params


    def __call__(self, image):
        """
        Apply a random artifact transformation to the image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            transformed_image (torch.Tensor): The transformed image.
            severity_level (float): The severity level of the applied artifact.
            artifact_name (str): The name of the chosen artifact.
            parameters (dict): The parameters of the chosen artifact.
        """
        # List of artifact generation methods
        artifact_methods = [
            self._get_random_noise,
            self._get_random_ghosting,
            self._get_random_bias_field,
            self._get_random_downsampling,
        ]

        # Randomly choose an artifact generation method and generate the artifact
        artifact_transform, severity_level, artifact_name, parameters = random.choice(artifact_methods)()

        # Apply the chosen artifact transformation
        transformed_image = artifact_transform(image)

        return transformed_image, severity_level, artifact_name, parameters

class AddBackgroundNoise:
    def __init__(self, snr, background_mask=None):
        self.snr = snr
        self.background_mask = background_mask

    def __call__(self, img):
        # Ensure the input is a NumPy array
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.array(img)

        # Create a background mask if not provided
        if self.background_mask is None:
            background_mask = (img <= 0).astype(np.uint8)  # Zero-intensity as background
        else:
            background_mask = self.background_mask

        # Estimate signal intensity (mean) within the brain region
        brain_region = img[background_mask == 0]  # Invert mask to find brain
        signal_mean = brain_region.mean() if len(brain_region) > 0 else 0

        # Calculate noise standard deviation based on the SNR
        sigma = signal_mean / self.snr

        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, img.shape)

        # Apply noise only to the background
        noisy_img = img + (background_mask * noise)

        # Clip values to valid range (e.g., 0-255 for images)
        # noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        # Convert the result back to a PIL image
        return Image.fromarray(noisy_img)

class ContrastiveMRIDataset(Dataset):
    def __init__(self, dataset_dirs, mode='train'):
        """
        Dataset for contrastive learning with paired MRI images.

        Args:
            dataset_dirs (list): Directories containing MRI data.
            mode (str): Dataset mode ('train', 'val', 'test').
        """
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.img_paths = self._get_files()


    def _get_files(self):
        """
        Collect file paths for the dataset.

        Returns:
            img_paths (list): List of image paths.
        """
        img_paths = []
        for dataset_dir in self.dataset_dirs:
            pattern = os.path.join(dataset_dir, self.mode, f"*.nii.gz")
            niis = sorted(glob(pattern))
            img_paths.extend(niis)  # Use extend to flatten the list
        #print(f"Collected {len(img_paths)} paths: {img_paths[:1]}")  # Debugging
        return img_paths


    def __len__(self):
        #print(f"Dataset length (self.img_paths): {len(self.img_paths)}")
        return len(self.img_paths)

    def get_tensor_from_path(self, img_path):
        # Load the image and apply transformations
        img = nib.load(img_path).get_fdata().astype(np.float32).transpose([1, 0])  # Example: loading NIfTI image
        img = default_transform(img)
        img = ToTensor()(np.array(img))

        return img

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - img_a: A slice from the first image path.
                - img_b: A slice from a different image path.
                - img_with_artifacts: img_a with artifacts applied.
                - artifact_severity: level of artifacts (int)
        """
        # Get two different random image paths
        img_path_a = self.img_paths[idx]
        random_idx = random.randint(0, len(self.img_paths) - 1)
        img_path_b = self.img_paths[random_idx]

        # Load the images
        img_a = self.get_tensor_from_path(img_path_a)
        img_b = self.get_tensor_from_path(img_path_b)

        # Adjust dimensions to be 4D: (channels, x, y, z)
        img_a_4d = img_a.unsqueeze(-1)  # Add channel and depth dimensions

        # Apply artifact transformations
        img_with_artifacts_4d, artifact_severity, artifact_name, params = artifact_transform(img_a_4d)

        # Convert back to (C, H, W)
        img_with_artifacts = img_with_artifacts_4d.squeeze(-1)  # Remove depth dimension

        # Ensure the shape of img_with_artifacts matches img_a
        if img_with_artifacts.shape != img_a.shape:
            img_with_artifacts = torch.nn.functional.interpolate(
                img_with_artifacts.unsqueeze(0),  # Add batch dimension for resizing
                size=img_a.shape[-2:],  # Target size (H, W)
                mode="bilinear",  # Interpolation method
                align_corners=False
            ).squeeze(0)  # Remove batch dimension

        return {
            "img_a": img_a,
            "img_b": img_b,
            "img_with_artifacts": img_with_artifacts,
            "artifact_severity": artifact_severity,
            "artifact_name": artifact_name,
            "parameters": params
        }

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

artifact_transform = ArtifactTransform()

default_transform = Compose([
    ToPILImage(),
    AddBackgroundNoise(snr=30),  # Add noise with SNR = 20
    Pad(400),
    CenterCrop((224, 224))
])

# Define dataset parameters
mode = 'train'
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
encoder_out_dim = 1
lr = 1e-3

dataset_dirs = {args.dataset}
exp_name = args.exp_name
pretrained = args.pretrained

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

artifact_encoder = SimpleEncoder(in_ch=1, out_ch=encoder_out_dim).to(device)

if args.pretrained is not None:
    print(f"Loading weights from {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device)
    artifact_encoder.load_state_dict(checkpoint['encoder'])
else:
    print("Training from scratch.")

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
