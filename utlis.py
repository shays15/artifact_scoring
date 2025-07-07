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
