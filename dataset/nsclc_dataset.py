import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.utility.dictionary import ConcatItemsd
from monai.transforms.spatial.dictionary import Spacingd

from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.spatial.dictionary import Rand3DElasticd
from monai.transforms.croppad.dictionary import (
    RandSpatialCropd,
    ResizeWithPadOrCropd,
)
from monai.transforms.spatial.dictionary import (
    Orientationd,
    RandAffined,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
)


def get_nsclc_transforms(
    train: bool,
    pet_key: str = "pet",
    ct_key: str = "ct",
    pet_clip: float = 35.0,
    patch: int | None = None,
    spacing=(4.0, 4.0, 4.0),
):
    """
    NSCLC PET/CT preprocessing.

    - Load NIfTI
    - Channel first
    - Reorient to RAS
    - Resample to `spacing`
    - CropForeground using CT as body mask (drop air)
    - Intensity:
        CT: clip [-1000, 1000] -> [0, 1]
        PET: clip [0, pet_clip] -> [0, 1]
    - Optional:
        - patch crop
        - train time augs (affine, elastic, contrast, noise)
    - Concatenate [pet, ct] -> 'image' (2 channels)
    """

    keys = [ct_key, pet_key]

    ops = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        # body crop using CT as mask
        CropForegroundd(
            keys=keys,
            source_key=ct_key,
            select_fn=lambda x: x > -900,
            margin=8,
        ),
        # per modality scaling
        ScaleIntensityRanged(
            keys=[ct_key],
            a_min=-1000.0,
            a_max=1000.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=[pet_key],
            a_min=0.0,
            a_max=float(pet_clip),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    # Optional patch crop (same size for train/val if you want fixed input)
    if patch is not None:
        ops += [
            RandSpatialCropd(
                keys=keys,
                roi_size=(patch, patch, patch),
                random_center=True,
                random_size=False,
            ),
            ResizeWithPadOrCropd(
                keys=keys,
                spatial_size=(patch, patch, patch),
                mode="constant",
            ),
        ]

    # Train time augmentations
    if train:
        ops += [
            RandAffined(
                keys=keys,
                mode=("bilinear", "bilinear"),
                prob=0.5,
                translate_range=(8, 8, 8),
                rotate_range=(0.0, 0.0, 0.26),
                scale_range=(0.10, 0.10, 0.10),
            ),
            Rand3DElasticd(
                keys=keys,
                mode=("bilinear", "bilinear"),
                sigma_range=(0.0, 1.0),
                magnitude_range=(0.0, 1.0),
                prob=0.3,
            ),
            RandAdjustContrastd(keys=[ct_key, pet_key], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=[ct_key, pet_key], prob=0.5),
        ]

    ops += [
        ConcatItemsd(keys=[pet_key, ct_key], name="image", dim=0),
        EnsureTyped(keys=["image"]),
    ]

    return Compose(ops)


class NsclcImageDataset(Dataset):
    def __init__(self, manifest_csv, transforms=None):
        df = pd.read_csv(manifest_csv)
        required_cols = {
            "case_id",
            "pet_path",
            "ct_path",
            "event_time",
            "event_indicator",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        pet_path = row["pet_path"]
        ct_path = row["ct_path"]
        event_time = float(row["event_time"])
        event_indicator = int(row["event_indicator"])

        data = {
            "pet": pet_path,
            "ct": ct_path,
        }
        if self.transforms is not None:
            data = self.transforms(data)

        image = data["image"]  # [2, D, H, W]

        return {
            "case_id": case_id,
            "image": image,
            "time": torch.tensor(event_time, dtype=torch.float32),
            "event": torch.tensor(event_indicator, dtype=torch.float32),
        }
