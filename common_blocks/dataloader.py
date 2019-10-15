import os
from sklearn.model_selection import StratifiedKFold, KFold
import cv2
import joblib
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from .utils import make_mask
from .metric import Meter, epoch_log
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, CropNonEmptyMaskIfExists, Resize, Compose,
                            RandomBrightnessContrast, VerticalFlip, RandomBrightness, RandomContrast)
from albumentations.pytorch import ToTensor
import sys

sys.path.append('..')
from configs.train_params import *

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        if crop_image_size is not None:
            list_transforms.extend(
                [CropNonEmptyMaskIfExists(crop_image_size[0], crop_image_size[1], p=0.85),
                 HorizontalFlip(p=0.5),
                 VerticalFlip(p=0.5),
                 RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
                 ])
        else:
            list_transforms.extend(
                [HorizontalFlip(p=0.5),
                 VerticalFlip(p=0.5),
                 RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
                 ]
            )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor()
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider_trai_test_split(
            data_folder,
            df_path,
            phase,
            mean=None,
            std=None,
            batch_size=8,
            num_workers=4,
    ):
        '''
        Returns dataloader for the model training
        '''

        df = pd.read_csv(df_path)
        # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df['defects'] = df.count(axis=1)

        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
        df = train_df if phase == "train" else val_df
        image_dataset = SteelDataset(df, data_folder, mean, std, phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

def provider_cv(
            fold,
            total_folds,
            data_folder,
            df_path,
            phase,
            mean=None,
            std=None,
            batch_size=8,
            num_workers=4,
    ):
        """

        :param fold:
        :param total_folds:
        :param data_folder:
        :param df_path:
        :param phase:
        :param mean:
        :param std:
        :param batch_size:
        :param num_workers:
        :return:

        # example of usage
        dataloader = provider_cv(
        fold=0,
        total_folds=5,
        data_folder=data_folder,
        df_path=train_df_path,
        phase="train",
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225),
        batch_size=16,
        num_workers=4,
    )
        """
        if isDebug:
            df = pd.read_csv(df_path).head(200)
            df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
            df['ClassId'] = df['ClassId'].astype(int)
            df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
            df['defects'] = df.count(axis=1)

            kfold = KFold(total_folds, shuffle=True,
                          random_state=69)  # StratifiedKFold(total_folds, shuffle=True, random_state=69)
            train_idx, val_idx = list(kfold.split(df))[fold]  # , df["defects"]
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        else:
            df = pd.read_csv(df_path)
            df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
            df['ClassId'] = df['ClassId'].astype(int)
            df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
            df['defects'] = df.count(axis=1)
            folds_idx = joblib.load(FOLDS_ids)
            train_idx, val_idx = list(folds_idx)[fold]
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        df = train_df if phase == "train" else val_df
        image_dataset = SteelDataset(df, data_folder, mean, std, phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader
