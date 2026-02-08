import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAIdentity(Dataset):
    def __init__(self, img_dir, identity_file):

        self.img_dir = img_dir
        self.data = pd.read_csv(identity_file, sep=" ", header=None)
        self.data.columns = ["image", "label"]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        label = torch.tensor(row["label"]-1)

        return image, label
