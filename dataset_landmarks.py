import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CelebALandmarks(Dataset):
    def __init__(self, img_dir, landmark_file):
        self.img_dir = img_dir
        self.landmarks = pd.read_csv(
            landmark_file,
            sep=r"\s+",
            skiprows=2,
            header=None      # <- добавлено
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        row = self.landmarks.iloc[idx]
        img_name = row[0]
        points = row[1:].values.astype("float32")

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        points = torch.tensor(points)

        return image, points

if __name__ == "__main__":
    img_dir = "data/img_align_celeba"
    landmark_file = "data/list_landmarks_align_celeba.txt"

    if not os.path.exists(img_dir):
        print(f"Папка с изображениями не найдена: {img_dir}")
    elif not os.path.exists(landmark_file):
        print(f"Файл с ключевыми точками не найден: {landmark_file}")
    else:
        dataset = CelebALandmarks(img_dir, landmark_file)
        print("Длина датасета:", len(dataset))
        img, points = dataset[0]
        print("Размер изображения:", img.shape)
        print("Ключевые точки:", points)
