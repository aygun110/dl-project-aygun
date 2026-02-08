import torch
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.facenet import FaceNet
from dataset_identity import CelebAIdentity

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)


# Загружаем датасет, чтобы узнать количество классов

dataset = CelebAIdentity(
    "data/img_align_celeba",
    "data/identity_CelebA.txt"
)

num_classes = dataset.data["label"].nunique()
print("Количество классов:", num_classes)


# Создаем модель так же, как при обучении

model = FaceNet(num_classes=num_classes).to(device)

# Загружаем веса
model.load_state_dict(torch.load("face_ce.pth", map_location=device))
model.eval()

print("Модель успешно загружена")


# Функция получения embedding

def get_embedding(path):
    print(f"\nОбработка изображения: {path}")

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Изображение не найдено: {path}")

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255
    img = img.to(device)

    with torch.no_grad():
        emb, logits = model(img)

    print("Размер embedding:", emb.shape)

    return emb.cpu().numpy()


# Сравнение двух изображений

emb1 = get_embedding("/Users/aygun/Desktop/lab deep learning/data/img_align_celeba/000001.jpg")
emb2 = get_embedding("/Users/aygun/Desktop/lab deep learning/data/img_align_celeba/000002.jpg")

sim = cosine_similarity(emb1, emb2)
print("\nCosine Similarity:", sim[0][0])
