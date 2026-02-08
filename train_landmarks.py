import torch
from torch.utils.data import DataLoader
from models.landmark_model import LandmarkNet
from dataset_landmarks import CelebALandmarks

# ------------------------------
# 1. Настройка устройства
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# ------------------------------
# 2. Загружаем датасет
# ------------------------------
dataset = CelebALandmarks(
    "data/img_align_celeba",
    "data/list_landmarks_align_celeba.txt"
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Длина датасета: {len(dataset)}")
print(f"Количество батчей: {len(loader)}")

# ------------------------------
# 3. Создаём модель, loss и optimizer
# ------------------------------
model = LandmarkNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# 4. Обучение
# ------------------------------
num_epochs = 2
for epoch in range(num_epochs):
    total_loss = 0
    for i, (imgs, points) in enumerate(loader):
        imgs, points = imgs.to(device), points.to(device)

        preds = model(imgs)
        loss = criterion(preds, points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Вывод прогресса каждые 100 батчей
        if (i + 1) % 100 == 0:
            print(f"Эпоха {epoch+1}, Батч {i+1}/{len(loader)}, Потеря: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Эпоха {epoch+1} завершена, Средняя потеря: {avg_loss:.4f}\n")

# ------------------------------
# 5. Сохраняем веса модели
# ------------------------------
torch.save(model.state_dict(), "landmark_model.pth")
print("Модель успешно сохранена в landmark_model.pth")
