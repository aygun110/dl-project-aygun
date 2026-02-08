import torch
from torch.utils.data import DataLoader
from dataset_identity import CelebAIdentity
from models.facenet import FaceNet

# 1. Настройка устройства

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# 2. Загружаем датасет

dataset = CelebAIdentity(
    "data/img_align_celeba",
    "data/identity_CelebA.txt"
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
num_classes = dataset.data["label"].nunique()
print(f"Количество классов: {num_classes}")
print(f"Длина датасета: {len(dataset)}")
print(f"Количество батчей: {len(loader)}")

# 3. Создаём модель, loss и optimizer

model = FaceNet(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Обучение

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        emb, logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Прогресс каждые 100 батчей
        if (i + 1) % 100 == 0:
            print(f"Эпоха {epoch+1}, Батч {i+1}/{len(loader)}, Потеря: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Эпоха {epoch+1} завершена, Средняя потеря: {avg_loss:.4f}\n")


# 5. Сохраняем веса модели

torch.save(model.state_dict(), "face_ce.pth")
print("Модель успешно сохранена в face_ce.pth")
