import torch
from torch.utils.data import DataLoader
from dataset_identity import CelebAIdentity
from models.facenet import FaceNet
from models.arcface import ArcFace
import torch.nn.functional as F

# ------------------------
# 1. Устройство
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

# ------------------------
# 2. Датасет
# ------------------------
dataset = CelebAIdentity(
    "data/img_align_celeba",
    "data/identity_CelebA.txt"
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
num_classes = dataset.data["label"].nunique()

print("Количество классов:", num_classes)
print("Количество батчей:", len(loader))

# ------------------------
# 3. Модель + ArcFace
# ------------------------
model = FaceNet().to(device)                      # без embedding_size
arcface = ArcFace(emb_size=512, num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(arcface.parameters()), lr=1e-3
)

# ------------------------
# 4. Обучение
# ------------------------
epochs = 2  # для быстрого теста
for epoch in range(epochs):
    total_loss = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        # 1. Прямой проход
        embeddings = model(imgs)           # эмбеддинги
        logits = arcface(embeddings, labels)

        # 2. Вычисление лосса
        loss = F.cross_entropy(logits, labels)

        # 3. Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # вывод каждые 100 батчей
        if (batch_idx + 1) % 100 == 0:
            print(f"Эпоха {epoch+1}, Батч {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Эпоха {epoch+1} завершена. Средний Loss: {avg_loss:.4f}\n")

# ------------------------
# 5. Сохраняем модель
# ------------------------
torch.save(model.state_dict(), "face_arcface.pth")
print("Модель ArcFace сохранена как face_arcface.pth")

