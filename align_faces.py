import cv2
import numpy as np
import torch
from models.landmark_model import LandmarkNet

# ------------------------------
# 1. Настройка устройства
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# ------------------------------
# 2. Загружаем модель
# ------------------------------
model = LandmarkNet().to(device)
model.load_state_dict(torch.load("landmark_model.pth", map_location=device))
model.eval()
print("Модель LandmarkNet загружена")

# ------------------------------
# 3. Функция выравнивания лица
# ------------------------------
def align_face(image_path, show=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (128, 128))

    tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255
    tensor = tensor.to(device)

    with torch.no_grad():
        landmarks = model(tensor)[0].cpu().numpy()

    # Выводим предсказанные ключевые точки
    print("Ключевые точки:", landmarks)

    # Выравниваем лицо по линии глаз
    left_eye = landmarks[0:2]
    right_eye = landmarks[2:4]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (64, 64)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, rot_mat, (128, 128))

    if show:
        # Конвертируем обратно в BGR для OpenCV и показываем
        cv2.imshow("Aligned Face", cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return aligned

# ------------------------------
# 4. Пример использования
# ------------------------------
if __name__ == "__main__":
    test_image = "data/img_align_celeba/000001.jpg"  # путь к тестовому изображению
    aligned_face = align_face(test_image, show=True)
