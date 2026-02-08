import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
from models.facenet import FaceNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

def get_embedding(image_path, model_path):
    model = FaceNet().to(device)
    # Загружаем веса CE или ArcFace, игнорируя classifier при необходимости
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    tensor = torch.tensor(img.transpose(2,0,1)).unsqueeze(0).float()/255
    tensor = tensor.to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.cpu().numpy()

# ------------------------
# Пары изображений для сравнения
# ------------------------
img1 = "data/img_align_celeba/000001.jpg"
img2 = "data/img_align_celeba/000002.jpg"

# ------------------------
# Сравнение через CrossEntropy
# ------------------------
emb1_ce = get_embedding(img1, model_path="face_ce.pth")
emb2_ce = get_embedding(img2, model_path="face_ce.pth")
sim_ce = cosine_similarity(emb1_ce, emb2_ce)

# ------------------------
# Сравнение через ArcFace
# ------------------------
emb1_arc = get_embedding(img1, model_path="face_arcface.pth")
emb2_arc = get_embedding(img2, model_path="face_arcface.pth")
sim_arc = cosine_similarity(emb1_arc, emb2_arc)

# ------------------------
# Вывод результатов на русском
# ------------------------
print(f"Сходство (CE): {sim_ce[0][0]:.4f}")
print(f"Сходство (ArcFace): {sim_arc[0][0]:.4f}")

# Интерпретация
if sim_ce[0][0] > 0.8:
    print("CE: вероятно, это одно и то же лицо")
else:
    print("CE: разные лица")

if sim_arc[0][0] > 0.8:
    print("ArcFace: вероятно, это одно и то же лицо")
else:
    print("ArcFace: разные лица")
