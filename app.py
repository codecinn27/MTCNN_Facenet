import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

def reading_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def Face_Detection_FaceNet(image, margin=0.2):
    model = MTCNN()
    detection = model.detect_faces(image)
    if detection:
        x, y, width, height = detection[0]['box']
        x_min = max(x - int(margin * width), 0)
        y_min = max(y - int(margin * height), 0)
        x_max = min(x + width + int(margin * width), image.shape[1])
        y_max = min(y + height + int(margin * height), image.shape[0])
        face = image[y_min:y_max, x_min:x_max]
        return cv2.resize(face, (160, 160))
    return None

def get_embedding(model, face):
    return model.embeddings([face])[0]

def is_match(known_embeddings, candidate_embedding, threshold=0.7):
    # known_embeddings is a list of embeddings for one person
    distances = [cosine(known_emb, candidate_embedding) for known_emb in known_embeddings]
    min_distance = min(distances) if distances else 1.0  # default large distance
    return min_distance, min_distance < threshold

# Load FaceNet model once
embedder = FaceNet()

# Path to dataset folder
dataset_path = 'img_dataset'

# Load dataset recursively
known_faces = {}
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue
    embeddings = []
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(person_dir, filename)
            img = reading_img(img_path)
            face = Face_Detection_FaceNet(img)
            if face is not None:
                emb = get_embedding(embedder, face)
                embeddings.append(emb)
            else:
                print(f"Warning: Face not detected in {img_path}")
    if embeddings:
        known_faces[person_name] = embeddings

# Webcam and real-time recognition
print("Starting webcam... Press 'q' to quit.")
detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    for det in detections:
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        face = rgb_frame[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160))
            face_embedding = get_embedding(embedder, face)

            name = "Unknown"
            min_distance = 1.0
            for person_name, embeddings in known_faces.items():
                dist, matched = is_match(embeddings, face_embedding)
                if matched and dist < min_distance:
                    min_distance = dist
                    name = person_name

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        except:
            pass  # skip any face errors

    cv2.imshow("Face Recognition (Webcam)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
