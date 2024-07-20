from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np


mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embedding conversion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def recognize_faculty(img):
    img_cv = img.copy()
    # img_cv = cv2.resize(img_cv, (224, 224))
    boxes, probs = mtcnn.detect(img_cv)
    boxes, probs = mtcnn.detect(img)
    print(boxes)
    if boxes is not None:
        # Process each face
        faces = mtcnn.extract(img, boxes, save_path=None)
        face_embeddings = resnet(faces.unsqueeze(0)).detach().numpy()

        # Load saved embeddings and labels
        embeddings = np.load('faces_embeddings_teachers.npz')["arr_0"]
        #         print(embeddings)
        labels = np.load('faces_embeddings_teachers.npz')["arr_1"]

        # Calculate distances and find the closest match
        for i, face_embedding in enumerate(face_embeddings):
            distances = np.linalg.norm(embeddings - face_embedding, axis=1)
            min_distance_idx = np.argmin(distances)
            label = labels[min_distance_idx]

            return label
    else:
        return 'Intruder'


def recognize_student(img):
    img_cv = img.copy()
    # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img_cv = cv2.resize(img_cv, (224, 224))
    boxes, probs = mtcnn.detect(img_cv)
    boxes, probs = mtcnn.detect(img)
    print(boxes)
    if boxes is not None:
        # Process each face
        faces = mtcnn.extract(img, boxes, save_path=None)
        face_embeddings = resnet(faces.unsqueeze(0)).detach().numpy()

        # Load saved embeddings and labels
        embeddings = np.load('faces_embeddings_students.npz')["arr_0"]
        #         print(embeddings)
        labels = np.load('faces_embeddings_students.npz')["arr_1"]

        # Calculate distances and find the closest match
        for i, face_embedding in enumerate(face_embeddings):
            distances = np.linalg.norm(embeddings - face_embedding, axis=1)
            min_distance_idx = np.argmin(distances)
            label = labels[min_distance_idx]
            print(label)
            return label
    else:
        return 'Intruder'
