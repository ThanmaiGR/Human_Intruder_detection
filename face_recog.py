from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2
import os

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def video():
    # This function captures video from a webcam or a video file and performs object detection on the frames.

    # For capturing video from the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    # For capturing video from a file
    # cap = cv2.VideoCapture(r"E:\vending machine\AC2 Corridors_ch13_20240710130005_20240710140005.mp4")

    return detect(cap)


def detect(cap):

    # Load saved embeddings and labels
    embeddings = np.load('faces_embeddings.npz')["arr_0"]
    labels = np.load('faces_embeddings.npz')["arr_1"]
    # Infinite loop to process each frame from the video stream
    while True:
        # Capture the current frame from the video source
        success, img = cap.read()  # 'success' indicates if a frame was successfully captured

        if success:
            img_cv = img.copy()
            # Perform object detection on the captured frame
            boxes, probs = mtcnn.detect(img)
            if boxes is not None:
            # Process each face
                for bbox in boxes:
                    # print(box.shape)
                    bbox = np.expand_dims(bbox, axis=0)
                    # print(box.shape, boxes.shape)
                    faces = mtcnn.extract(img, bbox, save_path=None)
                    face_embeddings = resnet(faces.unsqueeze(0)).detach().numpy()

                    # Calculate distances and find the closest match
                    # print(boxes.shape, face_embeddings.shape)
                    for i, face_embedding in enumerate(face_embeddings):
                        # continue
                        distances = np.linalg.norm(embeddings - face_embedding, axis=1)
                        min_distance_idx = np.argmin(distances)
                        label = labels[min_distance_idx]
                        distance = distances[min_distance_idx]

                        # Draw bounding box
                        box = bbox[i].astype(int)
                        cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                        # Draw label
                        label_text = f'{label}: {distance:.2f}'
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img_cv, (box[0], box[1] - text_height - baseline), (box[0] + text_width, box[1]),
                                      (0, 0, 255), thickness=cv2.FILLED)
                        cv2.putText(img_cv, label_text, (box[0], box[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1, lineType=cv2.LINE_AA)
                        print(label_text)
                else:
                    print("No face detected in the image.")
            cv2.imshow('Output with Boxes and Labels', img_cv)
            cv2.waitKey(1)
        else:
            break  # Break the loop if there are no more frames or an error occurs


def predict_for_folder(folder_path):
    # Load saved embeddings and labels
    embeddings = np.load('faces_embeddings.npz')["arr_0"]
    labels = np.load('faces_embeddings.npz')["arr_1"]
    for image_name in os.listdir(folder_path):
        image = cv2.imread(folder_path + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (640, 640))
        img = image.copy()
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # img_cv = cv2.resize(img_cv, (224, 224))
        boxes, probs = mtcnn.detect(img_cv)
        boxes, probs = mtcnn.detect(img)

        if boxes is not None:
            # Process each face
            for bbox in boxes:
                # print(box.shape)
                bbox = np.expand_dims(bbox, axis=0)
                # print(box.shape, boxes.shape)
                faces = mtcnn.extract(img, bbox, save_path=None)
                face_embeddings = resnet(faces.unsqueeze(0)).detach().numpy()

                # Calculate distances and find the closest match
                # print(boxes.shape, face_embeddings.shape)
                for i, face_embedding in enumerate(face_embeddings):
                    # continue
                    distances = np.linalg.norm(embeddings - face_embedding, axis=1)
                    min_distance_idx = np.argmin(distances)
                    label = labels[min_distance_idx]
                    distance = distances[min_distance_idx]
                    if distance < 0.9:
                        # Draw bounding box
                        box = bbox[i].astype(int)
                        cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                        # Draw label
                        label_text = f'{label}: {distance:.2f}'
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img_cv, (box[0], box[1] - text_height - baseline), (box[0] + text_width, box[1]),
                                      (0, 0, 255), thickness=cv2.FILLED)
                        cv2.putText(img_cv, label_text, (box[0], box[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1, lineType=cv2.LINE_AA)
                        print(label_text)
            else:
                print("No face detected in the image.")
        cv2.imshow('Output with Boxes and Labels', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



path = r"img for human intrusion/test/images/"
# print(model.eval())

predict_for_folder(path)
# video()
