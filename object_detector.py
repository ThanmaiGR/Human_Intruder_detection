from ultralytics import YOLO
import torch
import cv2
import cvzone
import numpy as np
import os
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import math
import check_valid
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def video():
    # This function captures video from a webcam or a video file and performs object detection on the frames.

    # For capturing video from the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # For capturing video from a file
    # cap = cv2.VideoCapture(r"")
    # cap = cv2.VideoCapture(r"")

    return detect(cap)


def detect(cap):
    # Infinite loop to process each frame from the video stream
    while True:
        # Capture the current frame from the video source
        success, img = cap.read()  # 'success' indicates if a frame was successfully captured

        if success:
            # Perform object detection on the captured frame
            results = model(img, stream=True)  # 'results' contains detected objects' information

            # Iterate through detected objects
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract box coordinates and dimensions
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cropped_img = img[y1:y2, x1:x2]

                    # Extract confidence (accuracy) and class label
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    # Display the object's class name and confidence level on the frame
                    label = "Intruder"
                    if class_names[cls] == 'known_person':
                        label = check_valid.recognize_student(cropped_img)
                    if label is None:
                        label = check_valid.recognize_faculty(cropped_img)
                    # # Display the object's class name and confidence level on the frame
                    cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(30, y1)))

                    # Draw a rectangle around the detected object
                    cvzone.cornerRect(img, (x1, y1, w, h))

            img = cv2.resize(img, (640, 640))
            # Display the processed frame with object detection
            cv2.imshow("Image", img)

            # Wait for a key press to prevent the video from closing immediately
            cv2.waitKey(1)
        else:
            break  # Break the loop if there are no more frames or an error occurs


def saliency(rgb_img):
    image = np.float32(rgb_img) / 255
    # target_layers = [model.model.model[-2], model.model.model[-3], model.model.model[-4]]
    target_layers =[model.model.model[-2]]
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=False)
    # plt.imshow(cam_image)
    # plt.show()
    return cam_image


def predict_for_folder(folder_path, salience=False):
    # Infinite loop to process each frame from the video stream
    for image_name in os.listdir(folder_path):
        image = cv2.imread(folder_path + image_name)
        image = cv2.resize(image, (640, 640))
        img = image.copy()
        # Perform object detection on the captured frame
        results = model(image) # 'results' contains detected objects' information
        # Iterate through detected objects
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract box coordinates and dimensions
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cropped_img = img[y1:y2, x1:x2]
                # Extract confidence (accuracy) and class label
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = class_names[cls]
                # label = "Intruder"
                # if class_names[cls] == 'Intruder':
                #     label = check_valid.recognize_student(cropped_img)
                # if label is None:
                #     label = check_valid.recognize_faculty(cropped_img)
                # # Display the object's class name and confidence level on the frame
                cvzone.putTextRect(image, f'{label} {conf}', (max(0, x1), max(30, y1)))
                # Draw a rectangle around the detected object
                cvzone.cornerRect(image, (x1, y1, w, h))

        if salience:
            rgb_img = cv2.resize(img, (640, 640))
            sal_img = saliency(rgb_img)
            final_image = cv2.hconcat([img, sal_img])
        else:
            final_image = image
        # Display the processed frame with object detection

        cv2.imshow(f"{image_name}", final_image)

        # Wait for a key press to prevent the video from closing immediately
        cv2.waitKey(0)
        cv2.destroyAllWindows()


model = YOLO('updated_best.pt')
class_names = ["Intruder", "known_person"]
path = r"img for human intrusion/test/images/"

# path = r"images-for-testing/"
# print(model.eval())

predict_for_folder(path)
# video()
