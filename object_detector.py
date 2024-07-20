from ultralytics import YOLO
import torch
import cv2
import cvzone
import numpy as np
import os
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
import math
import check_valid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def video(salience=False):
    # This function captures video from a webcam or a video file and performs object detection on the frames.

    # For capturing video from the webcam
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 1920)  # Set width
    # cap.set(4, 1080)  # Set height

    # For capturing video from a file
    cap = cv2.VideoCapture(r"Videos for testing/video3.mp4")

    return detect(cap, salience)


def detect(cap, salience=False):
    # Infinite loop to process each frame from the video stream
    while True:
        # Capture the current frame from the video source
        success, img = cap.read()  # 'success' indicates if a frame was successfully captured

        if success:
            image = cv2.resize(img, (640, 640))
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
                    label = "intruder"
                    if class_names[cls] == 'student':
                        label = check_valid.recognize_student(cropped_img)
                    if class_names[cls] == 'faculty':
                        label = check_valid.recognize_faculty(cropped_img)
                    # # Display the object's class name and confidence level on the frame
                    cvzone.putTextRect(image, f'{label} {conf}', (max(0, x1), max(30, y1)))

                    # Draw a rectangle around the detected object
                    cvzone.cornerRect(img, (x1, y1, w, h))

            # Display the processed frame with object detection
            if salience:
                rgb_img = cv2.resize(image, (640, 640))
                img = cv2.resize(img, (640, 640))
                sal_img = saliency(rgb_img)
                final_image = cv2.hconcat([img, sal_img])
            else:
                final_image = img
                final_image = cv2.resize(final_image, (2560, 1600))
            cv2.imshow("Image", final_image)
            # Wait for a key press to prevent the video from closing immediately
            cv2.waitKey(1)
        else:
            break  # Break the loop if there are no more frames or an error occurs


def saliency(rgb_img):
    image = np.float32(rgb_img) / 255
    # target_layers = [model.model.model[-2], model.model.model[-3], model.model.model[-4]]
    target_layers =[model.model.model[-4]]
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
                label = "intruder"
                if class_names[cls] == 'student':
                    label = check_valid.recognize_student(cropped_img)
                if class_names[cls] == 'faculty':
                    label = check_valid.recognize_faculty(cropped_img)
                # # Display the object's class name and confidence level on the frame
                cvzone.putTextRect(image, f'{label} {conf}', (max(0, x1), max(30, y1)))
                # Draw a rectangle around the detected object
                cvzone.cornerRect(image, (x1, y1, w, h))

        if salience:
            rgb_img = cv2.resize(img, (640, 640))
            sal_img = saliency(rgb_img)
            final_image = cv2.hconcat([image, sal_img])
        else:
            final_image = image
        # Display the processed frame with object detection

        cv2.imshow("Image", final_image)

        # Wait for a key press to prevent the video from closing immediately
        cv2.waitKey(0)
        cv2.destroyAllWindows()


model = YOLO('best.pt')
class_names = ["intruder", "student", "faculty"]
path = r"./img for human intrusion/train/images/"


# path = r"images-for-testing/"
# print(model.eval())

predict_for_folder(path, salience=True)
video()
