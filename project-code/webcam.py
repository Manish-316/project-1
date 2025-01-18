from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Class names for COCO dataset
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# Function to initialize webcam
def initialize_webcam():
    cap = cv2.VideoCapture(0)  # Use webcam
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)  # Set height
    return cap


# Function to initialize video
def initialize_video(video_path):
    if os.path.exists(video_path):  # Check if video file exists
        cap = cv2.VideoCapture(video_path)  # Open video file
        return cap
    else:
        print(f"Error: Video file '{video_path}' not found!")
        return None


# Load the YOLOv8 model
def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


# Process a frame for object detection
def process_frame(img, model):
    results = model(img, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates as integers
            conf = math.ceil(box.conf[0] * 100) / 100  # Confidence score

            # Draw bounding box with confidence score
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            label = f'{CLASS_NAMES[int(box.cls[0])]} {conf * 100:.2f}%'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    return img


# Function for webcam object detection loop
def webcam_detection():
    cap = initialize_webcam()  # Initialize webcam
    model = load_model()  # Load YOLOv8 model

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Process the current frame for object detection
        img = process_frame(img, model)

        # Display the result
        cv2.imshow("Object Detection (Webcam)", img)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Function for video object detection loop
def video_detection(video_path):
    cap = initialize_video(video_path)  # Initialize video file
    if not cap:
        return  # Exit if video file is not found

    model = load_model()  # Load YOLOv8 model

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from video")
            break

        # Process the current frame for object detection
        img = process_frame(img, model)

        # Display the result
        cv2.imshow("Object Detection (Video)", img)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Main function to switch between webcam and video detection
def main():
    # Uncomment one of the following functions depending on the input type

    # For Webcam Input:
    webcam_detection()  # Uncomment to use webcam

    # For Video Input:
    # video_detection("path/to/your/video.mp4")  # Uncomment to use a video file (replace with your video path)


if __name__ == "__main__":
    main()
