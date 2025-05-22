import cv2

from ultralytics import YOLO

# Load the YOLOv8 model
model_file = "/home/nakahira/workspace/nakahira/seavis/runs/segment/train49/weights/best.pt"
#model_file = "/home/nakahira/workspace/coral/runs/segment/train9/weights/best.pt"
#model_file = "yolov8n.pt"
model = YOLO(model_file)

# Open the video file
video_path = "/home/workspace/nakahira/real_white_coral_train_yolov8/GH011990_short.MP4"
#video_path = "/home/nakahira/Downloads/video3.mp4"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.8)
        #print(results)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()