import cv2

from ultralytics import YOLO

# Load the YOLOv8 model
model_file = "/home/nakahira/nakasone/runs/segment/train3/weights/best.pt"

model = YOLO(model_file)

# Open the video file
video_path = "/home/nakahira/nakasone/video/NAICe5_WIN_20250519_09_39_43_Pro.mp4"

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