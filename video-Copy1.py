import cv2

from ultralytics import YOLO

# Load the YOLOv8 model
model_file = "/home/nakahira/nakasone/runs/segment/train3/weights/best.pt"
#model_file = "/home/nakahira/workspace/coral/runs/segment/train9/weights/best.pt"
#model_file = "yolov8n.pt"
model = YOLO(model_file)

# Open the video file
video_path = "/home/nakahira/nakasone/video/NAICe5_WIN_20250519_09_39_43_Pro.mp4"
#video_path = "/home/nakahira/Downloads/video3.mp4"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.8)

        annotated_frame = results[0].plot()

        # 魚のカウント数を取得
        fish_count = len(results[0].boxes)

        #カウント数を表示するテキスト
        count_text = f"Fish count: {fish_count}"

        # テキストをフレームの左上に描画
        cv2.putText(annotated_frame,
                    count_text,
                    (10, 30),  # 表示位置 (左上のx, y座標)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,         # フォントスケール
                    (0, 0, 0), # 色 (B, G, R) 
                    2)         # 文字の太さ
        # print(results)
        # Visualize the results on the frame
        

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
