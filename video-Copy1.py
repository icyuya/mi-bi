import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model_file = "/home/nakahira/nakasone/runs/segment/train3/weights/best.pt"
model = YOLO(model_file)

# Open the video file
video_path = "/home/nakahira/nakasone/video/NAICe5_WIN_20250519_09_39_43_Pro.mp4"
cap = cv2.VideoCapture(video_path)

# 出力ビデオファイルの設定
output_video_filename = "/home/nakahira/nakasone/video/output/output_video.avi"

# VideoWriterの準備
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# FPSが取得できない場合のデフォルト値 (例: 25.0 や 30.0)
if fps == 0:
    print("Warning: FPS not found in video metadata. Using default FPS = 25.0")
    fps = 25.0


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

# print(f"Input video: {video_path}")
# print(f"Output video will be saved as: {output_video_filename}")
# print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, conf=0.8)
        annotated_frame = results[0].plot()

        fish_count = 0
        areas_list = []

        if results[0].boxes is not None:
            fish_count = len(results[0].boxes)
            if results[0].masks is not None and results[0].masks.xy is not None:
                for i, mask_polygon_points in enumerate(results[0].masks.xy):
                    contour = mask_polygon_points.astype(np.int32)
                    area = cv2.contourArea(contour)
                    if area > 0:
                        areas_list.append(area)

        # 1. 総カウント数を表示
        count_text = f"Fish count: {fish_count}"
        cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 2. 面積リストを左上に表示 (行間を調整)
        start_x_areas = 10
        start_y_areas = 60
        line_height = 30  
        
        available_height = frame.shape[0] - start_y_areas - 10 
        max_items_on_screen = max(0, available_height // line_height)

        for idx in range(min(len(areas_list), max_items_on_screen)):
            current_y_position = start_y_areas + idx * line_height
            display_text = ""

            if idx == max_items_on_screen - 1 and len(areas_list) > max_items_on_screen:
                display_text = "..."
            else:
                display_text = f"F{idx+1}: {areas_list[idx]:.0f}px²"

            cv2.putText(annotated_frame, display_text, (start_x_areas, current_y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2)

        # 処理済みフレームをビデオファイルに書き込む
        video_writer.write(annotated_frame)

        # 画面に表示
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# リソースを解放
cap.release()
video_writer.release() # VideoWriterを解放
cv2.destroyAllWindows()

print(f"Processed video saved successfully to: {output_video_filename}")