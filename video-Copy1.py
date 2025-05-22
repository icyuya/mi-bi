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
# output_video_filename = "annotated_output_final_unique_fish_count.mp4" # 保存するファイル名

# VideoWriterの準備
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    print("Warning: FPS not found in video metadata. Using default FPS = 25.0")
    fps = 25.0

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

# print(f"Input video: {video_path}")
# print(f"Output video will be saved as: {output_video_filename}")
# print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

unique_fish_ids = set() # これまでに検出されたユニークな魚のIDを格納するセット

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # YOLOv8のトラッキング機能を使用
        results = model.track(frame, persist=True, conf=0.8, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot() # トラッキングIDも描画される場合がある

        areas_list = []

        if results[0].boxes is not None:
            # トラックIDを収集してユニークIDセットに追加
            if results[0].boxes.id is not None:
                detected_ids_in_frame = results[0].boxes.id.int().tolist()
                for fish_id in detected_ids_in_frame:
                    unique_fish_ids.add(fish_id)

            # マスクと面積の処理
            if results[0].masks is not None and results[0].masks.xy is not None:
                for i, mask_polygon_points in enumerate(results[0].masks.xy):
                    contour = mask_polygon_points.astype(np.int32)
                    area = cv2.contourArea(contour)
                    if area > 0:
                        areas_list.append(area)

        # 1. ★★★ ユニークな魚の総数をメインのカウントとして表示 ★★★
        unique_fish_count = len(unique_fish_ids)

        main_count_text = f"total fish: {unique_fish_count}"
        cv2.putText(annotated_frame, main_count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) # 色を緑に


        # 2. 面積リストを左上に表示 (開始位置を調整)
        start_x_areas = 10
        start_y_areas = 60  # 表示開始Y座標を調整
        line_height = 30   # 行の高さは維持 
        
        available_height = frame.shape[0] - start_y_areas - 10 
        max_items_on_screen = max(0, available_height // line_height)

        for idx in range(min(len(areas_list), max_items_on_screen)):
            current_y_position = start_y_areas + idx * line_height
            display_text = ""

            if idx == max_items_on_screen - 1 and len(areas_list) > max_items_on_screen:
                display_text = "..."
            else:
                if idx < len(areas_list):
                     display_text = f"F{idx+1}: {areas_list[idx]:.0f}px"
                else:
                    display_text = f"F{idx+1}: ---" # 万が一areas_listが短い場合

            cv2.putText(annotated_frame, display_text, (start_x_areas, current_y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # video_writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
# video_writer.release()
cv2.destroyAllWindows()

# print(f"Processed video saved successfully to: {output_video_filename}")