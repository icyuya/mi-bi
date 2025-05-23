import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model_file = "/home/nakahira/nakasone/mi-bi/runs/segment/train3/weights/best.pt"
model = YOLO(model_file)

# Open the video file
video_path = "/home/nakahira/nakasone/mi-bi/video/NAICe5_WIN_20250519_09_39_43_Pro.mp4"
cap = cv2.VideoCapture(video_path)


output_video_filename = "/home/nakahira/nakasone/mi-bi/video/output/output.mp4"

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    print("Warning: FPS not found in video metadata. Using default FPS = 25.0")
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

# print(f"Input video: {video_path}")
# print(f"Output video will be saved as: {output_video_filename}") # コメントアウト
# print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS") # コメントアウト


# ユニークIDを格納するセット
unique_fish_ids_overall = set() # ★★★ 全体のユニークな魚のID ★★★
unique_fish_ids_cat1 = set()    # Area <= 20000
unique_fish_ids_cat2 = set()    # 20000 < Area <= 40000
unique_fish_ids_cat3 = set()    # Area > 40000

# Loop through the video frames
try:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # YOLOv8のトラッキング機能を使用
            results = model.track(frame, persist=True, conf=0.8, tracker="bytetrack.yaml")
            annotated_frame = results[0].plot()

            areas_list_current_frame = []

            # ボックスとIDが存在する場合
            if results[0].boxes is not None and results[0].boxes.id is not None:
                all_ids_tensor = results[0].boxes.id
                
                # ★★★ 全体のユニークIDを収集 ★★★
                all_tracked_ids_in_frame = all_ids_tensor.int().tolist()
                for an_id in all_tracked_ids_in_frame:
                    unique_fish_ids_overall.add(an_id)

                # マスクが存在する場合に面積計算とカテゴリ分類
                if results[0].masks is not None and results[0].masks.xy is not None:
                    all_masks_xy = results[0].masks.xy
                    
                    num_detections_with_ids = len(all_ids_tensor)
                    num_masks = len(all_masks_xy)

                    for i in range(min(num_detections_with_ids, num_masks)):
                        fish_id = all_ids_tensor[i].item()
                        mask_polygon_points = all_masks_xy[i]
                        
                        contour = mask_polygon_points.astype(np.int32)
                        area = cv2.contourArea(contour)

                        if area > 0:
                            areas_list_current_frame.append(area)

                            # 面積に基づいて魚をカテゴリ分類し、対応するセットにIDを追加
                            if area <= 20000:
                                unique_fish_ids_cat1.add(fish_id)
                            elif area <= 40000:
                                unique_fish_ids_cat2.add(fish_id)
                            else:
                                unique_fish_ids_cat3.add(fish_id)
            
            # 表示テキストの準備と描画
            y_offset = 30
            line_spacing = 35 # 各テキスト行間の基本スペース

            # 1. 全体のユニークな魚の総数を表示
            overall_unique_count = len(unique_fish_ids_overall)
            text_overall_unique = f"Total Fish: {overall_unique_count}"
            cv2.putText(annotated_frame, text_overall_unique, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2) # マゼンタ色
            y_offset += line_spacing

            # 2. サイズカテゴリ別のユニークな魚の数を表示
            count_cat1 = len(unique_fish_ids_cat1)
            count_cat2 = len(unique_fish_ids_cat2)
            count_cat3 = len(unique_fish_ids_cat3)

            text_cat1 = f"Small: {count_cat1}" 
            cv2.putText(annotated_frame, text_cat1, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += line_spacing

            text_cat2 = f"Medium: {count_cat2}"
            cv2.putText(annotated_frame, text_cat2, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += line_spacing

            text_cat3 = f"Large: {count_cat3}"
            cv2.putText(annotated_frame, text_cat3, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += line_spacing # 次のエリアリスト表示のためのオフセット

            # 3. （オプション）現在のフレームで検出された魚の面積リストを表示
            start_x_areas = 10
            start_y_areas = y_offset # カテゴリ別カウントの下に表示開始
            list_line_height = 35 # こちらは面積リスト用の行の高さ (フォントが小さいため)
            
            available_height = frame.shape[0] - start_y_areas - 10 
            max_items_on_screen = max(0, available_height // list_line_height)

            for idx in range(min(len(areas_list_current_frame), max_items_on_screen)):
                current_y_position = start_y_areas + idx * list_line_height
                display_text = ""

                if idx == max_items_on_screen - 1 and len(areas_list_current_frame) > max_items_on_screen:
                    display_text = "..."
                else:
                    if idx < len(areas_list_current_frame):
                        display_text = f"F{idx+1} Area: {areas_list_current_frame[idx]:.0f}px"
                    else:
                        display_text = f"F{idx+1} Area: ---" # Should not happen if loop is correct

                # cv2.putText(annotated_frame, display_text, (start_x_areas, current_y_position),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            video_writer.write(annotated_frame)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
finally:
    print("Releasing resources...")
    if cap.isOpened():
            cap.release()
    if 'video_writer' in locals() and video_writer.isOpened(): # video_writerが初期化され、開いているか確認
            video_writer.release()
    cv2.destroyAllWindows()
    print(f"Script finished. Attempted to save video to: {output_video_filename}")