import os
import cv2
import csv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,  #468->478
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 81, 13, 311, 308, 317, 14, 87]

#tỉ lệ khung hình mắt
def calculate_ear(landmarks, indices, w, h):
    coords = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in indices]
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    return (A + B) / (2.0 * C)


def calculate_mar(landmarks, indices, w, h):
    coords = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in indices]
    A = np.linalg.norm(coords[1] - coords[7])
    B = np.linalg.norm(coords[2] - coords[6])
    C = np.linalg.norm(coords[3] - coords[5])
    D = np.linalg.norm(coords[0] - coords[4])
    return (A + B + C) / (2.0 * D)




# ----------------------------------------------------------------
ROOT_DIR = "UTA_Data"  
OUTPUT_CSV = "rldd_geometry_features.csv"

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['EAR', 'MAR', 'Label'])

    print("=== BẮT ĐẦU TIẾN TRÌNH TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC ===")
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            name_without_ext, ext = os.path.splitext(file)
            ext = ext.lower() 

            if ext in ['.mov', '.mp4']:
                video_path = os.path.join(root, file)
                if name_without_ext == "0":
                    label = 0     
                    frame_step = 30 # Lấy mẫu thưa - tránh trùng lặp dữ liệu mở mắt
                elif name_without_ext == "5" or name_without_ext == "10":
                    label = 1   
                    frame_step = 10 # Lấy mẫu dày hơn để bắt trọn khoảnh khắc nhắm mắt/ngáp ngắn
                else:
                    continue 

                print(f"Đang xử lý: {video_path} -> Gán nhãn: {label}")
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    if frame_count % frame_step == 0:
                        h, w, _ = frame.shape
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0]
                            
                            # Tính toán các chỉ số hình học
                            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                            ear = (left_ear + right_ear) / 2.0 
                            mar = calculate_mar(landmarks, MOUTH_INNER, w, h)
                            
                            if label == 1:
                                # Trong video buồn ngủ, ta chỉ giữ lại những khung hình tài xế
                                # sụp mí thật sự (EAR thấp) hoặc đang ngáp thật sự (MAR cao)
                                if ear <= 0.23 or mar >= 0.6:
                                    writer.writerow([round(ear, 4), round(mar, 4), label])
                            else:
                                writer.writerow([round(ear, 4), round(mar, 4), label])
                                
                    frame_count += 1
                cap.release()

print(f"\n Quá trình xử lý hoàn tất thành công!")
print(f"File dữ liệu tổng hợp phục vụ huấn luyện đã lưu tại: {OUTPUT_CSV}")