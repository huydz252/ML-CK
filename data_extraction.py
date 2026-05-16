# 
# 
# 
# 
# CẢNH BÁO KO ĐƯỢC CHẠY FILE NÀY! (NẾU KO SẼ PHẢI TRẢ GIÁ 1 TIẾNG)
# 
# 
# 
# 


import os
import cv2
import csv
import numpy as np
import mediapipe as mp

# =====================================================================
# 1. KHỞI TẠO CẤU HÌNH MEDIAPIPE FACE MESH
# =====================================================================
mp_face_mesh = mp.solutions.face_mesh
# Khởi tạo mô hình xử lý 1 khuôn mặt, bật refine_landmarks để lấy 478 điểm chính xác
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Định nghĩa các chỉ số Landmark cố định của MediaPipe cho Mắt và Môi trong
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 81, 13, 311, 308, 317, 14, 87]

# =====================================================================
# 2. CÁC HÀM TOÁN HỌC TÍNH TOÁN CHỈ SỐ HÌNH HỌC (EAR & MAR)
# =====================================================================
def calculate_ear(landmarks, indices, w, h):
    # Chuyển đổi tọa độ chuẩn hóa của MediaPipe thành tọa độ pixel thực tế
    coords = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in indices]
    # Khoảng cách hình học theo chiều dọc
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    # Khoảng cách hình học theo chiều ngang
    C = np.linalg.norm(coords[0] - coords[3])
    # Công thức EAR truyền thống:
    # tử số: khoảng cách 2 mí (trên - dưới)
    # mẫu số: khoảng cách chiều ngang mắt (khóe - đuôi)
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, indices, w, h):
    coords = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in indices]
    # Khoảng cách chiều dọc của bờ môi trong
    A = np.linalg.norm(coords[1] - coords[7])
    B = np.linalg.norm(coords[2] - coords[6])
    C = np.linalg.norm(coords[3] - coords[5])
    # Khoảng cách chiều ngang
    D = np.linalg.norm(coords[0] - coords[4])
    # Công thức MAR
    return (A + B + C) / (2.0 * D)

# =====================================================================
# 3. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC VÀ ĐẦU RA CSV
# =====================================================================
# Thay tên này bằng tên thư mục gốc chứa fold_part1, fold_part2 của bạn
ROOT_DIR = "UTA_Data"  
OUTPUT_CSV = "rldd_geometry_features.csv"

# Khởi tạo file CSV và ghi dòng tiêu đề (Header)
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['EAR', 'MAR', 'Label'])

    print("=== BẮT ĐẦU TIẾN TRÌNH TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC ===")
    
    # os.walk sẽ tự động lội sâu vào tất cả các thư mục con của từng người
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            # Tách phần tên file và phần đuôi mở rộng (ví dụ: "0", ".mov")
            name_without_ext, ext = os.path.splitext(file)
            ext = ext.lower() # Chuyển đuôi về chữ thường để tránh lỗi .MOV hoặc .MP4

            # Chấp nhận cả 2 định dạng file video xuất hiện trong tập dữ liệu của bạn
            if ext in ['.mov', '.mp4']:
                video_path = os.path.join(root, file)
                
                # Tiến hành gán nhãn dựa trên tên file (bỏ đuôi)
                if name_without_ext == "0":
                    label = 0       # Alert (Tỉnh táo)
                    frame_step = 30 # Lấy mẫu thưa (1 giây/mẫu ở video 30fps) để tránh trùng lặp dữ liệu mở mắt
                elif name_without_ext == "5" or name_without_ext == "10":
                    label = 1       # Drowsy (Buồn ngủ / Mệt mỏi)
                    frame_step = 10 # Lấy mẫu dày hơn để bắt trọn khoảnh khắc nhắm mắt/ngáp ngắn
                else:
                    # Bỏ qua nếu có file rác hoặc file log khác trong thư mục
                    continue 

                print(f"Đang xử lý: {video_path} -> Gán nhãn: {label}")
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break # Hết video, chuyển sang file tiếp theo
                        
                    # Lấy mẫu theo bước nhảy khung hình cấu hình sẵn
                    if frame_count % frame_step == 0:
                        h, w, _ = frame.shape
                        # MediaPipe yêu cầu ảnh hệ màu RGB thay vì BGR mặc định của OpenCV
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        
                        # Nếu phát hiện thấy khuôn mặt trong khung hình
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0]
                            
                            # Tính toán các chỉ số hình học
                            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                            ear = (left_ear + right_ear) / 2.0 # Lấy trung bình cộng 2 mắt
                            mar = calculate_mar(landmarks, MOUTH_INNER, w, h)
                            
                            # ---------------------------------------------------------
                            # BỘ LỌC THÔNG MINH (HEURISTIC FILTER) CHO VIDEO BUỒN NGỦ
                            # ---------------------------------------------------------
                            if label == 1:
                                # Trong video buồn ngủ, ta chỉ giữ lại những khung hình tài xế
                                # sụp mí thật sự (EAR thấp) hoặc đang ngáp thật sự (MAR cao)
                                if ear <= 0.23 or mar >= 0.45:
                                    writer.writerow([round(ear, 4), round(mar, 4), label])
                            else:
                                # Đối với video tỉnh táo (Nhãn 0), lưu tất cả các mẫu lấy được
                                writer.writerow([round(ear, 4), round(mar, 4), label])
                                
                    frame_count += 1
                cap.release()

print(f"\n Quá trình xử lý hoàn tất thành công!")
print(f"File dữ liệu tổng hợp phục vụ huấn luyện đã lưu tại: {OUTPUT_CSV}")