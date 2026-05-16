import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# =====================================================================
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# =====================================================================
print("Đang tải tập dữ liệu đặc trưng hình học...")
# Đọc file CSV chứa chỉ số EAR, MAR nhóm đã trích xuất từ video UTA-RLDD
df = pd.read_csv('rldd_geometry_features.csv')

# Loại bỏ các dòng bị khuyết thiếu dữ liệu (nếu có khung hình MediaPipe không bắt được mặt)
df = df.dropna()

# Tách ma trận đặc trưng X (EAR, MAR) và vector nhãn y (0: Tỉnh táo, 1: Buồn ngủ)
X = df[['EAR', 'MAR']].values
y = df['Label'].values

print(f" Tổng số mẫu dữ liệu đọc được: {len(df)} dòng.")
print(f"Cơ cấu nhãn: \n- Nhãn 0 (Tỉnh táo): {np.sum(y == 0)} mẫu\n- Nhãn 1 (Buồn ngủ): {np.sum(y == 1)} mẫu")

# =====================================================================
# 2. CHIA TẬP DỮ LIỆU & CHUẨN HÓA (FEATURE SCALING)
# =====================================================================
# Chia dữ liệu: 80% để học (Train) và 20% để kiểm tra độ chính xác độc lập (Test)
# Tham số stratify=y giúp tỷ lệ nhãn 0 và 1 ở 2 tập luôn cân bằng nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SVM cực kỳ nhạy cảm với khoảng cách. Do EAR (0.1 - 0.4) và MAR (0.1 - 0.8) có khoảng giá trị khác nhau,
# ta cần dùng StandardScaler để đưa chúng về cùng phân phối chuẩn (Gốc tọa độ 0, phương sai 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 3. HUẤN LUYỆN MÔ HÌNH SVM
# =====================================================================
print("\n[INFO] Đang tiến hành huấn luyện mô hình SVM với RBF Kernel...")
# Khởi tạo thuật toán SVM, dùng kernel 'rbf' để phân tách ranh giới phi tuyến phức tạp
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
print(" Huấn luyện hoàn tất!")

# =====================================================================
# 4. ĐÁNH GIÁ MÔ HÌNH (LẤY SỐ LIỆU CHO BÁO CÁO ĐỒ ÁN)
# =====================================================================
y_pred = model.predict(X_test_scaled)

print("\n================ KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ================")
# Xuất bảng báo cáo độ chính xác chi tiết (Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred, target_names=['Tinh tao (0)', 'Buon ngu (1)']))

print("Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

# =====================================================================
# 5. XUẤT FILE .PKL ĐỂ LƯU TRỮ MÔ HÌNH VẬT LÝ
# =====================================================================
print("\n[INFO] Đang đóng gói và lưu mô hình xuống ổ cứng...")
# Lưu mô hình toán học SVM
with open('drowsiness_svm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Lưu cả bộ chuẩn hóa Scaler để khi chạy camera thực tế cũng scale dữ liệu theo tỷ lệ tương tự
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(" Thành công! Đã sinh ra file 'drowsiness_svm_model.pkl' và 'scaler.pkl' trong thư mục.")