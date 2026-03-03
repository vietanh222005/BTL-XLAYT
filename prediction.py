import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Import kiến trúc model
from model import ResUNetPlusPlus

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_PATH = "best_checkpoint.pth.tar" # File lưu trọng số mô hình sau khi train

# ĐƯỜNG DẪN TỚI 1 BỨC ẢNH BẤT KỲ ĐỂ TEST (Có thể lấy trong tập train hoặc test)
# Bạn hãy thay tên file ảnh ở cuối cho đúng với file có trong máy bạn nhé
TEST_IMAGE_PATH = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_train\images\Hình 1.png"

# ==========================================
# 2. HÀM TẢI MÔ HÌNH
# ==========================================
def load_model(checkpoint_file, model):
    print("=> Đang tải mô hình đã huấn luyện...")
    # Load trọng số vào thiết bị tương ứng (CPU hoặc GPU)
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Chuyển model sang chế độ dự đoán (quan trọng)
    return model

# ==========================================
# 3. HÀM XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# ==========================================
def predict_and_plot(img_path, model):
    print(f"=> Đang xử lý ảnh: {img_path}")
    
    # 1. Đọc ảnh (Hỗ trợ đường dẫn tiếng Việt)
    image_stream = np.fromfile(img_path, np.uint8)
    image_bgr = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError("Không thể đọc được ảnh! Hãy kiểm tra lại đường dẫn.")
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Giữ lại ảnh gốc (resize về 256x256) để hiển thị
    original_image = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # 2. Các bước tiền xử lý giống y hệt lúc train
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Áp dụng biến đổi và thêm chiều Batch (1, 3, 256, 256)
    augmented = transform(image=image_rgb)
    x = augmented["image"].unsqueeze(0).to(DEVICE)

    # 3. Cho mô hình dự đoán
    print("=> Đang tiến hành phân đoạn đốt sống...")
    with torch.no_grad(): # Tắt tính toán gradient để tăng tốc
        prediction = model(x)
        
        # Output của model qua Sigmoid có giá trị từ 0 đến 1
        # Ta dùng ngưỡng 0.5: Điểm nào > 0.5 là màu trắng (đốt sống), < 0.5 là màu đen (nền)
        prediction = (prediction > 0.5).float()
        
    # Chuyển tensor từ GPU/CPU về dạng numpy array để vẽ ảnh
    predicted_mask = prediction.squeeze().cpu().numpy()

    # 4. Hiển thị kết quả bằng Matplotlib
    print("=> Đang hiển thị kết quả...")
    plt.figure(figsize=(10, 5))

    # Vẽ ảnh gốc
    plt.subplot(1, 2, 1)
    plt.title("Ảnh X-quang Gốc")
    plt.imshow(original_image)
    plt.axis("off")

    # Vẽ ảnh AI dự đoán
    plt.subplot(1, 2, 2)
    plt.title("AI Dự Đoán (Mask)")
    plt.imshow(predicted_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Khởi tạo kiến trúc mô hình trắng
    model = ResUNetPlusPlus(in_channels=3, num_classes=1).to(DEVICE)
    
    try:
        # Nạp "bộ não" đã train vào mô hình
        model = load_model(MODEL_PATH, model)
        
        # Dự đoán ảnh
        predict_and_plot(TEST_IMAGE_PATH, model)
    except FileNotFoundError:
         print(f" LỖI: Không tìm thấy file '{MODEL_PATH}'. Bạn đã bỏ comment đoạn code lưu mô hình trong file train.py và chạy lại chưa?")