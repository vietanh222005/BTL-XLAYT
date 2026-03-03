import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ResUNetPlusPlus

# ==========================================
# CẤU HÌNH THÔNG SỐ
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_PATH = "best_checkpoint.pth.tar"

# SỬA LẠI ĐƯỜNG DẪN ẢNH TEST CHO ĐÚNG VỚI MÁY CỦA BẠN (Thêm 1 lớp data_train_test nữa)
TEST_IMG = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\images\0a6e81ba1d360d6a84e3bc83bfdd8c05.png"
TEST_MASK = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\masks\0a6e81ba1d360d6a84e3bc83bfdd8c05.png"

def calculate_dice(pred, target):
    """Tính điểm Dice (Độ chồng lấp) giữa dự đoán và thực tế"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """Chồng mask màu lên ảnh gốc"""
    # Tạo một bức ảnh màu có cùng kích thước
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color # Tô màu (Mặc định là màu Đỏ: R=255, G=0, B=0)
    
    # Trộn ảnh gốc và ảnh mask lại với nhau theo tỷ lệ alpha
    overlaid = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return overlaid

def main():
    print("=> Đang tải mô hình...")
    model = ResUNetPlusPlus(in_channels=3, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)["state_dict"])
    model.eval()

    # 1. Đọc ảnh gốc và ảnh mask thực tế (Ground Truth)
    img_stream = np.fromfile(TEST_IMG, np.uint8)
    img_bgr = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    mask_stream = np.fromfile(TEST_MASK, np.uint8)
    true_mask = cv2.imdecode(mask_stream, cv2.IMREAD_GRAYSCALE)
    true_mask = cv2.resize(true_mask, (IMAGE_SIZE, IMAGE_SIZE))
    true_mask = (true_mask > 0).astype(np.float32) # Nhị phân hóa mask gốc

    # Resize ảnh gốc để hiển thị
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # 2. Tiền xử lý ảnh gốc để đưa vào AI
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # 3. AI Dự đoán
    with torch.no_grad():
        pred_tensor = model(input_tensor)
        pred_mask = (pred_tensor > 0.5).float().squeeze().cpu().numpy()

    # 4. Tính toán điểm số
    dice_score = calculate_dice(pred_mask, true_mask)
    print(f"\n✅ KẾT QUẢ ĐÁNH GIÁ:")
    print(f"- Độ chính xác (Dice Score): {dice_score * 100:.2f}%")

    # 5. Tạo ảnh Overlay (Chồng lớp)
    # Ảnh 1: AI dự đoán (Màu Đỏ) đè lên ảnh gốc
    ai_overlay = overlay_mask(img_resized, pred_mask, color=(255, 0, 0), alpha=0.6)
    
    # Ảnh 2: Bác sĩ gán nhãn (Màu Xanh lá) đè lên ảnh gốc để đối chiếu
    doctor_overlay = overlay_mask(img_resized, true_mask, color=(0, 255, 0), alpha=0.6)

    # 6. Hiển thị
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Ảnh X-quang Gốc")
    plt.imshow(img_resized)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"AI Khoanh Vùng\nĐộ chính xác: {dice_score*100:.1f}%")
    plt.imshow(ai_overlay)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Thực tế (Ground Truth)")
    plt.imshow(doctor_overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()