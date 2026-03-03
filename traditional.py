import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN ẢNH TEST
# ==========================================
# Thay đường dẫn này bằng 1 ảnh X-quang bất kỳ trong tập test của bạn
TEST_IMG = r"d:\SUBJECT 2023-2027\XULYANHYTE\group 04\group 04\4a8fc0dbc13a4e9ae280a069e54a9121.png"

def process_traditional(img_path):
    print("=> Đang đọc và xử lý ảnh bằng phương pháp truyền thống nâng cao...")
    
    # 1. Đọc ảnh và chuyển về ảnh xám (Grayscale)
    img_stream = np.fromfile(img_path, np.uint8)
    img_bgr = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Không tìm thấy ảnh, hãy kiểm tra lại đường dẫn!")
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape
    
    # 2. Tăng cường độ tương phản (CLAHE) để làm rõ xương
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # 3. Lọc nhiễu mạnh (Median Blur) giúp làm mịn khối xương hơn
    img_blur = cv2.medianBlur(img_clahe, 7)
    
    # 4. Phân ngưỡng tự động (Otsu)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Phép toán Hình thái học (Morphology)
    # Dùng kernel hình elip dọc vì đốt sống thường xếp dọc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 6. TÌM VÀ LỌC ĐỐT SỐNG BẰNG HEURISTIC (LUẬT SUY NGHIỆM)
    contours, _ = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo mặt nạ đen trắng (giống đầu ra của AI)
    final_mask = np.zeros_like(img_gray)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Áp dụng các bộ lọc "mẹo" để loại bỏ hộp sọ, cằm và nhiễu:
        # Điều kiện 1: Diện tích không quá nhỏ (nhiễu) và không quá to (hộp sọ/phổi)
        if 200 < area < 8000:
            # Điều kiện 2: Trục Y (Loại bỏ 25% phía trên cùng vì thường là hộp sọ/cằm)
            if y > img_h * 0.25:
                # Điều kiện 3: Trục X (Cột sống thường nằm ở dải giữa ảnh)
                if img_w * 0.2 < x < img_w * 0.8:
                    
                    # TÔ ĐẶC MÀU TRẮNG VÀO MẶT NẠ ĐEN (Giống Mask của AI)
                    cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 7. TẠO ẢNH OVERLAY (Chồng mask đỏ lên ảnh gốc) để dễ so sánh
    colored_mask = np.zeros_like(img_bgr)
    colored_mask[final_mask == 255] = [0, 0, 255] # Màu đỏ (BGR)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # Trộn ảnh
    overlay_img = cv2.addWeighted(img_rgb, 1.0, colored_mask_rgb, 0.4, 0)

    # ==========================================
    # HIỂN THỊ CÁC BƯỚC SO SÁNH
    # ==========================================
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 4, 1)
    plt.title("1. Ảnh Gốc (Grayscale)")
    plt.imshow(img_gray, cmap='gray')
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.title("2. Phân ngưỡng (Otsu)")
    plt.imshow(img_thresh, cmap='gray')
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.title("3. Mask Truyền thống\n(Lọc theo diện tích & tọa độ)")
    plt.imshow(final_mask, cmap='gray')
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.title("4. Overlay\n(Phương pháp truyền thống)")
    plt.imshow(overlay_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_traditional(TEST_IMG)