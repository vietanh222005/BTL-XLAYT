import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN ẢNH TEST
# ==========================================
TEST_IMG = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\images\0c114909059d7995f85958b07458374c.png"

# ==========================================
# CÁC HÀM XỬ LÝ THEO ĐÚNG QUY TRÌNH
# ==========================================

def preprocess_image(img_gray):
    """BƯỚC 1: Tiền xử lý (CLAHE + Bilateral Filter)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)
    return img_blur

def morphology_extraction(img_blur):
    """BƯỚC 2 & 3: Top-Hat Hình Elip và Tách ngưỡng + Toán học hình thái"""
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 35))
    img_tophat = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel_tophat)

    _, img_thresh = cv2.threshold(img_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return img_morph

def apply_auto_roi(img_morph, img_gray):
    """BƯỚC 4: Tự động cắt rác (Auto-ROI) dựa trên trục cột sống"""
    img_h, img_w = img_gray.shape
    col_sums = np.sum(img_morph[int(img_h*0.2):int(img_h*0.8), :], axis=0)
    mid_start, mid_end = int(img_w * 0.2), int(img_w * 0.8)
    spine_x = np.argmax(col_sums[mid_start:mid_end]) + mid_start

    roi_mask = np.zeros_like(img_gray)
    roi_left = int(img_w * 0.15)  
    roi_right = int(img_w * 0.18) 
    roi_top = int(img_h * 0.10)
    roi_bottom = int(img_h * 0.92)
    
    safe_x1 = max(0, spine_x - roi_left)
    safe_x2 = min(img_w, spine_x + roi_right)
    cv2.rectangle(roi_mask, (safe_x1, roi_top), (safe_x2, roi_bottom), 255, -1)
    
    return cv2.bitwise_and(img_morph, roi_mask)

def apply_watershed(img_bgr, img_morph, img_gray):
    """BƯỚC 5: Thuật toán Phân thủy (Watershed)"""
    img_h, img_w = img_gray.shape
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(img_gray)
    MIN_AREA = img_w * img_h * 0.003 

    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    kernel_bg = np.ones((15,15), np.uint8)
    sure_bg = cv2.dilate(filtered_mask, kernel_bg, iterations=3)

    kernel_fg = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(filtered_mask, kernel_fg, iterations=2)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_watershed = img_bgr.copy()
    markers = cv2.watershed(img_watershed, markers)
    
    final_mask = np.zeros_like(img_gray)
    final_mask[markers > 1] = 255
    return final_mask, unknown, markers

def post_process_mask(final_mask, img_gray):
    """BƯỚC 6: Hậu xử lý (LCC + Làm mịn Mũm mĩm)"""
    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(img_gray)
    
    if final_contours:
        final_contours = sorted(final_contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(final_contours[0]) 
        for cnt in final_contours:
            # Chỉ lấy các mảng xương có diện tích >= 10% mảng lớn nhất
            if cv2.contourArea(cnt) >= max_area * 0.10:
                cv2.drawContours(cleaned_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Làm mũm mĩm và bo góc
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_expand, iterations=1)
    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (11, 11), 0)
    _, cleaned_mask = cv2.threshold(cleaned_mask, 127, 255, cv2.THRESH_BINARY)
    
    return cleaned_mask

def create_overlay(img_bgr, final_mask):
    """BƯỚC 7: Trộn ảnh và vẽ đường viền (Contour)"""
    colored_mask = np.zeros_like(img_bgr)
    colored_mask[final_mask == 255] = [0, 255, 0] # Xanh lá
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_rgb, 1.0, colored_mask_rgb, 0.4, 0)
    
    # Vẽ Contour Vàng
    box_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, box_contours, -1, (255, 255, 0), 2)
            
    if len(box_contours) > 0:
        c = max(box_contours, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        cv2.putText(overlay, "Spine Boundary", (extTop[0], extTop[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    
    return overlay

# ==========================================
# HÀM CHẠY CHÍNH (MAIN PIPELINE)
# ==========================================
def process_syllabus_pipeline(img_path):
    print("=> Đang chạy Pipeline Truyền thống (Bản Chuẩn hóa Cuối cùng)...")
    
    # 0. Đọc ảnh
    img_stream = np.fromfile(img_path, np.uint8)
    img_bgr = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
    if img_bgr is None: raise ValueError("Lỗi: Không tìm thấy ảnh!")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Thực thi tuần tự các bước
    img_blur = preprocess_image(img_gray)
    img_morph_raw = morphology_extraction(img_blur)
    img_roi = apply_auto_roi(img_morph_raw, img_gray)
    watershed_mask, unknown, markers = apply_watershed(img_bgr, img_roi, img_gray)
    final_mask = post_process_mask(watershed_mask, img_gray)
    overlay_img = create_overlay(img_bgr, final_mask)

    # Hiển thị trực quan 6 bước
    plt.figure(figsize=(20, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("1. Tiền xử lý (CLAHE + Bilateral)")
    plt.imshow(img_blur, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.title("2. Top-Hat + Morphology (Hình thái học)")
    plt.imshow(img_morph_raw, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.title("3. Vùng tranh chấp (Unknown - Watershed)")
    plt.imshow(unknown, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("4. Đánh dấu (Markers - Watershed)")
    plt.imshow(markers, cmap='jet')
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.title("5. Hậu xử lý LCC (Sạch & Mũm mĩm)")
    plt.imshow(final_mask, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.title("6. Kết quả Cuối cùng (Overlay + Contour)")
    plt.imshow(overlay_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_syllabus_pipeline(TEST_IMG)
