import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN ẢNH TEST
# ==========================================
TEST_IMG = r"d:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\images\0c114909059d7995f85958b07458374c.png"

def process_syllabus_pipeline(img_path):
    print("=> Đang chạy Pipeline Tổng hợp (Bản nâng cấp: Bám sát + Vẽ Contour)...")
    
    # Đọc ảnh
    img_stream = np.fromfile(img_path, np.uint8)
    img_bgr = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
    if img_bgr is None: raise ValueError("Không tìm thấy ảnh!")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape

    # ---------------------------------------------------------
    # BƯỚC 1: Tiền xử lý (Bilateral Filter giữ cạnh cực tốt)
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # ---------------------------------------------------------
    # BƯỚC 2: Trích xuất đặc trưng (Top-Hat Hình Elip)
    # ---------------------------------------------------------
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 35))
    img_tophat = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel_tophat)

    # ---------------------------------------------------------
    # BƯỚC 3: Tách ngưỡng và Morphology "Tự nhiên"
    # ---------------------------------------------------------
    _, img_thresh = cv2.threshold(img_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # ---------------------------------------------------------
    # BƯỚC 4: AUTO-ROI (Chặt rác xung quanh)
    # ---------------------------------------------------------
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
    img_morph = cv2.bitwise_and(img_morph, roi_mask)

    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(img_gray)
    MIN_AREA = img_w * img_h * 0.003 

    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # ---------------------------------------------------------
    # BƯỚC 5: Watershed Algorithm (Tinh chỉnh Nước)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # BƯỚC 6: Hậu xử lý (LCC + LÀM MŨM MĨM MASK)
    # ---------------------------------------------------------
    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(img_gray)
    
    if final_contours:
        final_contours = sorted(final_contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(final_contours[0]) 
        for cnt in final_contours:
            if cv2.contourArea(cnt) >= max_area * 0.10:
                cv2.drawContours(cleaned_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_expand, iterations=1)

    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (11, 11), 0)
    _, final_mask = cv2.threshold(cleaned_mask, 127, 255, cv2.THRESH_BINARY)

    # ---------------------------------------------------------
    # 7. TẠO ẢNH OVERLAY & VẼ CONTOUR (THAY ĐỔI: KHÔNG DÙNG HỘP BAO NỮA)
    # ---------------------------------------------------------
    colored_mask = np.zeros_like(img_bgr)
    colored_mask[final_mask == 255] = [0, 255, 0] # Tô xanh lá cây
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # Trộn mặt nạ xanh với ảnh gốc
    overlay_img = cv2.addWeighted(img_rgb, 1.0, colored_mask_rgb, 0.4, 0)
    
    # === VẼ CONTOUR (ĐƯỜNG VIỀN) MÀU VÀNG ===
    # Tìm lại viền của các vùng xanh lá trên final_mask
    box_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Thay vì vẽ hộp chữ nhật, ta dùng drawContours để vẽ viền
    cv2.drawContours(overlay_img, box_contours, -1, (255, 255, 0), 2)
            
    # Ghi text "Spine Boundary"
    # Tìm điểm cao nhất của contour đầu tiên để đặt text
    if len(box_contours) > 0:
        c = max(box_contours, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        cv2.putText(overlay_img, "Spine Boundary", (extTop[0], extTop[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # ==========================================
    # HIỂN THỊ TRỰC QUAN
    # ==========================================
    plt.figure(figsize=(20, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("1. Bilateral Filter + Top-Hat Elip")
    plt.imshow(img_tophat, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.title("2. Morphology Elip (Giữ dáng)")
    plt.imshow(img_morph, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.title("3. Sure BG & Sure FG (Lõi to hơn)")
    plt.imshow(unknown, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("4. Lab 12: Markers")
    plt.imshow(markers, cmap='jet')
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.title("5. Mask LCC + Làm mịn (Mũm mĩm)")
    plt.imshow(final_mask, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.title("6. Kết quả (Mask Xanh + Contour Vàng)")
    plt.imshow(overlay_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_syllabus_pipeline(TEST_IMG)