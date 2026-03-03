import gradio as gr
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ResUNetPlusPlus

# ==========================================
# 1. CẤU HÌNH VÀ TẢI MÔ HÌNH
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_PATH = "best_checkpoint.pth.tar"

print("Đang khởi động hệ thống AI...")
# Khởi tạo model và nạp trọng số
model = ResUNetPlusPlus(in_channels=3, num_classes=1).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)["state_dict"])
    model.eval()
    print("✅ Tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi tải mô hình: {e}. Vui lòng kiểm tra lại file best_checkpoint.pth.tar")

# ==========================================
# 2. HÀM XỬ LÝ ẢNH TỪ GIAO DIỆN WEB
# ==========================================
def predict_spine(image):
    """
    Hàm này nhận ảnh đầu vào từ người dùng (qua Web),
    đưa qua AI và trả về ảnh đã được tô màu đốt sống.
    """
    if image is None:
        return None

    # Gradio tự động đọc ảnh dưới dạng Numpy Array (RGB)
    original_h, original_w = image.shape[:2]

    # Resize ảnh về kích thước chuẩn để vẽ đè
    img_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Tiền xử lý để đưa vào AI
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    # AI Dự đoán
    with torch.no_grad():
        pred_tensor = model(input_tensor)
        pred_mask = (pred_tensor > 0.5).float().squeeze().cpu().numpy()

    # Tạo ảnh Overlay (Tô màu Đỏ tía rực rỡ lên vùng đốt sống)
    colored_mask = np.zeros_like(img_resized)
    # Tô màu đỏ (R=255, G=50, B=50)
    colored_mask[pred_mask == 1] = [255, 50, 50] 
    
    # Trộn ảnh gốc và ảnh mask (Alpha=0.5 để tạo độ trong suốt)
    output_image = cv2.addWeighted(img_resized, 0.8, colored_mask, 0.5, 0)

    # Phóng to trả lại kích thước gốc ban đầu của người dùng
    final_output = cv2.resize(output_image, (original_w, original_h))

    return final_output

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB (UI)
# ==========================================
# Cấu hình giao diện Gradio (Đã cập nhật chuẩn Gradio 6.0)
demo = gr.Interface(
    fn=predict_spine, # Hàm xử lý chính
    inputs=gr.Image(type="numpy", label="Tải ảnh X-quang cột sống cổ lên đây"), # Khung tải ảnh
    outputs=gr.Image(type="numpy", label="Kết quả AI chẩn đoán"), # Khung xuất ảnh
    title="🏥 HỆ THỐNG AI PHÂN ĐOẠN ĐỐT SỐNG CỔ",
    description="""
    **Đồ án Bài Tập Lớn - Môn Xử Lý Ảnh Y Tế**\n
    Hệ thống sử dụng mạng học sâu **ResUNet++** kết hợp hàm loss **Dice + BCE**. \n
    *Hướng dẫn: Kéo thả một bức ảnh X-quang vào ô bên trái và nhấn nút 'Submit' để AI tự động nhận diện và khoanh vùng các đốt sống.*
    """
)

# Chạy server
if __name__ == "__main__":
    print("🚀 Đang khởi tạo Giao diện Web...")
    demo.launch(inbrowser=True) # Tự động mở trình duyệt