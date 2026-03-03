import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CervicalSpineDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Khởi tạo Dataset.
        images_dir: Đường dẫn đến thư mục chứa ảnh X-quang gốc
        masks_dir: Đường dẫn đến thư mục chứa ảnh Mask
        transform: Các phép biến đổi ảnh (Resize, Normalize, ToTensor...)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Chỉ lấy các file có đuôi là ảnh để tránh đọc nhầm file rác ẩn của Windows (như desktop.ini)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.images_list = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(valid_extensions)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # 1. Lấy tên file
        img_name = self.images_list[idx]
        
        # 2. Tạo đường dẫn đầy đủ đến ảnh
        img_path = os.path.join(self.images_dir, img_name)
        
        # Xử lý đường dẫn mask (Dự phòng trường hợp ảnh gốc .jpg nhưng mask lại lưu .png)
        mask_path = os.path.join(self.masks_dir, img_name) 
        if not os.path.exists(mask_path):
            # Nếu không tìm thấy mask cùng đuôi, thử tìm mask đuôi .png
            name_without_ext = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.masks_dir, name_without_ext + '.png')
        
        # 3. Đọc ảnh gốc bằng NumPy + OpenCV (Hỗ trợ tên file có dấu tiếng Việt)
        image_stream = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f" LỖI: Không thể đọc ảnh gốc tại: {img_path}. Hãy kiểm tra lại file này.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Đọc ảnh mask bằng NumPy + OpenCV
        mask_stream = np.fromfile(mask_path, np.uint8)
        mask = cv2.imdecode(mask_stream, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f" LỖI: Không thể đọc ảnh mask tại: {mask_path}. Hãy chắc chắn thư mục mask có file tương ứng.")
        
        # Chuyển đổi mask về nhị phân thuần túy: Nền = 0.0, Đốt sống = 1.0
        mask = (mask > 0).astype(np.float32) 
        
        # 5. Áp dụng các phép biến đổi (Resize, Normalize, ToTensor)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        # Đảm bảo mask là tensor và thêm 1 chiều channel (từ H, W thành 1, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        
        return image, mask

# --- KIỂM TRA THỬ CODE ---
if __name__ == "__main__":
    # Đường dẫn của bạn
    TRAIN_IMG_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_train\images"
    TRAIN_MASK_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_train\masks"
    
    # Kích thước ảnh muốn đưa vào model (ResUNet++ thường dùng 256x256 hoặc 512x512)
    IMAGE_SIZE = 256 

    # ĐỊNH NGHĨA TRANSFORM (BẮT BUỘC PHẢI CÓ A.Resize)
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE), # Đưa mọi ảnh về cùng 1 size
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    print("Đang khởi tạo Dataset...")
    try:
        # Khởi tạo tập dữ liệu
        train_dataset = CervicalSpineDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=transform)
        print(f" Thành công! Đã tìm thấy {len(train_dataset)} cặp ảnh và mask.")
        
        # Khởi tạo DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Lấy thử 1 batch ra xem kích thước
        print("Đang lấy thử 1 batch dữ liệu...")
        for images, masks in train_loader:
            print(" Kích thước mẻ ảnh gốc (Batch, Channel, Height, Width):", images.shape)
            print(" Kích thước mẻ mask tương ứng:", masks.shape)
            print("=> Mọi thứ hoạt động hoàn hảo! Dữ liệu đã sẵn sàng cho ResUNet++.")
            break # Chỉ in thử mẻ đầu tiên rồi dừng
            
    except Exception as e:
        print(f"\n ĐÃ XẢY RA LỖI: {e}")