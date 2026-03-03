import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os

from dataset import CervicalSpineDataset
from model import ResUNetPlusPlus

# ==========================================
# 1. CẤU HÌNH CÁC THÔNG SỐ 
# ==========================================
# ĐƯỜNG DẪN CỦA BẠN (Tập Train) - ĐÃ SỬA LẠI ĐÚNG CẤU TRÚC THƯ MỤC
TRAIN_IMG_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_train\images"
TRAIN_MASK_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_train\masks"

# ĐƯỜNG DẪN CỦA BẠN (Tập Test/Val) - ĐÃ SỬA LẠI ĐÚNG CẤU TRÚC THƯ MỤC
VAL_IMG_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\images"
VAL_MASK_DIR = r"D:\SUBJECT 2023-2027\XULYANHYTE\data_train_test\data_train_test\dataset_test\masks"

LEARNING_RATE = 1e-4      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 2            
NUM_EPOCHS = 50           # 50 vòng để AI học chi tiết các đốt sống
IMAGE_SIZE = 256          

# ==========================================
# 2. HÀM MẤT MÁT (BCE + DICE)
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # --- FIX LỖI CUDA (TRÀN LOGARIT) ---
        # Kẹp giá trị lại để không bao giờ chạm mức tuyệt đối 0.0 hoặc 1.0
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        # -----------------------------------
        
        bce_loss = nn.BCELoss()(inputs, targets)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return bce_loss + dice_loss

# ==========================================
# 3. HÀM HUẤN LUYỆN 1 VÒNG
# ==========================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    epoch_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            
        # Đưa ra ngoài autocast để tránh lỗi
        loss = loss_fn(predictions.float(), targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return epoch_loss / len(loader)

# ==========================================
# 4. HÀM KIỂM TRA & TRẢ VỀ ĐIỂM SỐ
# ==========================================
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    avg_dice = dice_score / len(loader)
    print(f"Độ chính xác Pixel: {num_correct/num_pixels*100:.2f}%")
    print(f"Điểm số Dice (Độ chồng lấp): {avg_dice:.4f}")
    
    model.train() 
    return avg_dice.item() # Trả về điểm Dice để so sánh

# ==========================================
# 5. HÀM CHÍNH (MAIN FUNCTION)
# ==========================================
def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    
    train_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Rotate(limit=15, p=0.5), 
        A.HorizontalFlip(p=0.5),   
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    model = ResUNetPlusPlus(in_channels=3, num_classes=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scaler = torch.amp.GradScaler('cuda') 

    print("Đang nạp dữ liệu...")
    train_ds = CervicalSpineDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_ds = CervicalSpineDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False) 

    # Biến lưu trữ điểm số tốt nhất
    best_dice_score = 0.0

    print(f"Bắt đầu huấn luyện {NUM_EPOCHS} vòng...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Lấy điểm số từ tập Validation (Test)
        current_dice = check_accuracy(val_loader, model, device=DEVICE)
        
        # CHỈ LƯU MÔ HÌNH NẾU NÓ TỐT HƠN VÒNG TRƯỚC
        if current_dice > best_dice_score:
            print(f"🌟 Điểm Dice tăng từ {best_dice_score:.4f} lên {current_dice:.4f}. Đang lưu mô hình tốt nhất...")
            best_dice_score = current_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # Lưu thành file riêng mang tên best_checkpoint
            torch.save(checkpoint, "best_checkpoint.pth.tar", _use_new_zipfile_serialization=False)
        else:
            print(f"Điểm Dice không tăng (Tốt nhất hiện tại: {best_dice_score:.4f}). Không lưu file.")

if __name__ == "__main__":
    main()