import torch
import torch.nn as nn

class SqueezeAndExcitation(nn.Module):
    """
    Khối Squeeze-and-Excitation (SE Block)
    Giúp mô hình tập trung vào các kênh đặc trưng quan trọng nhất
    (ví dụ: tự động làm nổi bật các kênh chứa thông tin về cạnh của đốt sống).
    """
    def __init__(self, in_channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        # Bước Squeeze: Gom thông tin không gian (H, W) thành 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Bước Excitation: Học mức độ quan trọng của từng kênh
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Nhân trọng số (mức độ quan trọng) vào đặc trưng ban đầu
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Khối Residual (Skip Connection cục bộ)
    Giúp mạng học sâu hơn mà không bị triệt tiêu đạo hàm (Vanishing Gradient).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Nhánh chính (Mạng tích chập thông thường)
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        
        # Nhánh tắt (Skip Connection)
        # Nếu số kênh đầu vào khác đầu ra (hoặc có giảm kích thước), cần dùng Conv 1x1 để đồng bộ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    Nhìn hình ảnh ở nhiều "tầm nhìn" (FOV) khác nhau cùng lúc.
    Giúp nhận diện tốt các đốt sống có kích thước đa dạng.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Nhánh 1: Tích chập 1x1 thông thường
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Các nhánh có tỷ lệ giãn nở (Dilation) khác nhau
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        
        # Global Average Pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        # Gộp tất cả các nhánh lại và giảm số chiều
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.image_pool(x)
        image_features = nn.functional.interpolate(image_features, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([
            self.conv_1x1_1(x),
            self.conv_3x3_1(x),
            self.conv_3x3_2(x),
            self.conv_3x3_3(x),
            image_features
        ], dim=1)
        
        return self.final_conv(out)

class AttentionBlock(nn.Module):
    """
    Khối Attention (Tập trung không gian)
    Giúp bộ giải mã (Decoder) chọn lọc thông tin từ bộ mã hóa (Encoder) tốt hơn,
    làm nổi bật vùng đốt sống cổ và làm mờ các nhiễu nền.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResUNetPlusPlus(nn.Module):
    """
    Kiến trúc ResUNet++ Hoàn Chỉnh (Đã sửa lỗi đồng bộ kích thước)
    """
    def __init__(self, in_channels=3, num_classes=1, filters=[32, 64, 128, 256, 512]):
        super(ResUNetPlusPlus, self).__init__()
        
        # Lớp đầu vào
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # ----- BỘ MÃ HÓA (ENCODER) -----
        self.res_block1 = ResidualBlock(filters[0], filters[1], stride=2)
        self.se_block1 = SqueezeAndExcitation(filters[1])
        
        self.res_block2 = ResidualBlock(filters[1], filters[2], stride=2)
        self.se_block2 = SqueezeAndExcitation(filters[2])
        
        self.res_block3 = ResidualBlock(filters[2], filters[3], stride=2)
        self.se_block3 = SqueezeAndExcitation(filters[3])
        
        # THÊM KHỐI NÀY: Khối thứ 4 để giảm kích thước xuống cho khớp với kịch bản giải mã
        self.res_block4 = ResidualBlock(filters[3], filters[4], stride=2)
        self.se_block4 = SqueezeAndExcitation(filters[4])
        
        # ----- CỔ CHAI (BOTTLENECK) -----
        # Sử dụng ASPP tại điểm sâu nhất của mạng (sau khối 4)
        self.aspp = ASPP(filters[4], filters[4])
        
        # ----- BỘ GIẢI MÃ (DECODER) -----
        # Nhánh 1 (Từ dưới lên)
        self.up_conv1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.dec_res_block1 = ResidualBlock(filters[3] * 2, filters[3]) # *2 vì ghép nối (concatenate)
        self.dec_se_block1 = SqueezeAndExcitation(filters[3])
        
        # Nhánh 2
        self.up_conv2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.dec_res_block2 = ResidualBlock(filters[2] * 2, filters[2])
        self.dec_se_block2 = SqueezeAndExcitation(filters[2])
        
        # Nhánh 3
        self.up_conv3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.dec_res_block3 = ResidualBlock(filters[1] * 2, filters[1])
        self.dec_se_block3 = SqueezeAndExcitation(filters[1])

        # Lớp đầu ra (ASPP một lần nữa để làm mịn nhãn)
        self.final_aspp = ASPP(filters[1], filters[0])
        
        # Cắt về số lớp mong muốn (1 lớp cho phân đoạn nhị phân: Nền / Đốt sống)
        self.output_layer = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        # Sử dụng Sigmoid ở đầu ra vì chúng ta dự đoán xác suất [0, 1] cho mỗi pixel
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # -- Encoder --
        e0 = self.input_layer(x)              # Cấp độ 0 (size chuẩn)
        
        e1 = self.res_block1(e0)              # Cấp độ 1 (size / 2)
        e1 = self.se_block1(e1)
        
        e2 = self.res_block2(e1)              # Cấp độ 2 (size / 4)
        e2 = self.se_block2(e2)
        
        e3 = self.res_block3(e2)              # Cấp độ 3 (size / 8)
        e3 = self.se_block3(e3)
        
        e4 = self.res_block4(e3)              # Cấp độ 4 (size / 16) - ĐÃ THÊM
        e4 = self.se_block4(e4)
        
        # -- Bottleneck --
        bottleneck = self.aspp(e4)            # Cổ chai xử lý ở size / 16
        
        # -- Decoder --
        # Giải mã nhánh 1
        d1 = self.up_conv1(bottleneck)        # Upsample từ size / 16 lên size / 8
        x_att1 = self.att1(g=d1, x=e3)        # Ghép nối với cấp độ 3 (size / 8) -> KHỚP KÍCH THƯỚC!
        d1 = torch.cat((d1, x_att1), dim=1)   
        d1 = self.dec_res_block1(d1)          
        d1 = self.dec_se_block1(d1)
        
        # Giải mã nhánh 2
        d2 = self.up_conv2(d1)                # Upsample từ size / 8 lên size / 4
        x_att2 = self.att2(g=d2, x=e2)        # Ghép nối với cấp độ 2 (size / 4)
        d2 = torch.cat((d2, x_att2), dim=1)   
        d2 = self.dec_res_block2(d2)          
        d2 = self.dec_se_block2(d2)
        
        # Giải mã nhánh 3
        d3 = self.up_conv3(d2)                # Upsample từ size / 4 lên size / 2
        x_att3 = self.att3(g=d3, x=e1)        # Ghép nối với cấp độ 1 (size / 2)
        d3 = torch.cat((d3, x_att3), dim=1)   
        d3 = self.dec_res_block3(d3)          
        d3 = self.dec_se_block3(d3)
        
        # -- Đầu ra --
        out = self.final_aspp(d3)             # ASPP làm mịn
        
        # Đưa ảnh về đúng kích thước gốc
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = self.output_layer(out)          
        out = self.sigmoid(out)               
        
        return out

# --- KIỂM TRA THỬ MÔ HÌNH ---
if __name__ == "__main__":
    print("Đang khởi tạo mô hình ResUNet++...")
    # in_channels=3 vì ảnh X-quang bạn đã chuyển sang RGB (3 kênh màu)
    # num_classes=1 vì ta chỉ cần phân loại 2 lớp: 0 (Nền) và 1 (Đốt sống)
    model = ResUNetPlusPlus(in_channels=3, num_classes=1)
    
    # Tạo thử 1 batch dữ liệu giả lập có kích thước giống y hệt output của DataLoader lúc nãy
    # Batch = 4, Channel = 3, Height = 256, Width = 256
    dummy_input = torch.randn(4, 3, 256, 256) 
    
    print(f"Kích thước tensor đầu vào giả lập: {dummy_input.shape}")
    
    # Cho dữ liệu chạy qua mô hình (quá trình Feed Forward)
    output = model(dummy_input)
    
    print(f"Kích thước tensor đầu ra dự đoán: {output.shape}")
    
    # Nếu output có dạng [4, 1, 256, 256] (khớp với mask) thì mô hình được code đúng!
    if output.shape == (4, 1, 256, 256):
         print("=> TUYỆT VỜI! Kích thước đầu ra của mô hình hoàn toàn khớp với kích thước của ảnh Mask nhãn (Mask). Kiến trúc mô hình đã chính xác.")
    else:
         print("=> LỖI: Kích thước đầu ra không khớp, hãy kiểm tra lại cấu trúc mạng.")