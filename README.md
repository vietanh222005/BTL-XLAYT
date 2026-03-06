# BTL-XLAYT
File dataset.py
1. Quản lý I/O và tối ưu bộ nhớ
   Mô hình học sâu không thể tải toàn bộ hàng ngàn bức ảnh độ phân giải cao vào RAM/VRAM cùng 1 lúc vì sẽ gây tràn bộ nhớ
     + dataset.py được thiết kế theo cơ chế lazy loading (Tải lười biếng). Nó chỉ lưu đường dẫn của các file
     + Chỉ khi nào mô hình yêu cầu chữ liệu cho 1 Batch cụ thể (ví dụ: cần 4 ảnh để train), file này mới thực hiện lệnh đọc file (I/O) từ ổ cứng lên bộ nhớ
2. Tiền xử lý và tăng cường dữ liệu đồng bộ
   Trong phân đoạn ảnh y tế, ảnh gốc và ảnh nhãn là 2 ma trận độc lập nhưng có liên kết chặt chẽ về mặt tọa độ không gian
     + file này sử dụng thư viện Albumentations để đảm bảo: Mọi phép biến đổi hình học áp dụng lên image thì bắt buộc phải áp dụng ma trận biến đổi y hệt lên mask
     + Nếu không có cơ chế xử lý chung này trong file này, ảnh gốc có thể bị xoay nhưng mask không xoay, dẫn đến nhãn bị sai lệch hoàn toàn so với vật thể thực tế
3. Chuyển đổi cấu trúc dữ liệu
   Dữ liệu đọc từ ổ cứng (qua OpenCV) mặc định là NumPy Array với kiểu dữ liệu unit8 (giá trị 0-255). Card đồ họa và PyTorch không tính toán tối nưu trên định dạnh này
    + file này thực hiện ép kiểu và chuẩn hóa: Đưa giá trị pixel về dải float[0,1] hoặc phân bố chuẩn (mean, std)
    + CUối cùng, nó chuyển đổi từ NumPy Array sang PyTorch Tensor - cấu trúc dữ liệu hỗ trợ tính toán gradient trên GPU
5.Kiến trúc bắt buộc của dataset.py
  Để framework PyTorch hiểu và chạy được, class CervicalSpineDataset tring file này bắt buộc phải ghì đè 3 phương thức chuẩn:
  I. __init__(self, img_dir, mask_dir, transform):
    + Hàm khởi tạo. Quét các thư mục lưu trữ, liệt kê tên tất cả các file ảnh và mask hợp lệ. Xử lý logic ghép cặt trên file (ví dụ: ảnh gốc là .jpg nhưng mask được lưu dưới định dạng .png)
  II. __len__(self):
    + Trả về tổng số lượng mẫu có trong tập dữ liệu. Tham số này giúp dataloader tính toán chính xác số lượng Batch cần thiết để hoàn thành 1 Epoch (1 chu kì huấn luyện)
  III. __getitem__(self, index):
    + Hàm cốt lõi. Trích xuất và trả về 1 mẫu dữ liệu duy nhất tại vị trí index
    + Luồng thực thi bên trong: Đọc ảnh OpenCV --> CHuyển BGR sang RGB --> Đọc mask --> Nhị phân hóa mask (ép các giá trị >0 thành 1.0) --> Đưa các image và mask qua toàn bộ transform --> trả về tuple (image_tensor, mask_tensor)
MỤC   ĐÍCH
1. Định vị và ghép các cặp dữ liệu
   + Quét ảnh thư mục chứa ảnh X-Quang gốc và thư mục chứa ảnh nhãn, sau đó ghép chính xác ảnh của bệnh nhân A với đúng mask của bệnh nhân A, đồng thời xử lý các trường hợp lệch đuôi file
2. Tiền xử lý và tăng cường dữ liệu đồng bộ
   + Mục đích quan trọng nhất của file này trong bài toán phân đoạn nahr là đảm bảo tính đồng bộ không gian. Khi áp dụng các kỹ thuật như thu phóng lật,... thì ma trận biến đổi áp dụng lên ảnh gốc cũng phải được áp dụng y hệt lên ảnh mask. Nếu không có bước này, tọa độ đốt sống trên nhãn sẽ bị lệch hoàn toàn so với ảnh gốc

MODEL.py
1. Khối SE (Squeeze-and-Excitation)
- Phân tích luồng xử lý
  + Giả sử Tensor có đầu vào x đang ở 1 lớp nào đó giữa mạng có kích thước là: (Batch=4, Channels=64, Height=128, Width=128)
  + Nghĩa là tại thời điểm này, máy tính đang nhìn vào 64 bức ảnh (64 kênh đặc trưng), mỗi bức kích thước 128x128. Mỗi kênh đang nhìn một thứ khác nhau: Kênh 1 thấy nền đen, kênh 2 thấy viền xương, kênh 3 thấy mô mềm, ...
Khối SE sẽ xử lý 64 kênh này qua 3 bước
  - Bước 1: Squeeze (Ép không gian) 
    + Code y = self.avg_pool(x).view(b, c)
    + Bản chất toán học: Hàm AdaptiveAvgPool2d(1) sẽ lấy tính trung bình cộng của toàn bộ điểm ảnh trên mặt phẳng 128x128 của từng kênh một
    + Kết quả: Ma trận từ 128x128 bị ép dẹp thành đúng 1x1 pixel
    + Ý nghĩa: Lúc này, mỗi kênh trong số 64 kênh chỉ còn lại 1 con số duy nhất. Con số này đại diện cho mức độ hiện diện của đặc trưng đó trên toàn bộ bức ảnh
   - Bước 2: Excitation (Kích thích trọng số)
     + Code: y = self.fc(y).view(b, c, 1, 1)
     + Bản chất toán học: 64 con số vừa thu được sẽ đi qua 1 mạng nơ-ron nhỏ gồm 2 lớp (Linear/Fully Connected)
       + Lớp Linear 1: Giảm số lượng từ 64 xuống còn 4 (giảm 16 lần - reduction=16). Mục đích là để các kênh phải nói chuyện và tổng hợp thông tin với nhau, đồng thời giảm khối lượng tính toán
       + Hàm ReLU: Lọc bỏ các giá trị âm
       + Lớp Linear 2: Phóng to ngược lại từ 4 lên đúng 64 kênh như ban đầu
       + Hàm Sigmoid: Ép 64 con số này về khoảng [0,1]
    + Kết quả: Ta thu được một vector gồm 64 con số thập phân từ 0-1. Đây chính là Vector trọng số
   - Bước 3: Scale ( Nhân điều chỉnh lại ma trận gốc )
     + Code: return x * y.expand_as(x)
     + Bản chất toán học: Phép nhân trực tiếp
       + Nó lấy Vector trọng số nhân ngược lại vào ma trận gốc x (size 64x128x128)
     + Ý nghĩa thực tế:
       + Giả sử kênh số 2 (chuyên tìm xương) được mạng định giá trọng số là 0.95. Toàn bộ điểm ảnh của kênh 2 sẽ nhân được với 0.95
       + Giả sử Kênh số 5 (chuyên nhìn mỡ/nhiễu) bị mạng định giá trọng số là 0.01. Toàn bộ điểm ảnh của Kênh 5 nhân với 0.01 $\rightarrow$ Các giá trị pixel gần như biến thành 0 (Bị triệt tiêu/Squeeze).
   
