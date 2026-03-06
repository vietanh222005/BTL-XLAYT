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
