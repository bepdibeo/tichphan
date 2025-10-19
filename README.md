## NHÓM 1- Ứng dụng minh họa hai phương pháp tính gần đúng tích phân
Ứng dụng được viết bằng **Python và Streamlit**, giúp **trực quan hóa quá trình tính gần đúng tích phân** bằng hai quy tắc cơ bản:

* **Phương pháp Hình thang (Trapezoidal rule)** – xấp xỉ hàm bằng các đoạn thẳng nối các điểm chia.
* **Phương pháp Simpson** – xấp xỉ hàm bằng các cung **parabol bậc 2 nội suy** nối các điểm chia.
* Cho phép nhập **số khoảng chia `n`** hoặc **sai số `ε`**, chương trình tự động lặp đến khi hội tụ.
* Hiển thị **kết quả, sai số, bảng giá trị trọng số** và **đồ thị tô vùng tích phân** cho từng phương pháp.
* Kiểm tra **lỗi miền xác định, giá trị phức, vô hạn, NaN (Not a Number)** để tránh sai về mặt toán học.
* Giúp **so sánh trực tiếp độ chính xác và mức hội tụ** của hai quy tắc trên cùng một hàm.
