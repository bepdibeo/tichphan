## NHÓM 1 - Ứng dụng minh họa hai phương pháp tính gần đúng tích phân
- app.py: file chương trình.
- requirements.txt: các thư viện.
- Khi chạy trên máy cục bộ: cài đặt các thư viện: pip install -r requirements.txt
- Nếu không, cài từng thư viện: pip install streamlit numpy sympy plotly pandas
- Cuối cùng, chạy lệnh: streamlit run app.py

Ứng dụng được viết bằng **Python và Streamlit**, giúp **trực quan hóa quá trình tính gần đúng tích phân** bằng hai phương pháp cơ bản:
- **Phương pháp Hình thang (Trapezoidal rule)** 
- **Phương pháp Simpson** 
- Cho phép nhập **số khoảng chia `n`** hoặc **sai số `ε`**, chương trình tự động lặp đến khi hội tụ.
- Kiểm tra lỗi để tránh sai về mặt toán học.
- Hiển thị **kết quả, sai số, bảng giá trị trọng số** và **đồ thị tô vùng tích phân** cho từng phương pháp.
- Giúp **so sánh trực tiếp độ chính xác** của hai quy tắc trên cùng một hàm.
