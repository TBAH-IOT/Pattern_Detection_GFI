Hướng dẫn Sử dụng (Usage Guide)
4.1. Khởi chạy Giao diện Scanner (Streamlit Web UI)
Để mở ứng dụng quét mô hình trực quan trên trình duyệt:

Bash
streamlit run app.py
Trình duyệt sẽ tự động mở tại địa chỉ: http://localhost:8501.

Chức năng chính trên UI:

Khung thời gian: Chọn khung nến (15m, 1h, 4h,...).

Năm Replay: Chọn năm trong quá khứ để test khả năng quét mô hình.

Điều khiển nến: Sử dụng các nút ⏩ Tiến 1 nến, ⏪ Lùi 1 nến, hoặc ⏭️ Tới Live để mô phỏng thị trường realtime.

Biểu đồ Plotly: Hiển thị nến, đường vẽ mô hình (Pattern Path), mức Entry, TP (xanh) và SL (đỏ).

Bảng Tín hiệu: Hiển thị tên mô hình, Win Rate lịch sử, điểm Similarity %, phương pháp Pivot và thông số TP/SL cụ thể.

4.2. Chạy Calibrate Ngưỡng Similarity (Kiểm định thống kê)
Đo đạc "null distribution" của từng mô hình bằng kỹ thuật Block Bootstrap trên dữ liệu lịch sử thật để tìm ra ngưỡng vượt nhiễu đạt ý nghĩa thống kê (hiệu chỉnh Bonferroni):

Bash
python calibrate_engine.py --calibrate [thư_mục_chứa_data] [số_lần_bootstrap]
Ví dụ:

Bash
python calibrate_engine.py --calibrate . 5000
Sau khi chạy xong, kết quả sim_threshold sẽ được in ra terminal. Bạn chỉ cần copy-paste các giá trị ngưỡng này vào dict WIN_RATE_BY_TF_METHOD trong file similar_engine.py.
