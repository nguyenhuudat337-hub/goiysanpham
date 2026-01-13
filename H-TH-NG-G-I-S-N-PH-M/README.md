# 🛒 Context-Aware Recommender System for Instacart
### Hệ thống Gợi ý Sản phẩm Instacart kết hợp Phân cụm & Luật kết hợp theo Ngữ cảnh

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Algorithm](https://img.shields.io/badge/Algorithm-KMeans%20%7C%20FPGrowth-green)

## 📖 Tổng quan (Overview)
Link data: https://drive.google.com/drive/folders/1A9nho8fR9CSi5m4L3ZcWpVNqCI4jtdny?usp=drive_link
Dự án này xây dựng một hệ thống gợi ý sản phẩm lai (Hybrid Recommender System) trên bộ dữ liệu **Instacart Market Basket Analysis**. 

Khác với các hệ thống gợi ý truyền thống (chỉ dựa trên lịch sử mua hàng), hệ thống này tích hợp yếu tố **Ngữ cảnh thời gian thực (Real-time Context)** như: Buổi sáng/Tối, Ngày thường/Cuối tuần. Hệ thống giải quyết bài toán "Cold Start" và cá nhân hóa sâu nhờ chiến lược "Chia để trị" (Segment-based Rule Mining).

**Điểm nổi bật:**
* **Phân cụm người dùng (User Profiling):** Hiểu rõ hành vi khách hàng (VD: Nhóm "Cú đêm", Nhóm "Nội trợ").
* **Gợi ý theo ngữ cảnh:** Sản phẩm gợi ý thay đổi tùy thuộc vào thời điểm khách hàng truy cập.
* **Demo trực quan:** Giao diện tương tác xây dựng bằng Streamlit.

## 📂 Cấu trúc dự án (Project Structure)

```text
├── data/                       # Thư mục chứa dữ liệu thô (orders.csv, products.csv...)
├── output/                     # Thư mục chứa kết quả (Heatmap, Model rules, Metrics...)
├── user_features.csv           # Dữ liệu đặc trưng người dùng (sau khi Feature Engineering)
├── rename_departments.py       # [Bước 1] Script chuẩn hóa tên ngành hàng
├── regenerate_heatmap.py       # [Bước 2] Trực quan hóa dữ liệu (EDA) & Phân tích cụm
├── association_rules.py        # [Bước 3] Chạy thuật toán FP-Growth tìm luật theo ngữ cảnh
├── evaluation.py               # [Bước 4] Đánh giá mô hình (Precision, Recall, F1)
├── app.py                      # [Bước 5] Giao diện Demo (Streamlit)
├── requirements.txt            # Danh sách các thư viện cần cài đặt
└── README.md                   # Tài liệu hướng dẫn này
🛠️ Cài đặt & Môi trường (Installation)
Clone dự án:

Bash

git clone [https://github.com/username/instacart-recommender.git](https://github.com/username/instacart-recommender.git)
cd instacart-recommender
Tạo môi trường ảo (Khuyên dùng):

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Cài đặt thư viện:

Bash

pip install -r requirements.txt
(Nếu chưa có file requirements.txt, hãy cài thủ công: pandas numpy matplotlib seaborn scikit-learn mlxtend streamlit)

🚀 Hướng dẫn thực hiện (Pipeline Execution)
Để tái hiện kết quả, vui lòng chạy các script theo thứ tự sau:

Bước 1: Tiền xử lý dữ liệu
Chuẩn hóa dữ liệu và ánh xạ tên các ngành hàng (Departments).

Bash

python rename_departments.py
Bước 2: Phân tích & Trực quan hóa (EDA)
Vẽ biểu đồ Heatmap để kiểm tra mối tương quan giữa các đặc trưng người dùng và đánh giá chất lượng phân cụm.

Bash

python regenerate_heatmap.py
➡️ Kết quả: File ảnh heatmap.png sẽ được lưu vào thư mục output/.

Bước 3: Khai phá luật kết hợp (Mining)
Đây là bước cốt lõi. Thuật toán FP-Growth sẽ chạy trên từng Cụm (Cluster) và từng Ngữ cảnh (Context).

Bash

python association_rules.py
⚠️ Lưu ý: Quá trình này có thể mất 5-10 phút tùy thuộc vào cấu hình máy tính.

Bước 4: Đánh giá hiệu năng (Evaluation)
Hệ thống sẽ ẩn đi các giao dịch cuối cùng (Test set) và đo lường khả năng dự đoán chính xác.

Bash

python evaluation.py
➡️ Kết quả: Hiển thị các chỉ số Precision@K, Recall@K và F1-Score trên màn hình console.

Bước 5: Chạy Demo (Deployment)
Khởi động ứng dụng web để trải nghiệm gợi ý thực tế.

Bash

streamlit run app.py
Truy cập đường dẫn hiển thị trên terminal (thường là http://localhost:8501).

🧠 Phương pháp luận (Methodology)
Hệ thống hoạt động dựa trên quy trình 3 giai đoạn:

Giai đoạn 1: User Feature Engineering

Trích xuất các đặc trưng hành vi: Morning_Ratio, Night_Ratio, Weekend_Ratio, Avg_Basket_Size.

Chuẩn hóa dữ liệu bằng StandardScaler.

Giai đoạn 2: Clustering (K-Means)

Phân nhóm người dùng dựa trên vector đặc trưng.

Xác định số cụm tối ưu K bằng phương pháp Elbow.

Giai đoạn 3: Contextual Rule Mining

Áp dụng thuật toán FP-Growth cho từng tổ hợp (Cluster, Context).

Cơ chế xếp hạng (Ranking): Ưu tiên luật khớp ngữ cảnh -> Ưu tiên luật có Lift cao.
