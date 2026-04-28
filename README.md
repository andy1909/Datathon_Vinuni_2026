# Datathon_Vinuni_2026
# Datathon 2026: Sales Forecasting - Đội NHLBike

CẤU TRÚC THƯ MỤC
├── raw_data/                   
│   ├── sales.csv               
│   ├── orders.csv              
│   ├── order_items.csv         
│   ├── web_traffic.csv         
│   ├── promotions.csv
│   └── ...     
├── submission.ipynb                   
├── submission.csv              
└── README.md       

CÀI ĐẶT THƯ VIỆN
Yêu cầu sử dụng Python 3.8 trở lên. Vui lòng mở terminal và cài đặt các thư viện cần thiết bằng lệnh sau:
pip install pandas numpy lightgbm shap scikit-learn matplotlib seaborn

HƯỚNG DẪN CHẠY LẠI KÊT QUẢ
Bước 1: Chuẩn bị dữ liệu
  Đảm bảo tất cả các file dữ liệu thô (.csv) đã được đặt đầy đủ vào trong thư mục raw_data/ nằm cùng cấp với các file code.
Bước 2: Khởi chạy dự báo: 
  Mở file submission.ipynb bằng Jupyter Notebook hoặc VS Code. Chạy tuần tự các ô code (Run All) từ trên xuống dưới để xem chi tiết quá trình tiền xử lý, Cross-validation và biểu đồ SHAP.
  
