# BÁO CÁO TOÀN DIỆN: PHÂN TÍCH LỢI NHUẬN & DỰ BÁO DOANH THU
## Đội thi: NHLBike (Datathon VinUni 2026)

Báo cáo này tài liệu hóa toàn bộ thông tin dự án, từ cấu trúc cơ sở dữ liệu, các phân tích thống kê chẩn đoán (EDA), kỹ thuật xây dựng mô hình học máy chuỗi thời gian (Sales Forecasting), đến các đề xuất hành động thực tiễn.

---

## 📂 1. Cấu trúc Dự án (Enterprise Project Layout)

Dự án được cấu trúc theo tiêu chuẩn vận hành Data Science tại các tập đoàn lớn (MNCs):

```text
Datathon_Vinuni_2026/
├── data/
│   ├── raw/                  # 14 file CSV dữ liệu thô ban đầu (Read-only)
│   ├── processed/            # Dữ liệu trung gian sau khi xử lý (nếu có)
│   └── submissions/          # Kết quả dự phóng đầu ra (submission.csv)
├── notebooks/                # Jupyter Notebooks thực nghiệm
│   ├── 00_baseline.ipynb     # Mô hình baseline seasonal mẫu của BTC
│   ├── 01_eda_and_insights.ipynb  # Diagnostic EDA & Trực quan hóa báo cáo
│   └── 02_sales_forecasting.ipynb # Pipeline LightGBM, trích xuất đặc trưng & Dự báo
├── src/                      # Mã nguồn module hóa (Production-ready)
│   ├── __init__.py           # Đánh dấu src là một python package
│   ├── data_loader.py        # Hàm tải dữ liệu thô đồng bộ
│   ├── features.py           # Kỹ nghệ đặc trưng, chuẩn hóa xu hướng
│   └── models.py             # Huấn luyện mô hình, kiểm định chéo và hậu xử lý
├── reports/                  # Báo cáo và tài liệu hỗ trợ
│   ├── figures/              # Biểu đồ PNG kết xuất tự động từ Notebook
│   │   ├── fig1_gross_profit.png         # Lợi nhuận gộp theo Tháng & Năm
│   │   ├── fig2_discount_rates.png       # Tỷ lệ chiết khấu danh nghĩa Chẵn - Lẻ
│   │   ├── fig3_customer_decay.png       # Cohort Retention Heatmap
│   │   ├── fig4_new_customer_forecast.png# Dự báo lượng khách mới lũy kế
│   │   ├── fig5_regional_loyalty.png     # Phân tích Recency & Frequency địa lý
│   │   ├── fig6_feature_importance.png   # Độ quan trọng đặc trưng LightGBM
│   │   └── fig7_shap_analysis.png        # Phân tích đóng góp đặc trưng SHAP
│   ├── NHLBike_report.pdf    # Báo cáo chi tiết đề xuất kinh doanh
│   └── Đề thi Vòng 1.pdf     # Đề bài gốc của Datathon
├── .gitignore                # Chặn các file rác, checkpoints và môi trường ảo (.venv)
├── requirements.txt          # Danh sách thư viện đồng bộ môi trường
├── README.md                 # Hướng dẫn nhanh dự án
└── reports/DATATHON_REPORT.md # Báo cáo kỹ thuật chi tiết (File này)
```

---

## 🗄️ 2. Từ điển Dữ liệu (Data Dictionary & Schemas)

Dự án xử lý dữ liệu từ 14 nguồn bảng CSV thô, được chia thành 4 nhóm nghiệp vụ chính:

### 2.1. Nhóm Dữ liệu Phân tích (Analytical Data)
* **[sales.csv](../data/raw/sales.csv)**: Dữ liệu doanh số và chi phí tài chính hàng ngày.
  * `Date` (datetime): Ngày ghi nhận.
  * `Revenue` (float): Tổng doanh thu trong ngày (VND).
  * `COGS` (float): Giá vốn hàng bán trong ngày (VND).

### 2.2. Nhóm Dữ liệu Giao dịch (Transactional Data)
* **[orders.csv](../data/raw/orders.csv)**: Thông tin đơn đặt hàng.
  * `order_id` (int, Primary Key): Mã đơn hàng.
  * `order_date` (datetime): Ngày đặt hàng.
  * `customer_id` (int): Mã khách hàng.
  * `zip` (int): Mã bưu chính nơi nhận.
  * `order_status` (str): Trạng thái đơn hàng (`delivered`, `cancelled`, `returned`, `shipped`, ...).
  * `payment_method` (str): Phương thức thanh toán (`credit_card`, `cod`, ...).
  * `device_type` (str): Thiết bị đặt hàng (`desktop`, `mobile`).
  * `order_source` (str): Kênh tiếp cận khách hàng (`paid_search`, `organic_search`, ...).
* **[order_items.csv](../data/raw/order_items.csv)**: Chi tiết sản phẩm trong đơn hàng.
  * `order_id` (int): Mã đơn hàng.
  * `product_id` (int): Mã sản phẩm.
  * `quantity` (int): Số lượng mua.
  * `unit_price` (float): Đơn giá bán lẻ sản phẩm.
  * `discount_amount` (float): Số tiền chiết khấu thực tế áp dụng.
  * `promo_id` / `promo_id_2` (str): Mã chiến dịch khuyến mại áp dụng.
* **[returns.csv](../data/raw/returns.csv)**: Quản lý hàng trả.
  * `order_id` (int): Mã đơn hàng bị hoàn trả.
  * `return_date` (datetime): Ngày hoàn trả.
  * `return_reason` (str): Lý do trả hàng.

### 2.3. Nhóm Dữ liệu Khách hàng & Tiếp thị (Customer & Marketing Master)
* **[customers.csv](../data/raw/customers.csv)**: Cơ sở dữ liệu khách hàng.
  * `customer_id` (int, Primary Key): Mã khách hàng.
  * `zip` (int): Mã bưu chính.
  * `city` (str): Thành phố đăng ký.
  * `signup_date` (datetime): Ngày đăng ký tài khoản.
  * `gender` (str): Giới tính.
  * `age_group` (str): Nhóm tuổi (`18-24`, `25-34`, ...).
  * `acquisition_channel` (str): Kênh thu hút khách hàng.
* **[promotions.csv](../data/raw/promotions.csv)**: Các chiến dịch ưu đãi.
  * `promo_id` (str, Primary Key): Mã khuyến mại.
  * `promo_name` (str): Tên chiến dịch khuyến mại.
  * `promo_type` (str): Loại ưu đãi (`percentage` hoặc `fixed`).
  * `discount_value` (float): Giá trị ưu đãi (Phần trăm hoặc số tiền cố định tương ứng).
  * `start_date` / `end_date` (datetime): Thời gian diễn ra chiến dịch.
  * `applicable_category` (str): Danh mục hàng áp dụng.

### 2.4. Nhóm Dữ liệu Vận hành (Operational Data)
* **[geography.csv](../data/raw/geography.csv)**: Phân vùng địa lý Việt Nam.
  * `zip` (int, Primary Key): Mã bưu chính.
  * `city` (str): Tên tỉnh/thành phố.
  * `region` (str): Vùng miền (`Central`, `East`, `West`).
  * `district` (str): Quận/Huyện.
* **[web_traffic.csv](../data/raw/web_traffic.csv)**: Lượng truy cập website hàng ngày.
  * `date` (datetime): Ngày ghi nhận.
  * `sessions` (int): Số lượt truy cập.
  * `unique_visitors` (int): Số người dùng duy nhất truy cập.

---

## 📊 3. EDA & Phân tích Chẩn đoán (Diagnostic EDA)

Các thống kê mô tả nghiệp vụ và chẩn đoán tài chính được thực thi tự động trong notebook [01_eda_and_insights.ipynb](notebooks/01_eda_and_insights.ipynb):

### 3.1. Phân tích Lợi nhuận gộp & Tính chu kỳ (Hình 1)
* **Mục tiêu**: Xác định các quy luật dài hạn của Lợi nhuận gộp (Gross Profit = Revenue - COGS).
* **Công thức & Logic**: 
  1. *Lợi nhuận theo năm*: Lọc các năm đầy đủ dữ liệu (2013-2022), nhóm theo `year` và tính tổng `GrossProfit` hàng năm.
  2. *Lợi nhuận theo tháng*: Nhóm theo `(year, month)` để tính tổng lợi nhuận gộp hàng tháng của từng năm, sau đó lấy trung bình của từng tháng (1 đến 12) trên toàn lịch sử để loại bỏ nhiễu tăng trưởng.
* **Insights**:
  * *Mùa vụ*: Lợi nhuận gộp đạt đỉnh vào Quý II (đạt cao nhất vào Tháng 5 với ~40.7 triệu VND) và chạm đáy sâu vào Tháng 8 (~-0.2 triệu VND) và Tháng 12 (~0.6 triệu VND).
  * *Chu kỳ chẵn - lẻ*: Lợi nhuận gộp năm chẵn luôn cao hơn năm lẻ liền kề khoảng **44%** (Ví dụ: 2018 đạt ~0.308 tỷ VND trong khi 2019 rơi về ~0.132 tỷ VND). Năm 2021 ghi nhận mức thấp kỷ lục kỷ lục là ~0.102 tỷ VND.

![Hình 1: Lợi nhuận gộp trung bình theo tháng và năm](figures/fig1_gross_profit.png)

### 3.2. Chẩn đoán nguyên nhân từ tỷ lệ chiết khấu (Hình 2)
* **Mục tiêu**: Kiểm định xem chính sách khuyến mại có phải nguyên nhân gây xói mòn lợi nhuận năm lẻ.
* **Công thức & Logic**: 
  Ghép bảng giao dịch chi tiết `order_items` với bảng đơn hàng `orders` và bảng khuyến mại `promotions` theo `promo_id`.
  Do các chiến dịch lớn như *Urban Blowout* sử dụng hình thức giảm giá cố định (`fixed` = 50.0), nhóm nghiên cứu đã xử lý cột `discount_value` trực tiếp như một tỷ lệ phần trăm (tức là 50% cho các giao dịch áp dụng khuyến mại này, và 0% cho giao dịch không có khuyến mại). Tính giá trị trung bình cột này theo tháng cho năm chẵn và lẻ riêng biệt.
* **Insights**:
  Năm lẻ có nhiều mã khuyến mại phát sinh hơn trong dữ liệu giao dịch (34 mã ở năm lẻ so với 25 mã ở năm chẵn) và mức chiết khấu định danh thực tế sâu hơn rõ rệt. Cụ thể ở năm lẻ, chiết khấu trung bình đạt **28.6%** vào Tháng 8 (do chiến dịch Urban Blowout) và **19.9%** vào Tháng 12 (do chiến dịch Year-End Sale), giải thích vì sao lợi nhuận gộp chạm đáy trong các tháng này ở năm lẻ.

![Hình 2: Tỷ lệ chiết khấu trung bình theo tháng giữa năm chẵn và lẻ](figures/fig2_discount_rates.png)

### 3.3. Sự suy giảm cấu trúc của Nền tảng khách hàng (Hình 3)
* **Mục tiêu**: Đánh giá sức khỏe của tệp khách hàng theo thời gian.
* **Công thức & Logic**: 
  1. *Khách hàng mới*: Số lượng khách hàng thực hiện đơn hàng đầu tiên (first transaction) trong năm.
  2. *Khách hàng hoạt động*: Số lượng khách hàng duy nhất (`nunique`) có phát sinh giao dịch trong năm.
  3. *Cohort Retention Heatmap*: Nhóm người dùng dựa trên năm mua hàng đầu tiên (Cohort Year). Với mỗi cohort, tính tỷ lệ phần trăm khách hàng hoạt động trở lại ở năm thứ $k$ tiếp theo ($Y_k = \frac{\text{Active Users in Year (Cohort + k)}}{\text{Total Size of Cohort}} \times 100$).
* **Insights**:
  * Lượng khách hàng mới giảm **95%** sau 9 năm (từ 25.1K khách năm 2013 xuống còn 1.3K khách năm 2022).
  * Tỉ lệ giữ chân khách hàng sụt giảm nghiêm trọng: Cohort 2012 giữ chân được **64.7%** khách hàng sau 1 năm ($Y_1$). Tuy nhiên, các cohort gần đây từ năm 2018 trở đi chỉ còn tỷ lệ giữ chân cực kỳ thấp khoảng **9.3%**.
  * Lượng khách hàng hoạt động thực tế trượt dài từ đỉnh 40.9K (năm 2016) xuống còn 24.7K (năm 2022).

![Hình 3: Biến động quy mô khách hàng mới/hoạt động và ma trận Cohort Retention](figures/fig3_customer_decay.png)

### 3.4. Dự báo xu hướng suy thoái khách hàng mới (Hình 4)
* **Mục tiêu**: Ngoại suy lượng khách hàng mới để cảnh báo ban lãnh đạo.
* **Công thức & Logic**: Sử dụng mô hình xu hướng Power Law trên dữ liệu lịch sử (2013-2022) và thực hiện dự báo cho năm 2023, 2024.
* **Insights**: Số lượng khách hàng mới dự kiến tiếp tục chạm đáy thấp kỷ lục: chỉ còn **1,060 khách hàng (năm 2023)** và **945 khách hàng (năm 2024)**, cho thấy mức độ khẩn cấp phải thay đổi chiến lược thu hút (Acquisition).

![Hình 4: Dự báo lượng khách hàng mới giai đoạn 2013-2024](figures/fig4_new_customer_forecast.png)

### 3.5. Phân tích lòng trung loyalty theo Vùng địa lý (Hình 5)
* **Mục tiêu**: Định vị nhóm khách hàng có giá trị cao nhất theo địa lý.
* **Công thức & Logic**: 
  Loại bỏ các giao dịch hoàn trả (`returns.csv`). Ghép bảng đơn hàng hợp lệ với bảng địa lý `geography.csv`.
  1. *Thời gian mua lại (Median Recency Days)*: Sắp xếp đơn hàng của mỗi khách hàng theo thời gian, tính số ngày chênh lệch giữa các đơn hàng liên tiếp, và lấy trung vị theo khu vực (`region`).
  2. *Tần suất mua sắm (Average Orders per Customer)*: Số đơn hàng trung bình một khách hàng đặt theo từng vùng.
* **Insights**:
  Miền Tây (West) thể hiện lòng trung thành vượt trội với trung vị thời gian mua lại ngắn nhất là **95 ngày** (miền East là 185 ngày, Central là 179 ngày), đồng thời tần suất mua hàng trung bình đạt **10.86 lần/khách hàng** (gần gấp đôi các miền khác chỉ đạt khoảng 5.8-6.3 lần).

![Hình 5: Lòng trung thành và tần suất mua hàng theo vùng địa lý](figures/fig5_regional_loyalty.png)

---

## 🤖 4. Mô hình Dự báo Doanh thu & COGS (Predictive Modeling)

Được phát triển và chạy thử trong notebook [02_sales_forecasting.ipynb](notebooks/02_sales_forecasting.ipynb) với mã nguồn tại thư mục [src/](src/):

### 4.1. Kỹ nghệ Đặc trưng (Feature Engineering)
Tích hợp các nhóm đặc trưng hành vi và chu kỳ nhằm tối ưu hóa học máy giám sát:
* **Mã hóa Chu kỳ Lượng giác (Cyclical Features)**: 
  $$\sin\_doy = \sin\left(\frac{2\pi \cdot doy}{365.25}\right), \quad \cos\_doy = \cos\left(\frac{2\pi \cdot doy}{365.25}\right)$$
  Giúp mô hình hiểu sự liên tục của chu kỳ thời gian (giữa 31/12 và 01/01).
* **Đặc trưng Hành vi trượt**: Mapped lịch sử trượt trung bình (rolling averages) giai đoạn lịch sử hiện đại (post-2019) cho Tỷ lệ chuyển đổi (`feat_conv`), Giá trị đơn hàng trung bình (`feat_aov`), Biên lợi nhuận (`feat_margin`) và Lượt truy cập (`feat_sessions`) theo từng tháng của năm.
* **Bối cảnh Khuyến mại**: Tính số lượng chương trình khuyến mại đang hoạt động đồng thời tại mỗi thời điểm để mô hình hóa ảnh hưởng cộng gộp của khuyến mại.

### 4.2. Chuẩn hóa xu hướng (Trend Normalization)
Để đối phó với sự tăng trưởng/suy giảm dài hạn, mô hình áp dụng kỹ thuật tách xu hướng:
1. Tính tốc độ tăng trưởng lũy kế của doanh thu và COGS các năm gần nhất (2020-2022):
   $$\text{Growth Rate} = \left(\prod_{t} (1 + \text{pct\_change}_t)\right)^{\frac{1}{n}}$$
2. Tính đường cơ sở xu hướng hàng ngày (Base Trend) dự phóng cho tương lai:
   $$\text{Trend}_t = \text{Base Daily 2022} \times (\text{Growth Rate})^{\text{Years Ahead}}$$
3. Chuẩn hóa target thực tế về dạng không tỷ lệ (scale-free):
   $$\text{Revenue}_{norm} = \frac{\text{Revenue}}{\text{Trend}_{rev}}, \quad \text{COGS}_{norm} = \frac{\text{COGS}}{\text{Trend}_{cogs}}$$
   LightGBM chỉ tập trung học các dao động mùa vụ ngắn hạn quanh đường xu hướng.

### 4.3. Huấn luyện & Đánh giá Chéo (Walk-Forward Validation)
* **Trọng số mẫu (Time-decay Weighting)**: Đánh trọng số cao hơn cho dữ liệu gần hiện tại (dữ liệu từ năm 2019 trở đi nhân hệ số 1.5) để mô hình ưu tiên học mẫu hình tiêu dùng mới.
* **Kiểm định Chéo**: Thực hiện Time-series Cross-Validation bằng cách lăn thời gian (lấy năm 2021 và 2022 làm fold validation độc lập) để đánh giá độ ổn định của mô hình và tránh rò rỉ dữ liệu (Data Leakage).
* **Mô hình**: Thuật toán LightGBM Regressor tối ưu hóa trực tiếp hàm mục tiêu MAE để hạn chế nhạy cảm với các điểm dị biệt (outliers).

#### Bảng 2: Hiệu suất Mô hình trên tập Validation (Trung bình CV 2021-2022)

| Biến dự báo | Sai số tuyệt đối trung bình (MAE) | Căn phương sai sai số (RMSE) | Hệ số xác định ($R^2$) |
|---|---|---|---|
| **Doanh thu (Revenue)** | 521,087.38 | 722,325.27 | **0.8138** |
| **Giá vốn (COGS)** | 475,310.88 | 656,932.62 | **0.7971** |

---

## 🔍 5. Giải thích Mô hình (XAI & Model Interpretation)

### 5.1. Độ quan trọng của Đặc trưng (Feature Importance)
Biểu đồ Độ quan trọng đặc trưng (Hình 6) cho thấy:
* Các đặc trưng thời gian như `month_progress`, `cos_doy`, `sin_doy`, `day`, `dow` và `doy` đứng đầu, cho thấy mô hình chủ yếu học được cấu trúc mùa vụ và nhịp vận hành trong tháng.
* Các đặc trưng hành vi như `feat_aov`, `feat_conv` và `feat_margin` vẫn có đóng góp, giúp mô hình gắn tín hiệu mùa vụ với logic thương mại điện tử như giá trị giỏ hàng, chuyển đổi và biên lợi nhuận.

![Hình 6: Độ quan trọng đặc trưng LightGBM](figures/fig6_feature_importance.png)

### 5.2. Phân tích đóng góp của Đặc trưng (SHAP Summary Analysis)
Biểu đồ SHAP (Hình 7) làm rõ chiều hướng tác động của các biến:
* Chỉ số khuyến mại (`promo_count`) có tương tác đáng kể với dự báo, nhưng trên mô hình doanh thu hiện tại các giá trị khuyến mại cao thường dịch chuyển SHAP về phía âm. Điều này phù hợp với chẩn đoán rằng các giai đoạn khuyến mại sâu có thể kéo doanh thu/lợi nhuận ròng xuống nếu chiết khấu làm xói mòn giá trị đơn hàng.
* Các đặc trưng mùa vụ và nhịp thời gian (`post_2019`, `cos_doy`, `month_progress`, `sin_doy`) có biên độ SHAP lớn nhất, phản ánh sự thay đổi cấu trúc sau 2019 và tính chu kỳ trong nhu cầu mua sắm.
* Biến chu kỳ lượng giác (`cos_doy`, `sin_doy`) phản ánh rõ rệt các điểm đảo chiều mùa vụ ở cuối năm.

![Hình 7: Phân tích SHAP Summary](figures/fig7_shap_analysis.png)

### 5.3. Hậu xử lý áp đặt ràng buộc Nghiệp vụ (Domain Constraints)
Để đảm bảo kết quả dự báo luôn an toàn tài chính, chúng tôi thiết lập bộ lọc sau xử lý nhằm kiểm soát tỷ lệ COGS/Revenue luôn nằm dưới mức 130%:
$$\text{COGS}_{final} = \max(0, \min(\text{COGS}_{pred}, \text{Revenue}_{pred} \times 1.30))$$

---

## 🎯 6. Đề xuất Hành động dựa trên Dữ liệu (Prescriptive Analytics)

Dựa trên các phân tích thống kê chẩn đoán và dự báo mô hình, chúng tôi đề xuất 3 giải pháp chiến lược định lượng cho doanh nghiệp:

1. **Chiến dịch "The West Loyalists"**: 
   * *Phát hiện*: Khách hàng miền Tây (West) mua hàng rất thường xuyên (trung vị thời gian mua lại là 95 ngày) và tần suất mua cực cao (10.86 lần/khách).
   * *Hành động*: Xây dựng hệ thống Marketing Automation tự động gửi email/tin nhắn ưu đãi mua lại ("Repurchase Voucher") vào ngày thứ **80 - 85** (5-10 ngày trước mốc trung vị quay lại của họ) để duy trì lòng trung thành của nhóm này.
2. **Tái thiết kế khuyến mại Urban Blowout**:
   * *Phát hiện*: Khuyến mại Tháng 8 (năm lẻ) làm sụt giảm lợi nhuận gộp nghiêm trọng do áp dụng đại trà và chiết khấu định danh trung bình lên tới 28.6%.
   * *Hành động*: Chuyển đổi chiến dịch Urban Blowout từ khuyến mãi công khai (Public) sang khuyến mãi tri ân riêng tư **(Loyalty-only Event)** gửi qua mã code kín cho khách hàng VIP đã mua >5 đơn hàng. Chiến lược này giúp bảo vệ biên lợi nhuận gộp mà vẫn giữ chân được khách hàng cốt lõi.
3. **Tái định vị Acquisition trẻ**:
   * *Phát hiện*: Số lượng khách hàng mới đã suy giảm nghiêm trọng 95% qua 9 năm trên tất cả các kênh tuyển dụng.
   * *Hành động*: Hợp tác với Micro-influencers thời trang trong phân khúc chủ lực Streetwear/Outdoor và đẩy mạnh Social Commerce (TikTok Shop, Instagram Shop) nhằm trẻ hóa tệp khách hàng.

---

## 🛠️ 7. Hướng dẫn Chạy lại & Kiểm thử kết quả

### 7.1. Cài đặt Môi trường
Dự án yêu cầu cài đặt **Python 3.10+** và các thư viện cần thiết trong file [requirements.txt](requirements.txt):

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate  # Trên Windows dùng: .venv\Scripts\activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 7.2. Chạy Phân tích & Trực quan hóa
Để chạy lại toàn bộ quy trình tính toán EDA và xuất các hình ảnh đồ thị phục vụ báo cáo:
```bash
python -c "import json; nb=json.load(open('notebooks/01_eda_and_insights.ipynb')); exec('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))"
```

### 7.3. Chạy Mô hình Học máy & Xuất dự báo
Để huấn luyện mô hình LightGBM, thực hiện đánh giá Cross-Validation và xuất file dự báo kết quả cuối cùng ra [data/submissions/submission.csv](data/submissions/submission.csv):
```bash
python -c "import json; nb=json.load(open('notebooks/02_sales_forecasting.ipynb')); exec('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))"
```
