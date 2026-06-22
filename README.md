# Datathon VinUni 2026 - NHLBike

Phân tích lợi nhuận, hành vi khách hàng và dự báo doanh thu/COGS cho một doanh nghiệp thương mại điện tử thời trang. Dự án gồm hai phần chính: EDA phục vụ đề xuất kinh doanh và pipeline LightGBM phục vụ file dự báo nộp bài.

## Tổng Quan

**Bài toán:** dự báo `Revenue` và `COGS` theo ngày cho giai đoạn test, đồng thời giải thích các nguyên nhân làm lợi nhuận suy giảm.

**Kết quả chính:**

| Hạng mục | Kết quả |
|---|---:|
| Validation Revenue MAE | 521,087.38 |
| Validation Revenue RMSE | 722,325.27 |
| Validation Revenue R2 | 0.8138 |
| Validation COGS MAE | 475,310.88 |
| Validation COGS RMSE | 656,932.62 |
| Validation COGS R2 | 0.7971 |

## Cấu Trúc Dự Án

```text
Datathon_Vinuni_2026/
├── data/
│   ├── raw/                  # 14 CSV gốc từ đề bài
│   ├── processed/            # Dữ liệu trung gian nếu cần mở rộng
│   └── submissions/          # File dự báo cuối cùng
├── notebooks/
│   ├── 00_baseline.ipynb     # Baseline seasonal/trend
│   ├── 01_eda_and_insights.ipynb
│   └── 02_sales_forecasting.ipynb
├── src/
│   ├── data_loader.py        # Hàm tải dữ liệu
│   ├── features.py           # Feature engineering
│   └── models.py             # Huấn luyện, đánh giá, hậu xử lý
├── reports/
│   ├── DATATHON_REPORT.md    # Báo cáo kỹ thuật chi tiết
│   ├── NHLBike_report.pdf    # Báo cáo nộp/chia sẻ
│   ├── NHLBike_report.txt
│   ├── de_thi_vong_1.txt
│   ├── Đề thi Vòng 1.pdf
│   └── figures/              # 7 biểu đồ dùng trong báo cáo
├── requirements.txt
└── README.md
```

## Dữ Liệu

Dữ liệu gốc nằm trong `data/raw/` và được giữ nguyên để đảm bảo khả năng tái lập. Các bảng quan trọng:

| Nhóm | File |
|---|---|
| Master | `products.csv`, `customers.csv`, `promotions.csv`, `geography.csv` |
| Transaction | `orders.csv`, `order_items.csv`, `payments.csv`, `shipments.csv`, `returns.csv`, `reviews.csv` |
| Analytical | `sales.csv`, `sample_submission.csv` |
| Operational | `inventory.csv`, `web_traffic.csv` |

## Biểu Đồ Và Mức Độ Phù Hợp Với Báo Cáo

Các biểu đồ hiện tại đều phục vụ trực tiếp cho luận điểm trong báo cáo. Một số diễn giải đã được chỉnh lại để khớp số liệu thực tế từ dữ liệu và hình ảnh.

| Hình | File | Vai trò trong report | Ghi chú kiểm tra |
|---|---|---|---|
| 1 | `reports/figures/fig1_gross_profit.png` | Mùa vụ lợi nhuận và chu kỳ năm chẵn/lẻ | Tháng 8 khoảng `-0.2M`, tháng 12 khoảng `0.6M`; README/report đã sửa theo số liệu này |
| 2 | `reports/figures/fig2_discount_rates.png` | Chẩn đoán chiết khấu làm xói mòn lợi nhuận | T8 năm lẻ `28.6%`, T12 năm lẻ `19.9%` |
| 3 | `reports/figures/fig3_customer_decay.png` | Suy giảm khách hàng mới, khách hoạt động và retention | Khách mới giảm từ `25.1K` xuống `1.3K` |
| 4 | `reports/figures/fig4_new_customer_forecast.png` | Cảnh báo xu hướng khách hàng mới 2023-2024 | Đã chỉnh notebook để nhãn không chồng lên nhau khi xuất lại |
| 5 | `reports/figures/fig5_regional_loyalty.png` | Cơ sở cho chiến dịch West Loyalists | West: median recency `95` ngày, frequency `10.86` đơn/khách |
| 6 | `reports/figures/fig6_feature_importance.png` | Giải thích importance của LightGBM | Notebook đã được sửa để lưu hình tự động |
| 7 | `reports/figures/fig7_shap_analysis.png` | Giải thích chiều tác động SHAP | Notebook đã được sửa để lưu hình tự động |

## Insight Chính

1. **Lợi nhuận có mùa vụ rõ:** lợi nhuận gộp đạt đỉnh vào tháng 5 với khoảng `40.7M VND`, nhưng giảm rất mạnh vào tháng 8 và tháng 12.
2. **Năm lẻ chịu áp lực chiết khấu sâu hơn:** đặc biệt tháng 8 và tháng 12, trùng với các giai đoạn khuyến mại mạnh.
3. **Nền tảng khách hàng suy giảm:** khách hàng mới giảm khoảng 95% từ 2013 đến 2022, trong khi khách hàng hoạt động giảm từ `40.9K` xuống `24.7K`.
4. **West là vùng retention tốt nhất:** khách hàng West quay lại nhanh hơn và mua nhiều hơn hẳn các vùng còn lại.
5. **Mô hình học được mùa vụ và thay đổi cấu trúc:** Feature importance và SHAP cho thấy các biến thời gian/chu kỳ là nhóm tín hiệu mạnh nhất; các biến hành vi như AOV, conversion và margin bổ sung logic thương mại điện tử.

## Pipeline Mô Hình

Notebook chính: `notebooks/02_sales_forecasting.ipynb`

Các bước chính:

1. Tải `sales.csv`, `sample_submission.csv`, `promotions.csv`, `web_traffic.csv`, `orders.csv`, `order_items.csv`.
2. Tạo đặc trưng lịch, chu kỳ `sin_doy`/`cos_doy`, hành vi tháng và số khuyến mại đang hoạt động.
3. Chuẩn hóa xu hướng dựa trên tăng trưởng 2020-2022.
4. Huấn luyện LightGBM cho `Revenue` và `COGS`.
5. Kiểm định walk-forward trên 2021 và 2022.
6. Áp ràng buộc nghiệp vụ: `0 <= COGS <= Revenue * 1.30`.
7. Xuất `data/submissions/submission.csv`.

## Cách Chạy Lại

Tạo và kích hoạt môi trường:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy EDA và tạo lại Hình 1-5:

```bash
cd notebooks
python -c "import json; nb=json.load(open('01_eda_and_insights.ipynb')); exec('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))"
cd ..
```

Chạy mô hình, tạo lại Hình 6-7 và xuất submission:

```bash
cd notebooks
python -c "import json; nb=json.load(open('02_sales_forecasting.ipynb')); exec('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))"
cd ..
```

File kết quả sau cùng: `data/submissions/submission.csv`.

## Tài Liệu Liên Quan

- Báo cáo kỹ thuật chi tiết: `reports/DATATHON_REPORT.md`
- Báo cáo PDF: `reports/NHLBike_report.pdf`
- Đề bài: `reports/Đề thi Vòng 1.pdf`
