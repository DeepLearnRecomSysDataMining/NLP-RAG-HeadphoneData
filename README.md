# NLP RAG Project - Headphone Data Search & Evaluation

Dự án này triển khai hệ thống Tìm kiếm thông tin (Retrieval) kết hợp với Reranking (Hybrid Search) dành cho dữ liệu tai nghe và thiết bị âm thanh. Hệ thống hỗ trợ tìm kiếm ngữ nghĩa (Semantic Search), tìm kiếm từ khóa (BM25) và đánh giá hiệu năng tự động.

## Tính năng chính

- **Hybrid Retrieval**: Kết hợp Dense Search (FAISS + Bi-Encoder) và Sparse Search (BM25).
- **Reranking**: Sử dụng Cross-Encoder để xếp hạng lại kết quả, tăng độ chính xác.
- **Metadata Bonus**: Tự động cộng điểm ưu tiên cho thương hiệu và lọc giá từ truy vấn người dùng.
- **Tự động Đánh giá**: Tính toán các chỉ số Precision@K, Hit Rate@K, MRR, nDCG để so sánh các chế độ tìm kiếm.
- **Hỗ trợ CPU/GPU**: Tự động nhận diện thiết bị (hỗ trợ tốt khi chuyển từ Colab sang máy cá nhân).

## Cài đặt

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.9 trở lên. Nên sử dụng môi trường ảo (venv):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu cần thiết

Để chạy ngay mà không cần build index, bạn cần đặt các file sau vào thư mục gốc của dự án:
- `faiss_index_v2/`: Thư mục chứa dữ liệu vector FAISS.
- `docstore_v2.pkl`: File lưu trữ nội dung văn bản gốc.
- `full_xuanvu_database.csv`: Dữ liệu thô để build lại index nếu cần.

các file này đã có trên drive tại thư mục drive : `https://drive.google.com/drive/folders/1GSWRWE7ydo1tcmuvmlTuG-aOEvQQMTz8?usp=sharing` , chỉ cần download về và đặt vào thư mục gốc của dự án. và chạy `python main.py` để test.

## Hướng dẫn chạy
Có thể chạy .py  bằng file [main.py](main.py) hoặc chạy [notebook]() 

### Chạy hệ thống (Đánh giá + Chat)
Đây là file chính để khởi động toàn bộ quy trình:
```bash
python main.py
```
**Quy trình chạy:**
1. Load index từ local (hoặc build mới nếu thiếu).
2. Tự động chạy Evaluation trên bộ câu hỏi mẫu.
3. In bảng kết quả so sánh (MRR Gain).
4. Hỏi người dùng có muốn bắt đầu phiên hỏi đáp tương tác (Chat) không.

## Kết quả đánh giá
Sau khi chạy, kết quả chi tiết sẽ được lưu vào file `eval_results.csv`. Hệ thống sẽ so sánh giữa:
- **Dense**: Chỉ tìm kiếm bằng Vector.
- **Hybrid+Rerank**: Kết hợp Vector + BM25 + Cross-Encoder.

---
