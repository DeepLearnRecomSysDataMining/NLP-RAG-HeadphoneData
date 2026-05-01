from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import pandas as pd

from config import RAW_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_IF_LONGER_THAN
from sharedutil import check_file_exists, clean_price, extract_brand, clean_vietnamese_text


def prepare_data():
    """
    Tách riêng bước prepare.
    1 sản phẩm = 1 document mặc định. Chỉ chunk nếu review_content > CHUNK_IF_LONGER_THAN ký tự.
    """
    if not check_file_exists(RAW_DATA_PATH, "RAW_DATA"):
        return None

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"📂 Đọc dữ liệu: {len(df)} sản phẩm")
    print(f"   Các cột: {list(df.columns)}")

    # SỬA ĐIỂM 9: Kiểm tra dữ liệu rỗng
    required_cols = ['product_name', 'review_content', 'price', 'url']
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Thiếu cột bắt buộc: {col}")
            return None
    df = df.dropna(subset=['product_name', 'review_content'])
    print(f"   Sau khi lọc NaN: {len(df)} sản phẩm hợp lệ")

    df['price_num'] = df['price'].apply(clean_price)
    df['brand']     = df['product_name'].apply(extract_brand)

    # Splitter dùng khi review quá dài
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    n_chunked = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing documents"):
        raw_name    = str(row['product_name'])
        raw_review  = str(row['review_content'])
        raw_price   = str(row['price'])           # giá dạng chuỗi gốc, dễ đọc
        raw_brand   = str(row['brand'])
        price_num   = row['price_num']
        url         = str(row.get('url', ''))

        # ── THAY ĐỔI CHÍNH ───
        # Nhúng brand + price vào page_content để FAISS & BM25 đều thấy. Cấu trúc rõ ràng giúp BM25 match keyword đúng hơn.
        structured_raw = (
            f"Sản phẩm: {raw_name}. "
            f"Thương hiệu: {raw_brand}. "
            f"Giá: {raw_price}. "
            f"{raw_review}"
        )
        metadata_base = {
            "product_name":    raw_name,
            "brand":           raw_brand,
            "price":           price_num,
            "url":             url,
            "original_content": structured_raw   # lưu bản gốc chưa clean để hiển thị
        }

        # SỬA Chỉ chunk nếu review quá dài
        if len(raw_review) > CHUNK_IF_LONGER_THAN:
            n_chunked += 1
            raw_chunks = text_splitter.split_text(structured_raw)
            for chunk in raw_chunks:
                # SỬA ĐIỂM 2: Dùng clean_vietnamese_text chung
                clean_chunk = clean_vietnamese_text(chunk)
                if clean_chunk.strip():
                    documents.append(Document(
                        page_content=clean_chunk,
                        metadata={**metadata_base, "original_content": chunk}
                    ))
        else:
            # 1 sản phẩm = 1 document
            clean_doc = clean_vietnamese_text(structured_raw)
            if clean_doc.strip():
                documents.append(Document(
                    page_content=clean_doc,
                    metadata=metadata_base
                ))

    # SỬA ĐIỂM 9: Log sau khi chuẩn bị
    print(f"\n[INFO] Chuẩn bị xong:")
    print(f"   Tổng documents: {len(documents)}")
    print(f"   Sản phẩm được chunk (review dài): {n_chunked}")
    print(f"   Sản phẩm giữ nguyên (1 doc): {len(df) - n_chunked}")

    if documents:
        sample = documents[0]
        print(f"\n   [SAMPLE page_content đầu tiên]:")
        print(f"   {sample.page_content[:200]}...")
        print(f"   metadata keys: {list(sample.metadata.keys())}")

    return documents