import os
import re
import pandas as pd
from pyvi import ViTokenizer
from config import BRAND_KEYWORDS

def clean_vietnamese_text(text):
    """Chuẩn hoá + tokenize tiếng Việt (PyVi). Dùng chung toàn bộ pipeline."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,?!/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return ViTokenizer.tokenize(text)

def extract_brand(text):
    text_lower = str(text).lower()
    for b in BRAND_KEYWORDS:
        if b in text_lower:
            return b
    return "khác"

def clean_price(price_str):
    if pd.isna(price_str):
        return 0
    price_num = re.sub(r'[^\d]', '', str(price_str))
    return int(price_num) if price_num else 0

def check_file_exists(path, label="File"):
    """Kiểm tra file tồn tại, in thông báo rõ ràng."""
    if not os.path.exists(path):
        print(f"[ERROR] KHÔNG TÌM THẤY {label}: {path}")
        return False
    print(f"[INFO] {label} tồn tại: {path}")
    return True

def check_metadata_fields(doc, required_fields):
    """Kiểm tra metadata của document có đủ trường không."""
    missing = [f for f in required_fields if f not in doc.metadata]
    if missing:
        print(f"[WARNING]  Metadata thiếu trường: {missing}")
        return False
    return True

def dedup_by_url(docs):
    """
    Giữ lại 1 document đại diện cho mỗi URL (chunk tốt nhất = chunk đầu tiên sau rerank). Tránh cùng 1 sản phẩm chiếm nhiều slot trong Top K.
    """
    seen_urls = {}
    for doc in docs:
        url = doc.metadata.get('url', doc.metadata.get('product_name', ''))
        if url not in seen_urls:
            seen_urls[url] = doc
    result = list(seen_urls.values())
    return result