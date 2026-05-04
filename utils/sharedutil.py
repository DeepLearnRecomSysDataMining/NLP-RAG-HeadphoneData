import os
import re
import pandas as pd
from pyvi import ViTokenizer
from config.config import BRAND_KEYWORDS

# ─────────────────────────────────────────────
# SỬA ĐIỂM 2: Hàm duy nhất, dùng chung cho build index / query / evaluation
# ─────────────────────────────────────────────
def clean_vietnamese_text(text):
    """Chuẩn hoá + tokenize tiếng Việt (PyVi). Dùng chung toàn bộ pipeline."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,?!/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return ViTokenizer.tokenize(text)

def normalize_url(url):
    """Chuẩn hoá URL để so khớp ground-truth ổn định hơn."""
    url = str(url or '').strip().lower()
    url = url.split('?')[0].split('#')[0].rstrip('/')
    if url.endswith('.html'):
        url = url[:-5]
    return url

def clean_price(price_str):
    """Chuyển chuỗi giá về số nguyên."""
    if pd.isna(price_str):
        return 0
    price_num = re.sub(r'[^\d]', '', str(price_str))
    return int(price_num) if price_num else 0

def extract_brand(text):
    """Trích thương hiệu từ text bằng danh sách brand; ưu tiên brand dài trước."""
    text_lower = str(text or '').lower()
    for b in sorted(BRAND_KEYWORDS, key=len, reverse=True):
        if b in text_lower:
            return b
    return "khác"

def detect_query_brands(query):
    """Tìm các brand được nhắc trong query."""
    q = str(query or '').lower()
    return [b for b in sorted(set(BRAND_KEYWORDS), key=len, reverse=True) if b in q]

def parse_price_constraint(query):
    """
    Parse điều kiện giá đơn giản từ query.
    Hỗ trợ: dưới 500k, dưới 1tr, dưới 1 triệu, tầm/khoảng 2.5 triệu, 2,5 triệu đổ lại.
    Trả về max_price dạng int VND hoặc None.
    """
    q = str(query or '').lower().replace(',', '.')
    patterns = [
        r'(?:dưới|duoi|<=|không quá|khong qua|ít hơn|it hon|tối đa|toi da)\s*(\d+(?:\.\d+)?)\s*(triệu|tr|m|k|ngàn|nghìn)',
        r'(?:tầm|tam|khoảng|khoang|around)\s*(\d+(?:\.\d+)?)\s*(triệu|tr|m|k|ngàn|nghìn)',
        r'(\d+(?:\.\d+)?)\s*(triệu|tr|m|k|ngàn|nghìn)\s*(?:đổ lại|do lai|trở xuống|tro xuong)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ['triệu', 'tr', 'm']:
            return int(val * 1_000_000)
        if unit in ['k', 'ngàn', 'nghìn']:
            return int(val * 1_000)
    return None

def price_ok(doc_price, max_price):
    """Kiểm tra giá có thỏa max_price không. Nếu không có constraint thì True."""
    if max_price is None:
        return True
    try:
        p = int(doc_price)
    except Exception:
        p = 0
    return p > 0 and p <= max_price

# ─────────────────────────────────────────────
# SỬA ĐIỂM 9: Helper kiểm tra guard chung
# ─────────────────────────────────────────────
def check_file_exists(path, label="File"):
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