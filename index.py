import os
import pandas as pd
import re
import pickle
from pathlib import Path
from pyvi import ViTokenizer

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from tqdm import tqdm

# =============================================================================
# 1. CẤU HÌNH VÀ KHỞI TẠO
# =============================================================================
RAW_DATA_PATH = "full_xuanvu_database.csv"
INDEX_FOLDER = "faiss_index_v2"
DOCSTORE_PATH = "docstore_v2.pkl"
MODEL_NAME = 'keepitreal/vietnamese-sbert'

# Danh sách thương hiệu để trích xuất metadata
BRAND_KEYWORDS = ["sony", "jbl", "sennheiser", "soundpeats", "anker", "edifier",
                  "final", "fiio", "moondrop", "soundmagic", "akg", "bose",
                  "apple", "marshall", "hifiman"]


# =============================================================================
# 2. HÀM TIỀN XỬ LÝ (Tích hợp từ code cũ của bạn)
# =============================================================================
def extract_brand(text):
    text_lower = str(text).lower()
    for b in BRAND_KEYWORDS:
        if b in text_lower:
            return b
    return "khác"

def clean_vietnamese_text(text):
    if not isinstance(text, str): return ""
    # Loại bỏ ký tự thừa
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,?!/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # Word Segmentation (quan trọng cho Vietnamese-SBERT)
    return ViTokenizer.tokenize(text)

def clean_price(price_str):
    if pd.isna(price_str): return 0
    price_num = re.sub(r'[^\d]', '', str(price_str))
    return int(price_num) if price_num else 0

# 3. PIPELINE XỬ LÝ CHÍNH (Đã sửa lỗi logic Chunking)
def build_or_load_index(force_rebuild=False):
    if os.path.exists(INDEX_FOLDER) and not force_rebuild:
        print(f"✅ Tìm thấy Index cũ tại '{INDEX_FOLDER}'. Đang tải...")
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
        with open(DOCSTORE_PATH, "rb") as f:
            all_docs = pickle.load(f)
        return vector_store, all_docs

    print("⏳ Đang tiền xử lý dữ liệu với chiến lược: Chunk gốc -> Clean text...")
    df = pd.read_csv(RAW_DATA_PATH)
    df['price_num'] = df['price'].apply(clean_price)
    df['brand'] = df['product_name'].apply(extract_brand)

    # Text Splitter giờ đây sẽ hoạt động hoàn hảo vì nó nhận text chưa bị xóa \n
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step 1/2: Chunking & Cleaning"):
        raw_name, raw_review = str(row['product_name']), str(row['review_content'])
        full_raw_content = f"{raw_name}\n\n{raw_review}"
        raw_chunks = text_splitter.split_text(full_raw_content)

        for chunk in raw_chunks:
            clean_chunk = clean_vietnamese_text(chunk)
            if clean_chunk.strip():
                documents.append(
                    Document(
                        page_content=clean_chunk,
                        metadata={
                            "original_content": chunk,
                            "product_name": row['product_name'],
                            "brand": row['brand'],
                            "price": row['price_num'],
                            "url": row['url']
                        }
                    )
                )
    # for _, row in df.iterrows():
    #     raw_name = str(row['product_name'])
    #     raw_review = str(row['review_content'])
    #     # Gắn tên sản phẩm vào đầu mỗi bài review để không bị mất ngữ cảnh
    #     full_raw_content = f"{raw_name}\n\n{raw_review}"
    #     # 1. BƯỚC CHUNKING TRÊN TEXT GỐC
    #     raw_chunks = text_splitter.split_text(full_raw_content)
    #
    #     # 2. BƯỚC LÀM SẠCH VÀ TOKENIZE TỪNG CHUNK
    #     for chunk in raw_chunks:
    #         # Clean text riêng cho mô hình vietnamese-sbert
    #         clean_chunk = clean_vietnamese_text(chunk)
    #         # Bỏ qua các chunk rỗng sau khi clean
    #         if not clean_chunk.strip():
    #             continue
    #         doc = Document(
    #             page_content=clean_chunk,  # FAISS sẽ dùng text này (có ViTokenizer) để tạo vector
    #             metadata={
    #                 "original_content": chunk,  # QUAN TRỌNG: LLM sẽ đọc text này (giữ nguyên \n và không có dấu _)
    #                 "product_name": row['product_name'],
    #                 "brand": row['brand'],
    #                 "price": row['price_num'],
    #                 "url": row['url']
    #             }
    #         )
    #         documents.append(doc)
    print(f"\n📦 Đã tạo {len(documents)} chunks chuẩn. Đang tiến hành Vector hóa...")

    # Tách dữ liệu để xử lý lô (batching)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    # Gọi SentenceTransformer nguyên thủy để tối đa tốc độ phần cứng
    fast_embedder = SentenceTransformer(MODEL_NAME)
    # Biến show_progress_bar=True sẽ kích hoạt tqdm bar hiển thị phần % hoàn thành và tốc độ (it/s)
    embeddings_matrix = fast_embedder.encode(texts, batch_size=128, show_progress_bar=True)
    print("\n⏳ Đang đóng gói dữ liệu vào cấu trúc LangChain FAISS...")
    # Ghép lại thành cặp (văn_bản, vector)
    text_embedding_pairs = list(zip(texts, embeddings_matrix.tolist()))
    # Khởi tạo lớp vỏ Langchain để FAISS biết dùng mô hình nào lúc search
    lc_embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    # Nạp thẳng dữ liệu đã xử lý vào FAISS
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=lc_embedder,
        metadatas=metadatas
    )
    # Lưu lại
    vector_store.save_local(INDEX_FOLDER)
    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f" Hoàn thành! Index đã được lưu tại '{INDEX_FOLDER}'.")
    return vector_store, documents

# 4. HÀM KHÁM PHÁ DỮ LIỆU (EDA)
def run_eda():
    print("\n" + "=" * 50)
    print(" BẮT ĐẦU KHÁM PHÁ DỮ LIỆU (EDA)")
    print("=" * 50)

    if not os.path.exists(RAW_DATA_PATH):
        print(f" Không tìm thấy file {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    # Tiền xử lý nhanh để EDA
    df['price_num'] = df['price'].apply(clean_price)
    df['brand'] = df['product_name'].apply(extract_brand)
    df['review_length'] = df['review_content'].astype(str).apply(len)

    # 1. In Thống kê cơ bản
    print(f"Tổng số sản phẩm: {len(df)}")
    print(f"Số lượng thương hiệu duy nhất: {df['brand'].nunique()}")
    print(f"Giá trung bình: {df[df['price_num'] > 0]['price_num'].mean():,.0f} VNĐ")
    print(f"Độ dài bài review trung bình: {df['review_length'].mean():.0f} ký tự")
    print("-" * 50)

    # Cấu hình biểu đồ
    sns.set_theme(style="whitegrid")

    # 2. Biểu đồ Top Thương hiệu
    plt.figure(figsize=(10, 5))
    top_brands = df[df['brand'] != 'khác']['brand'].value_counts().head(10)
    sns.barplot(x=top_brands.values, y=top_brands.index, hue=top_brands.index, palette='viridis', legend=False)
    plt.title('Top 10 Thương hiệu có nhiều sản phẩm nhất', fontsize=14, pad=15)
    plt.xlabel('Số lượng sản phẩm')
    plt.ylabel('Thương hiệu')
    plt.tight_layout()
    plt.show()

    # 3. Biểu đồ phân phối Giá
    plt.figure(figsize=(10, 5))
    valid_prices = df[df['price_num'] > 0]['price_num']
    sns.histplot(valid_prices, bins=40, kde=True, color='#1f77b4')
    plt.title('Phân phối Giá sản phẩm (VNĐ)', fontsize=14, pad=15)
    plt.xlabel('Giá (VNĐ)')
    plt.ylabel('Số lượng')

    # Format trục X hiển thị số có dấu phẩy (vd: 10,000,000)
    formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    plt.gca().xaxis.set_major_formatter(formatter)

    # Giới hạn trục X ở percentile 95% để loại bỏ các outliners quá đắt làm biến dạng biểu đồ
    plt.xlim(0, valid_prices.quantile(0.95))
    plt.tight_layout()
    plt.show()

    print(" Hoàn thành phân tích EDA!\n")

def main():
    # Tự động check và chạy
    index_exists = os.path.exists(INDEX_FOLDER) and os.path.exists(DOCSTORE_PATH)
    if not index_exists:
        print(" Hệ thống: Chưa tìm thấy dữ liệu. Khởi tạo cơ sở dữ liệu vector...")
        build_or_load_index()
    else:
        print(" Hệ thống: Đã có Index trên Drive.")
    run_eda()
if __name__ == '__main__':
    main()