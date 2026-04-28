import os
import re
import pickle
import numpy as np
from pyvi import ViTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 1. CẤU HÌNH ĐỒNG BỘ VỚI INDEX.PY
INDEX_FOLDER = "faiss_index_v2"
DOCSTORE_PATH = "docstore_v2.pkl"
EMBED_MODEL_NAME = 'keepitreal/vietnamese-sbert'
RERANK_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'  # Mô hình rerank hỗ trợ Tiếng Việt

TOP_K_RETRIEVE = 15  # Lấy rộng để rerank
TOP_K_FINAL = 3  # Lấy hẹp để đưa vào LLM

# 2. HÀM TIỀN XỬ LÝ (Giống hệt index.py)
def clean_vietnamese_text(text):
    """Tiền xử lý câu hỏi của user để khớp với dữ liệu trong FAISS & BM25"""
    if not isinstance(text, str): return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,?!/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return ViTokenizer.tokenize(text)

# 3. HÀM TẢI TÀI NGUYÊN (Khởi tạo hệ thống)
def load_search_resources():
    print("⏳ Đang tải mô hình Embedding và Reranker...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    with open(DOCSTORE_PATH, "rb") as f:
        all_docs = pickle.load(f)

    print("⏳ Đang khởi tạo bộ máy BM25...")
    tokenized_corpus = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    reranker = CrossEncoder(RERANK_MODEL_NAME)
    print("✅ Hệ thống Retrieval sẵn sàng!")

    return vector_store, bm25, reranker, all_docs


# 4. HÀM TÌM KIẾM LAI (RETRIEVAL)
def retrieve_hybrid(query, vector_store, bm25, reranker, all_docs):
    clean_q = clean_vietnamese_text(query)
    # 1. DENSE SEARCH (FAISS)
    dense_docs = vector_store.similarity_search(clean_q, k=TOP_K_RETRIEVE)
    # 2. SPARSE SEARCH (BM25)
    tokenized_query = clean_q.split()
    sparse_docs = bm25.get_top_n(tokenized_query, all_docs, n=TOP_K_RETRIEVE)
    # 3. GỘP VÀ LỌC TRÙNG LẶP
    unique_docs = {}
    for doc in (dense_docs + sparse_docs):
        content_key = doc.metadata['original_content']
        if content_key not in unique_docs:
            unique_docs[content_key] = doc
    all_candidates = list(unique_docs.values())
    # 4. RERANKING BẰNG CROSS-ENCODER
    pairs = [[query, doc.metadata['original_content']] for doc in all_candidates]
    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        all_candidates[i].metadata['rerank_score'] = float(score)
    sorted_docs = sorted(all_candidates, key=lambda x: x.metadata['rerank_score'], reverse=True)
    return sorted_docs[:TOP_K_FINAL]

# 5. HÀM ĐÓNG GÓI DỮ LIỆU (CONTEXT)
def rag_answer(query, vector_store, bm25, reranker, all_docs, retrieval_mode="hybrid"):
    if retrieval_mode == "dense":
        clean_q = clean_vietnamese_text(query)
        context_docs = vector_store.similarity_search(clean_q, k=TOP_K_FINAL)
    else:
        context_docs = retrieve_hybrid(query, vector_store, bm25, reranker, all_docs)

    formatted_contexts = []
    for d in context_docs:
        product = d.metadata['product_name']
        price = f"{d.metadata['price']:,} VNĐ" if d.metadata['price'] > 0 else "Liên hệ"
        content = d.metadata['original_content']
        chunk_str = f"Sản phẩm: {product} (Giá: {price})\nThông tin chi tiết:\n{content}"
        formatted_contexts.append(chunk_str)

    context_text = "\n" + "=" * 40 + "\n" + "\n".join(formatted_contexts) + "\n" + "=" * 40

    return {
        "answer": "[Khu vực này dành cho output của API OpenAI/Gemini]",
        "context": context_text,
        "sources": [d.metadata.get('url', 'N/A') for d in context_docs]
    }

# 6. HÀM CHẠY CHƯƠNG TRÌNH CHÍNH (TESTING)
def run_test_interactive():
    vector_store, bm25, reranker, all_docs = load_search_resources()
    while True:
        user_query = input("  Nhập câu hỏi tìm kiếm (hoặc gõ 'exit' để thoát): ")
        if user_query.lower() in ['exit', 'quit', 'thoát']:
            print("  Tạm biệt!")
            break
        if not user_query.strip():
            continue
        print("\n  Đang tìm kiếm...")
        results = retrieve_hybrid(user_query, vector_store, bm25, reranker, all_docs)
        print(f"\n  TÌM THẤY {len(results)} KẾT QUẢ TỐT NHẤT:")
        for idx, doc in enumerate(results, 1):
            product = doc.metadata.get('product_name', 'N/A')
            price = doc.metadata.get('price', 0)
            price_str = f"{price:,} VNĐ" if price > 0 else "Liên hệ"
            score = doc.metadata.get('rerank_score', 0)
            content = doc.metadata.get('original_content', '')
            print(f"[{idx}] SẢN PHẨM: {product}")
            print(f"    💰 Giá: {price_str}")
            print(f"    📈 Điểm Rerank: {score:.2f}")
            print(f"    📄 Trích đoạn: {content[:200]}...")
            print("-" * 40)

if __name__ == "__main__":
    run_test_interactive()