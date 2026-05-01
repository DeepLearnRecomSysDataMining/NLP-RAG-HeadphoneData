import re
import pickle
from pyvi import ViTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import TOP_K_RETRIEVE, TOP_K_FINAL
from sharedutil import dedup_by_url, clean_vietnamese_text


def retrieve_dense(query, vector_store, k=TOP_K_RETRIEVE):
    clean_q = clean_vietnamese_text(query)
    # Lấy k*4 để sau dedup vẫn còn đủ k docs
    raw = vector_store.similarity_search(clean_q, k=k * 4)
    return dedup_by_url(raw)[:k]

def retrieve_hybrid(query, vector_store, bm25, all_docs, k_retrieve=50, verbose=False):
    """ Dedup dùng 'url' thay vì 'original_content'. Tách hàm riêng (chưa có reranking — xem rerank_candidates). Log số candidate dense/sparse.
    Đổi k_retrieve mặc định thành 50 (Lấy Rộng) để Reranker có đủ ứng viên chấm điểm thay vì chỉ lấy Top 5.
    """

    clean_q = clean_vietnamese_text(query)
    dense_docs = vector_store.similarity_search(clean_q, k=k_retrieve)
    # BM25 query cũng được tokenize chuẩn PyVi
    tokenized_query = clean_q.split()
    sparse_docs = bm25.get_top_n(tokenized_query, all_docs, n=k_retrieve)
    if verbose:
        print(f"   Dense candidates : {len(dense_docs)}")
        print(f"   Sparse candidates: {len(sparse_docs)}")

    # Dedup dùng URL (key duy nhất) thay vì original_content
    unique_docs = {}
    for doc in (dense_docs + sparse_docs):
        url_key = doc.metadata.get('url', doc.metadata.get('original_content', ''))
        if url_key not in unique_docs:
            unique_docs[url_key] = doc
    all_candidates = list(unique_docs.values())
    if verbose:
        print(f"   Sau dedup        : {len(all_candidates)} unique candidates")

    return all_candidates

def rerank_candidates(query, candidates, reranker, top_k=TOP_K_FINAL, verbose=False):
    """
    Sau reranking, cộng bonus brand/price từ metadata.
    Tách hàm riêng.
    Log điểm rerank + bonus.
    """
    if not candidates:
        print("⚠️  Không có candidates để rerank!")
        return []

    pairs = [[query, doc.metadata.get('original_content', doc.page_content)] for doc in candidates]
    rerank_scores = reranker.predict(pairs)

    # Tính metadata bonus
    query_lower = query.lower()
    for i, (score, doc) in enumerate(zip(rerank_scores, candidates)):
        bonus = 0.0
        # Brand bonus: cộng thêm nếu query đề cập đúng brand
        doc_brand = doc.metadata.get('brand', '').lower()
        if doc_brand and doc_brand != 'khác' and doc_brand in query_lower:
            bonus += 0.5
        # Price filter: trừ điểm nếu query có "dưới X triệu" mà giá sản phẩm cao hơn
        price_match = re.search(r'dưới\s*(\d+)\s*triệu', query_lower)
        if price_match:
            max_price = int(price_match.group(1)) * 1_000_000
            doc_price = doc.metadata.get('price', 0)
            if doc_price > max_price:
                bonus -= 1.0  # Phạt nếu lệch giá
        candidates[i].metadata['rerank_score'] = float(score) + bonus
        candidates[i].metadata['rerank_raw']   = float(score)
        candidates[i].metadata['meta_bonus']   = bonus

    # Sort toàn bộ trước
    sorted_all = sorted(candidates, key=lambda x: x.metadata['rerank_score'], reverse=True)
    # Dedup theo URL TRƯỚC khi cắt top_k — áp dụng cho mọi nơi gọi hàm này
    deduped = dedup_by_url(sorted_all)[:top_k]

    if verbose:
        print(f"   Rerank hoàn thành — Top {top_k} kết quả (sau dedup):")
        for d in deduped:
            print(f"     [{d.metadata['rerank_score']:.3f}] (raw={d.metadata['rerank_raw']:.3f}, " f"bonus={d.metadata['meta_bonus']:+.1f}) {d.metadata['product_name'][:40]}")
    return deduped

def rag_answer(query, vector_store, bm25, reranker, all_docs, retrieval_mode="hybrid"):
    """Pipeline RAG đầy đủ. GIỮ NGUYÊN cấu trúc output, cập nhật gọi hàm mới."""
    if retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, vector_store, bm25, all_docs)
        context_docs = rerank_candidates(query, candidates, reranker)
    else:
        context_docs = retrieve_dense(query, vector_store, k=TOP_K_FINAL)

    formatted_contexts = []
    for d in context_docs:
        product = d.metadata['product_name']
        price   = f"{d.metadata['price']:,} VNĐ" if d.metadata['price'] > 0 else "Liên hệ"
        chunk_str = f"Sản phẩm: {product} (Giá: {price})\nThông tin:\n{d.metadata['original_content']}"
        formatted_contexts.append(chunk_str)

    context_text = "\n" + "="*40 + "\n" + "\n".join(formatted_contexts) + "\n" + "="*40
    return {
        "answer":  "[Output từ LLM]",
        "context": context_text,
        "sources": [d.metadata.get('url', 'N/A') for d in context_docs]
    }