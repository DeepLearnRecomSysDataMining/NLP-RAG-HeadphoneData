import re

from config.config import TOP_K_RETRIEVE, TOP_K_FINAL
from utils.sharedutil import dedup_by_url, clean_vietnamese_text, price_ok, detect_query_brands, parse_price_constraint


def retrieve_dense(query, vector_store, k=TOP_K_RETRIEVE):
    clean_q = clean_vietnamese_text(query)
    # Lấy k*4 để sau dedup vẫn còn đủ k docs
    raw = vector_store.similarity_search(clean_q, k=k * 4)
    return dedup_by_url(raw)[:k]

def retrieve_hybrid(query, vector_store, bm25, all_docs, k_retrieve=50, verbose=False):
    """
    SỬA ĐIỂM 4: Dedup dùng 'url' thay vì 'original_content'.
    SỬA ĐIỂM 7: Tách hàm riêng (chưa có reranking — xem rerank_candidates).
    SỬA ĐIỂM 9: Log số candidate dense/sparse.
    SỬA LỖI HIỆU SUẤT: Đổi k_retrieve mặc định thành 50 (Lấy Rộng) để Reranker có đủ ứng viên chấm điểm thay vì chỉ lấy Top 5.
    """
    # SỬA ĐIỂM 2: Dùng clean_vietnamese_text chung
    clean_q = clean_vietnamese_text(query)

    # Dense Search
    dense_docs = vector_store.similarity_search(clean_q, k=k_retrieve)

    # SỬA ĐIỂM 8: BM25 query cũng được tokenize chuẩn PyVi
    tokenized_query = clean_q.split()
    sparse_docs = bm25.get_top_n(tokenized_query, all_docs, n=k_retrieve)

    if verbose:
        print(f"   Dense candidates : {len(dense_docs)}")
        print(f"   Sparse candidates: {len(sparse_docs)}")

    # SỬA ĐIỂM 4: Dedup dùng URL (key duy nhất) thay vì original_content
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
    Cross-encoder rerank + metadata-aware scoring.
    Điểm cuối = điểm reranker + brand bonus - price penalty.
    """
    if not candidates:
        print("⚠️ Không có candidates để rerank!")
        return []

    pairs = [[query, doc.metadata.get('original_content', doc.page_content)] for doc in candidates]
    rerank_scores = reranker.predict(pairs)

    query_brands = detect_query_brands(query)
    max_price = parse_price_constraint(query)

    for i, (score, doc) in enumerate(zip(rerank_scores, candidates)):
        bonus = 0.0

        # Brand bonus: nếu query nhắc brand và document đúng brand.
        doc_brand = str(doc.metadata.get('brand', '')).lower()
        doc_text = (str(doc.metadata.get('product_name', '')) + ' ' + str(doc.metadata.get('original_content', ''))).lower()
        if query_brands:
            if any((b == doc_brand) or (b in doc_text) for b in query_brands):
                bonus += 0.5
            else:
                # phạt nhẹ nếu query yêu cầu brand rõ nhưng doc không match brand
                bonus -= 0.2

        # Price penalty: nếu query có điều kiện giá mà sản phẩm vượt giá.
        doc_price = doc.metadata.get('price', 0)
        if max_price is not None and not price_ok(doc_price, max_price):
            bonus -= 1.0

        doc.metadata['rerank_score'] = float(score) + bonus
        doc.metadata['rerank_raw'] = float(score)
        doc.metadata['meta_bonus'] = bonus
        doc.metadata['detected_max_price'] = max_price
        doc.metadata['detected_query_brands'] = query_brands

    sorted_all = sorted(candidates, key=lambda x: x.metadata.get('rerank_score', -999), reverse=True)
    deduped = dedup_by_url(sorted_all)[:top_k]

    if verbose:
        print(f"   Rerank hoàn thành — Top {top_k} kết quả:")
        for d in deduped:
            print(f"     [{d.metadata.get('rerank_score', 0):.3f}] raw={d.metadata.get('rerank_raw', 0):.3f}, "
                  f"bonus={d.metadata.get('meta_bonus', 0):+.1f} | {d.metadata.get('product_name', '')[:60]}")

    return deduped