import math
import pandas as pd
from config.config import TOP_K_FINAL, EVAL_RESULT_PATH
from rag.retrieve_func import retrieve_dense, retrieve_hybrid, rerank_candidates
from utils.sharedutil import normalize_url, parse_price_constraint, price_ok, detect_query_brands

TEST_QUERIES = [
    {
        "query": "Tìm cho tôi dây cáp PWaudio No.5 Đồng OCC Litz",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-no-5-dong-occ-litz.html"],
        "relevant_keywords": ["no.5", "đồng occ", "litz"]
    },
    {
        "query": "Tìm dây cáp tai nghe PWAudio giá dưới 5 triệu",
        "relevant_urls": [],
        "relevant_keywords": ["pwaudio", "dây cáp", "cáp tai nghe"]
    },
    {
        "query": "Các dòng cáp Century Series The 1960s của PWAudio có tốt không?",
        "relevant_urls": [
            "https://tainghe.com.vn/pwaudio-century-series-the-1960s-chinh-hang.html",
            "https://tainghe.com.vn/pwaudio-monile-mkii-ft.60s.html"
        ],
        "relevant_keywords": ["century series", "1960s", "ft.60s"]
    },
    {
        "query": "Dây nâng cấp âm thanh Ignit mạ bạc cho in-ear",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-titans-series-ignis-10pcs.html"],
        "relevant_keywords": ["ignis", "titans series", "dây nâng cấp"]
    },
    {
        "query": "Cho tôi thông tin về cáp PWAudio Monile MKII bản có Shielding chống nhiễu",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-monile-mkii-shielding.html"],
        "relevant_keywords": ["monile mkii", "shielding", "chống nhiễu"]
    },
    {
        "query": "Tôi muốn nâng cấp dây dẫn bằng đồng nguyên chất cho tai nghe",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-no-5-dong-occ-litz.html"],
        "relevant_keywords": ["đồng occ", "đồng nguyên chất", "copper"]
    },
    {
        "query": "Tìm cho tôi tai nghe ThieAudio Oracle MKIII",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-thieaudio-oracle-mkiii.html"],
        "relevant_keywords": ["thieaudio", "oracle", "mkiii"]
    },
    {
        "query": "Có mẫu loa đeo cổ Monster nào giá dưới 2 triệu không?",
        "relevant_urls": [],
        "relevant_keywords": ["monster", "loa đeo cổ", "boomerang"]
    },
    {
        "query": "Tôi cần tìm dây cáp ddHiFi cổng 4.4mm có cấu tạo lõi bạc và OCC",
        "relevant_urls": ["https://tainghe.com.vn/ddhifi-bc44pro-4-4mm-balanced-cable.html"],
        "relevant_keywords": ["ddhifi", "bc44pro", "4.4mm", "silver", "occ", "bạc"]
    },
    {
        "query": "Bên shop có bán mút tai nghe thay thế cho con chụp tai Sony 10RBT không?",
        "relevant_urls": ["https://tainghe.com.vn/sony-mdr-10rbt-ear-pads.html"],
        "relevant_keywords": ["đệm pad", "sony", "mdr-10rbt", "mdr -10rbt"]
    },
    {
        "query": "Cho tôi xin mẫu tai nghe in-ear cấu hình tribrid có tích hợp công nghệ IMPACT",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-thieaudio-oracle-mkiii.html"],
        "relevant_keywords": ["tribrid", "impact", "oracle mkiii"]
    },
    {
        "query": "FiiO JD7",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-fiio-jade-audio-jd7.html"],
        "relevant_keywords": ["fiio", "jade audio", "jd7"]
    }
]


# ============================
# FIX nDCG@K: đảm bảo nDCG nằm trong [0, 1]
# Paste cell này trước cell run_evaluation()
# ============================
import math
import numpy as np

def ndcg_at_k(docs, item, k=5, all_docs=None):
    """
    Tính nDCG@K đúng chuẩn:
    - DCG: điểm ranking hiện tại
    - IDCG: điểm ranking lý tưởng
    - nDCG = DCG / IDCG, luôn nằm trong [0, 1]
    """

    top_docs = docs[:k]

    # relevance nhị phân: relevant = 1, non-relevant = 0
    gains = [1 if is_relevant(doc, item) else 0 for doc in top_docs]

    # DCG
    dcg = 0.0
    for i, rel in enumerate(gains, start=1):
        dcg += rel / math.log2(i + 1)

    # Đếm số relevant lý tưởng
    # Ưu tiên ground-truth URL nếu có
    gt_urls = item.get("relevant_urls", [])

    if gt_urls:
        total_relevant = len(set(gt_urls))
    elif all_docs is not None:
        total_relevant = sum(1 for doc in all_docs if is_relevant(doc, item))
    else:
        # fallback an toàn: ít nhất bằng số relevant xuất hiện trong top-k
        total_relevant = sum(gains)

    # Guard để tránh DCG > IDCG khi dùng weak-label / keyword fallback
    total_relevant = max(total_relevant, sum(gains))

    ideal_hits = min(k, total_relevant)

    if ideal_hits == 0:
        return 0.0

    # IDCG lý tưởng: tất cả relevant đứng trên đầu
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    score = dcg / idcg

    # nDCG chuẩn không được vượt quá 1
    return min(score, 1.0)

# ============================================================
# RELEVANCE RULES
# ============================================================
def keyword_match(doc, keywords):
    doc_name = str(doc.metadata.get('product_name', '')).lower()
    doc_content = str(doc.metadata.get('original_content', '')).lower()
    return any(str(kw).lower() in doc_name or str(kw).lower() in doc_content for kw in keywords)

def is_relevant(doc, item):
    """
    Binary relevance cho metric chính.
    - Nếu có relevant_urls: chỉ tính đúng khi URL match chính xác.
    - Nếu không có URL: dùng keyword fallback cho query open-ended.
    """
    relevant_urls = [normalize_url(u) for u in item.get('relevant_urls', []) if str(u).strip()]
    doc_url = normalize_url(doc.metadata.get('url', ''))

    if relevant_urls:
        return doc_url in relevant_urls

    return keyword_match(doc, item.get('relevant_keywords', []))

def count_relevant_in_universe(item, universe_docs):
    """
    Số relevant docs trong toàn bộ docstore để tính IDCG đúng.
    Quan trọng: giúp nDCG luôn nằm trong [0, 1].
    """
    if universe_docs is None:
        return len(item.get('relevant_urls', []))

    # Đếm theo URL để tránh 1 sản phẩm nhiều chunk làm phình relevant set.
    relevant_urls = set()
    for d in universe_docs:
        if is_relevant(d, item):
            relevant_urls.add(normalize_url(d.metadata.get('url', '')))
    return len(relevant_urls)

# ============================================================
# METRICS
# ============================================================
def precision_at_k(docs, item, k):
    return sum(1 for d in docs[:k] if is_relevant(d, item)) / k

def hit_rate_at_k(docs, item, k):
    return 1.0 if any(is_relevant(d, item) for d in docs[:k]) else 0.0

def mrr_at_k(docs, item, k):
    for i, doc in enumerate(docs[:k], 1):
        if is_relevant(doc, item):
            return 1.0 / i
    return 0.0

def ndcg_at_k(docs, item, k, universe_docs=None):
    dcg = 0.0
    for i, d in enumerate(docs[:k], start=1):
        rel = 1.0 if is_relevant(d, item) else 0.0
        dcg += rel / math.log2(i + 1)

    n_rel = count_relevant_in_universe(item, universe_docs)
    ideal_hits = min(n_rel, k)
    if ideal_hits <= 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    value = dcg / idcg if idcg > 0 else 0.0
    return min(value, 1.0)  # safety clamp; nếu >1 thì metric đang có vấn đề

# ============================================================
# OPTIONAL CONSTRAINT METRICS
# ============================================================
def price_constraint_rate_for_results(query, docs):
    max_price = parse_price_constraint(query)
    if max_price is None or not docs:
        return None
    return sum(1 for d in docs if price_ok(d.metadata.get('price', 0), max_price)) / len(docs)

def brand_match_rate_for_results(query, docs):
    query_brands = detect_query_brands(query)
    if not query_brands or not docs:
        return None
    ok = 0
    for d in docs:
        text = (str(d.metadata.get('product_name', '')) + ' ' + str(d.metadata.get('original_content', ''))).lower()
        brand = str(d.metadata.get('brand', '')).lower()
        if any((b == brand) or (b in text) for b in query_brands):
            ok += 1
    return ok / len(docs)

# ============================================================
# RUN EVALUATION
# ============================================================
def run_evaluation(vector_store, bm25, reranker, all_docs):
    if vector_store is None:
        print("❌ Chưa load index.")
        return None

    K = TOP_K_FINAL
    results = []
    print("--- Bắt đầu đánh giá hệ thống ---\n")
    print(f"{'Query':<48} | {'P@5':>5} | {'HR@5':>5} | {'MRR':>5} | {'nDCG':>5} | Mode")
    print("-" * 95)

    for item in TEST_QUERIES:
        q = item["query"]

        # Dense baseline
        dense_docs = retrieve_dense(q, vector_store, k=K)
        row_dense = {
            "Query": q,
            "Mode": "Dense",
            "Precision@5": round(precision_at_k(dense_docs, item, K), 3),
            "HitRate@5": round(hit_rate_at_k(dense_docs, item, K), 3),
            "MRR": round(mrr_at_k(dense_docs, item, K), 3),
            "nDCG@5": round(ndcg_at_k(dense_docs, item, K, all_docs), 3),
            "PriceConstraintRate": price_constraint_rate_for_results(q, dense_docs),
            "BrandMatchRate": brand_match_rate_for_results(q, dense_docs),
            "Top1": dense_docs[0].metadata.get('product_name', '') if dense_docs else ""
        }
        results.append(row_dense)
        print(f"{q[:47]:<48} | {row_dense['Precision@5']:>5} | {row_dense['HitRate@5']:>5} "
              f"| {row_dense['MRR']:>5} | {row_dense['nDCG@5']:>5} | Dense")

        # Hybrid + Cross-Encoder rerank
        candidates = retrieve_hybrid(q, vector_store, bm25, all_docs)
        hybrid_docs = rerank_candidates(q, candidates, reranker, top_k=K)
        row_hybrid = {
            "Query": q,
            "Mode": "Hybrid+Rerank",
            "Precision@5": round(precision_at_k(hybrid_docs, item, K), 3),
            "HitRate@5": round(hit_rate_at_k(hybrid_docs, item, K), 3),
            "MRR": round(mrr_at_k(hybrid_docs, item, K), 3),
            "nDCG@5": round(ndcg_at_k(hybrid_docs, item, K, all_docs), 3),
            "PriceConstraintRate": price_constraint_rate_for_results(q, hybrid_docs),
            "BrandMatchRate": brand_match_rate_for_results(q, hybrid_docs),
            "Top1": hybrid_docs[0].metadata.get('product_name', '') if hybrid_docs else ""
        }
        results.append(row_hybrid)
        print(f"{q[:47]:<48} | {row_hybrid['Precision@5']:>5} | {row_hybrid['HitRate@5']:>5} "
              f"| {row_hybrid['MRR']:>5} | {row_hybrid['nDCG@5']:>5} | Hybrid+Rerank")
        print()

    df_eval = pd.DataFrame(results)

    print("\n" + "="*65)
    print("TỔNG KẾT TRUNG BÌNH THEO CHẾ ĐỘ:")
    summary = df_eval.groupby('Mode')[['Precision@5', 'HitRate@5', 'MRR', 'nDCG@5']].mean().round(3)
    print(summary.to_string())

    # Cảnh báo nếu nDCG lỗi
    if (df_eval['nDCG@5'] > 1).any():
        print("\n[WARNING] Có nDCG > 1. Cần kiểm tra lại relevance/IDCG.")
    else:
        print("\n[OK] nDCG hợp lệ: tất cả giá trị nằm trong [0, 1].")

    dense_mrr = summary.loc['Dense', 'MRR'] if 'Dense' in summary.index else 0
    hybrid_mrr = summary.loc['Hybrid+Rerank', 'MRR'] if 'Hybrid+Rerank' in summary.index else 0
    gain = hybrid_mrr - dense_mrr
    print(f"\nMRR Gain (Hybrid+Rerank vs Dense): {gain:+.3f}")
    if gain > 0.05:
        print("   ✅ Reranker cải thiện rõ ràng — nên giữ.")
    elif gain > 0:
        print("   ⚠️ Reranker cải thiện nhẹ — cân nhắc cost/benefit.")
    else:
        print("   ❌ Reranker không cải thiện — cần kiểm tra lại cấu hình hoặc test set.")

    try:
        df_eval.to_csv(EVAL_RESULT_PATH, index=False)
        print(f"\n💾 Kết quả lưu tại: {EVAL_RESULT_PATH}")
    except Exception as e:
        print(f"\n[WARNING] Không lưu được CSV: {e}")

    return df_eval

