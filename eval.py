import math
import pandas as pd
from config import TOP_K_FINAL, EVAL_RESULT_PATH
from rag_answer import retrieve_dense, retrieve_hybrid, rerank_candidates

TEST_QUERIES = [
    # 1. Truy vấn khớp chính xác tên sản phẩm (Kiểm tra Keyword/BM25)
    {
        "query": "Tìm cho tôi dây cáp PWaudio No.5 Đồng OCC Litz",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-no-5-dong-occ-litz.html"],
        "relevant_keywords": ["no.5", "đồng occ", "litz"]
    },

    # 2. Truy vấn chứa điều kiện về giá và thương hiệu (Kiểm tra logic Metadata Bonus đã viết)
    {
        "query": "Tìm dây cáp tai nghe PWAudio giá dưới 5 triệu",
        "relevant_urls": [], # Bỏ trống URL để code dùng keyword fallback
        "relevant_keywords": ["pwaudio", "dây cáp", "cáp tai nghe"]
    },

    # 3. Truy vấn tìm theo dòng sản phẩm (Series) (Kiểm tra Vector Search & Reranker)
    {
        "query": "Các dòng cáp Century Series The 1960s của PWAudio có tốt không?",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-century-series-the-1960s-chinh-hang.html",
                          "https://tainghe.com.vn/pwaudio-monile-mkii-ft.60s.html"],
        "relevant_keywords": ["century series", "1960s", "ft.60s"]
    },

    # 4. Truy vấn sai chính tả nhẹ / Dùng từ đồng nghĩa (Kiểm tra Embedding Vector)
    {
        "query": "Dây nâng cấp âm thanh Ignit mạ bạc cho in-ear",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-titans-series-ignis-10pcs.html"],
        "relevant_keywords": ["ignis", "titans series", "dây nâng cấp"]
    },

    # 5. Truy vấn so sánh hoặc tìm tính năng cụ thể (Kiểm tra Reranker đọc hiểu ngữ cảnh)
    {
        "query": "Cho tôi thông tin về cáp PWAudio Monile MKII bản có Shielding (chống nhiễu)",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-monile-mkii-shielding.html"],
        "relevant_keywords": ["monile mkii", "shielding", "chống nhiễu"]
    },

    # 6. Truy vấn mở rộng, không chỉ đích danh sản phẩm (Kiểm tra Semantic Search)
    {
        "query": "Tôi muốn nâng cấp dây dẫn bằng đồng nguyên chất cho tai nghe",
        "relevant_urls": ["https://tainghe.com.vn/pwaudio-no-5-dong-occ-litz.html"],
        "relevant_keywords": ["đồng occ", "đồng nguyên chất", "copper"]
    },

    # 7. Truy vấn khớp chính xác tên sản phẩm (Kiểm tra Keyword & Exact Match)
    {
        "query": "Tìm cho tôi tai nghe ThieAudio Oracle MKIII",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-thieaudio-oracle-mkiii.html"],
        "relevant_keywords": ["thieaudio", "oracle", "mkiii"]
    },

    # 8. Truy vấn kết hợp loại sản phẩm, thương hiệu và điều kiện giá (Kiểm tra Metadata Bonus & Keyword Fallback)
    {
        "query": "Có mẫu loa đeo cổ Monster nào giá dưới 2 triệu không?",
        "relevant_urls": [], # Bỏ trống để test tính năng fallback từ khóa và filter giá
        "relevant_keywords": ["monster", "loa đeo cổ", "boomerang"]
    },

    # 9. Truy vấn theo thông số kỹ thuật/vật liệu cụ thể (Kiểm tra Vector/Semantic Search)
    {
        "query": "Tôi cần tìm dây cáp ddHiFi cổng 4.4mm có cấu tạo lõi bạc và OCC",
        "relevant_urls": ["https://tainghe.com.vn/ddhifi-bc44pro-4-4mm-balanced-cable"],
        "relevant_keywords": ["ddhifi", "bc44pro", "4.4mm", "silver", "occ", "bạc"]
    },

    # 10. Truy vấn dùng từ lóng/từ đồng nghĩa của khách hàng (Kiểm tra độ thông minh của LLM Query Rewrite hoặc Embedding)
    {
        "query": "Bên shop có bán mút tai nghe thay thế cho con chụp tai Sony 10RBT không?",
        "relevant_urls": ["https://tainghe.com.vn/sony-mdr-10rbt-ear-pads.html"],
        "relevant_keywords": ["đệm pad", "sony", "mdr-10rbt", "mdr -10rbt"]
    },

    # 11. Truy vấn chuyên sâu dựa vào nội dung bài viết đánh giá (Kiểm tra Reranker đọc hiểu văn bản dài)
    {
        "query": "Cho tôi xin mẫu tai nghe in-ear cấu hình tribrid có tích hợp công nghệ IMPACT",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-thieaudio-oracle-mkiii.html"],
        "relevant_keywords": ["tribrid", "impact", "oracle mkiii"]
    },

    # 12. Truy vấn siêu ngắn (Short-tail query - Thử thách khả năng xếp hạng trực tiếp)
    {
        "query": "FiiO JD7",
        "relevant_urls": ["https://tainghe.com.vn/tai-nghe-fiio-jade-audio-jd7.html"],
        "relevant_keywords": ["fiio", "jade audio", "jd7"]
    }
]

def is_relevant(doc, item):
    doc_url  = doc.metadata.get('url', '').lower().strip().rstrip('/')
    doc_name = doc.metadata.get('product_name', '').lower()
    doc_content = doc.metadata.get('original_content', '').lower()

    relevant_urls = [u.lower().strip().rstrip('/') for u in item.get('relevant_urls', [])]
    if relevant_urls:
        # Có URL → chỉ match URL, không fallback keyword (tránh false positive)
        return doc_url in relevant_urls

    # Không có URL → keyword fallback cho open-ended query
    return any(kw in doc_name or kw in doc_content for kw in item.get('relevant_keywords', []))

def precision_at_k(docs, item, k):
    if not docs: return 0.0
    return sum(1 for d in docs[:k] if is_relevant(d, item)) / k

def hit_rate_at_k(docs, item, k):
    if not docs: return 0.0
    return 1.0 if any(is_relevant(d, item) for d in docs[:k]) else 0.0

def mrr(docs, item, k):
    if not docs: return 0.0
    for i, doc in enumerate(docs[:k], 1):
        if is_relevant(doc, item):
            return 1.0 / i
    return 0.0

def ndcg_at_k(docs, item, k):
    if not docs: return 0.0
    dcg  = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(docs[:k]) if is_relevant(d, item))
    n_rel = len(item.get('relevant_urls', [])) or 1
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(vector_store, bm25, reranker, all_docs):
    if vector_store is None:
        print("❌ Chưa load index.")
        return None

    K = TOP_K_FINAL
    results = []
    print("\n" + "="*90)
    print("--- BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG ---")
    print(f"{'Query':<48} | {'P@5':>5} | {'HR@5':>5} | {'MRR':>5} | {'nDCG':>5} | Mode")
    print("-" * 90)

    for item in TEST_QUERIES:
        q = item["query"]

        # 1. Dense evaluation
        dense_docs = retrieve_dense(q, vector_store, k=K)
        row_dense = {
            "Query": q, "Mode": "Dense",
            "Precision@5": round(precision_at_k(dense_docs, item, K), 3),
            "HitRate@5":   round(hit_rate_at_k(dense_docs, item, K), 3),
            "MRR":         round(mrr(dense_docs, item, K), 3),
            "nDCG@5":      round(ndcg_at_k(dense_docs, item, K), 3),
        }
        results.append(row_dense)
        print(f"{q[:47]:<48} | {row_dense['Precision@5']:>5} | {row_dense['HitRate@5']:>5} "
              f"| {row_dense['MRR']:>5} | {row_dense['nDCG@5']:>5} | Dense")

        # 2. Hybrid + Rerank evaluation
        candidates  = retrieve_hybrid(q, vector_store, bm25, all_docs)
        hybrid_docs = rerank_candidates(q, candidates, reranker, top_k=K)
        row_hybrid = {
            "Query": q, "Mode": "Hybrid+Rerank",
            "Precision@5": round(precision_at_k(hybrid_docs, item, K), 3),
            "HitRate@5":   round(hit_rate_at_k(hybrid_docs, item, K), 3),
            "MRR":         round(mrr(hybrid_docs, item, K), 3),
            "nDCG@5":      round(ndcg_at_k(hybrid_docs, item, K), 3),
        }
        results.append(row_hybrid)
        print(f"{q[:47]:<48} | {row_hybrid['Precision@5']:>5} | {row_hybrid['HitRate@5']:>5} "
              f"| {row_hybrid['MRR']:>5} | {row_hybrid['nDCG@5']:>5} | Hybrid+Rerank")
        print("-" * 90)

    df_eval = pd.DataFrame(results)
    print("\n" + "="*60)
    print("TỔNG KẾT TRUNG BÌNH THEO CHẾ ĐỘ:")
    summary = df_eval.groupby('Mode')[['Precision@5','HitRate@5','MRR','nDCG@5']].mean().round(3)
    print(summary.to_string())

    if 'Dense' in summary.index and 'Hybrid+Rerank' in summary.index:
        dense_mrr  = summary.loc['Dense', 'MRR']
        hybrid_mrr = summary.loc['Hybrid+Rerank', 'MRR']
        gain = hybrid_mrr - dense_mrr
        print(f"\n📊 MRR Gain (Hybrid+Rerank vs Dense): {gain:+.3f}")
        if gain > 0.05:
            print("   ✅ Reranker mang lại cải thiện rõ ràng — nên giữ.")
        elif gain > 0:
            print("   ⚠️  Reranker cải thiện nhẹ — cân nhắc cost/benefit.")
        else:
            print("   ❌ Reranker không cải thiện — có thể bỏ để hệ thống gọn hơn.")

    df_eval.to_csv(EVAL_RESULT_PATH, index=False)
    print(f"\n💾 Kết quả lưu tại: {EVAL_RESULT_PATH}")
    return df_eval

if __name__ == "__main__":
    # Đây chỉ là block test nhanh nếu chạy riêng eval.py
    # Trong thực tế nên chạy main.py
    from index import load_index
    v_store, b25, r_ranker, docs = load_index()
    run_evaluation(v_store, b25, r_ranker, docs)