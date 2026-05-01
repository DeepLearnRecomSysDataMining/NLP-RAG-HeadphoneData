import os
import sys

from config import INDEX_FOLDER, DOCSTORE_PATH
from index import build_index, load_index
from prepare_data import prepare_data
from rag_answer import retrieve_hybrid, rerank_candidates
from eval import run_evaluation

def run_interactive_session(vector_store, bm25, reranker, all_docs):
    if vector_store is None:
        print("[ERROR] Chưa load index.")
        return
    
    print("\n" + "="*50)
    print("SẴN SÀNG CHO PHIÊN HỎI ĐÁP TƯƠNG TÁC")
    print("="*50)
    
    while True:
        user_query = input("\nNhập câu hỏi tìm kiếm (hoặc gõ 'exit' để thoát): ")
        if user_query.lower() in ['exit', 'quit', 'thoát']:
            print("Tạm biệt!")
            break
        if not user_query.strip():
            continue
            
        print("\nĐang tìm kiếm...")
        candidates = retrieve_hybrid(user_query, vector_store, bm25, all_docs, verbose=True)
        results    = rerank_candidates(user_query, candidates, reranker, verbose=True)

        print(f"\nTÌM THẤY {len(results)} KẾT QUẢ TỐT NHẤT:")
        for idx, doc in enumerate(results, 1):
            product   = doc.metadata.get('product_name', 'N/A')
            price     = doc.metadata.get('price', 0)
            price_str = f"{price:,} VNĐ" if price > 0 else "Liên hệ"
            score     = doc.metadata.get('rerank_score', 0)
            content   = doc.metadata.get('original_content', '')
            print(f"[{idx}] SẢN PHẨM: {product}")
            print(f"    💰 Giá: {price_str}")
            print(f"    📈 Điểm Rerank: {score:.2f} (Càng cao càng sát nghĩa)")
            print(f"    📄 Trích đoạn: {content[:200]}...")

if __name__ == "__main__":
    index_exists = os.path.exists(INDEX_FOLDER) and os.path.exists(DOCSTORE_PATH)
    
    if not index_exists:
        print("[WARNING] Chưa có Index — bắt đầu build từ đầu...")
        all_docs_prepared = prepare_data()
        vector_store = build_index(all_docs_prepared)
        # Load lại để lấy đầy đủ các object (bm25, reranker, all_docs)
        vector_store, bm25, reranker, all_docs = load_index()
    else:
        print("[INFO] Index đã tồn tại — load trực tiếp...")
        vector_store, bm25, reranker, all_docs = load_index()

    # 1. Chạy Evaluation tự động
    run_evaluation(vector_store, bm25, reranker, all_docs)

    # 2. Hỏi người dùng có muốn chat không
    choice = input("\nBạn có muốn bắt đầu phiên hỏi đáp tương tác không? (y/n): ")
    if choice.lower() in ['y', 'yes', 'có']:
        run_interactive_session(vector_store, bm25, reranker, all_docs)
    else:
        print("Kết thúc chương trình.")

