import os

from config.config import INDEX_FOLDER, DOCSTORE_PATH
from build_index.index import build_index, load_index
from build_index.prepare_data import prepare_data
from rag.rag import rag_answer
from rag.retrieve_func import retrieve_hybrid, rerank_candidates
from evaluate.evaluation_func import run_evaluation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def run_interactive_session(vector_store, bm25, reranker, all_docs):
    """Chạy demo tìm kiếm tương tác. Gọi hàm này thủ công khi cần demo."""
    if vector_store is None:
        print("[ERROR] Chưa load index. Hãy chạy Cell 9 trước.")
        return

    while True:
        user_query = input("Nhập câu hỏi tìm kiếm (hoặc gõ 'exit' để thoát): ")
        if user_query.lower() in ['exit', 'quit', 'thoát']:
            print("Tạm biệt!")
            break
        if not user_query.strip():
            continue

        print("\nĐang tìm kiếm...")
        candidates = retrieve_hybrid(user_query, vector_store, bm25, all_docs, verbose=True)
        results = rerank_candidates(user_query, candidates, reranker, verbose=True)

        print(f"Bảng 3.3: Top-5 kết quả cho truy vấn \"{user_query}\"\n")
        print(f"{'Hạng':<5} | {'Sản phẩm':<38} | {'Score':<5} | {'Liên quan?'}")

        for idx, doc in enumerate(results[:5], 1):
            product = doc.metadata.get('product_name', 'N/A')
            score = doc.metadata.get('rerank_score', 0)
            doc_brand = str(doc.metadata.get('brand', '')).lower()

            q_lower = user_query.lower()
            p_lower = product.lower()

            # Cắt gọn tên sản phẩm nếu quá dài để không bị vỡ bảng
            short_product = product[:35] + "..." if len(product) > 38 else product

            # In từng dòng với định dạng căn lề
            print(f"{idx:<5} | {short_product:<38} | {score:<5.2f} ")

def run_demo(vector_store, bm25, reranker, all_docs, retrieval_mode="hybrid", top_k=5, verbose=False):
    demo_queries = [
        "Tìm cho tôi tai nghe ThieAudio Oracle MKIII",
        "Tìm dây cáp tai nghe PWAudio giá dưới 5 triệu",
        "FiiO JD7"
    ]

    for q in demo_queries:
        print("=" * 50)
        print("QUERY:", q)
        print("=" * 50)

        result = rag_answer( query=q, vector_store=vector_store, bm25=bm25, reranker=reranker, all_docs=all_docs,
                             retrieval_mode=retrieval_mode, top_k=top_k, verbose=verbose )

        print(result["answer"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print("-", s)
        print()

# Giả định các hàm này được import từ logic trong notebook của bạn
# từ rag_logic import vector_store, bm25, reranker, all_docs, rag_answer

app = FastAPI(title="Logistics & Audio RAG API")

# Cấu hình CORS để Angular có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/ask")
async def ask_rag(request: ChatRequest):
    try:
        # Gọi hàm rag_answer từ notebook của bạn
        # Lưu ý: Bạn cần đảm bảo các biến global (vector_store,...) đã được load
        result = rag_answer(
            query=request.query,
            vector_store=vector_store,
            bm25=bm25,
            reranker=reranker,
            all_docs=all_docs,
            retrieval_mode="hybrid"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    #
    # # 1. Chạy Evaluation tự động
    # run_evaluation(vector_store, bm25, reranker, all_docs)
    #
    # run_demo(vector_store, bm25, reranker, all_docs, top_k=5, verbose=True)
    #
    # # 2. Hỏi người dùng có muốn chat không
    # choice = input("\nBạn có muốn bắt đầu phiên hỏi đáp tương tác không? (y/n): ")
    # if choice.lower() in ['y', 'yes', 'có']:
    #     run_interactive_session(vector_store, bm25, reranker, all_docs)
    # else:
    #     print("Kết thúc chương trình.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # http://localhost:8000

