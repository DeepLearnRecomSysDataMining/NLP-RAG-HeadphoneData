import os
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Cấu hình
INDEX_PATH = "faiss_index"
DOCSTORE_PATH = "docstore.pkl"
TOP_K_RETRIEVE = 15  # Lấy rộng để rerank
TOP_K_FINAL = 3  # Lấy hẹp để đưa vào LLM

# Load tài nguyên
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

with open(DOCSTORE_PATH, "rb") as f:
    all_docs = pickle.load(f)

# Khởi tạo BM25
tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)

# Khởi tạo Reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rrf(dense_results, sparse_results, k=60):
    """Reciprocal Rank Fusion để gộp kết quả Dense và Sparse"""
    scores = {}
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_hybrid(query):
    # 1. Dense Search
    dense_docs = vector_store.similarity_search(query, k=TOP_K_RETRIEVE)

    # 2. Sparse Search (BM25)
    tokenized_query = query.lower().split()
    sparse_docs = bm25.get_top_n(tokenized_query, all_docs, n=TOP_K_RETRIEVE)

    # Gộp dựa trên page_content (đơn giản hóa cho demo)
    # Trong thực tế nên dùng ID của chunk
    all_candidates = list({doc.page_content: doc for doc in (dense_docs + sparse_docs)}.values())

    # 3. Reranking bằng Cross-Encoder
    pairs = [[query, doc.page_content] for doc in all_candidates]
    rerank_scores = reranker.predict(pairs)

    # Sắp xếp lại theo điểm rerank
    for i, score in enumerate(rerank_scores):
        all_candidates[i].metadata['rerank_score'] = score

    sorted_docs = sorted(all_candidates, key=lambda x: x.metadata['rerank_score'], reverse=True)
    return sorted_docs[:TOP_K_FINAL]


def rag_answer(query, retrieval_mode="hybrid"):
    if retrieval_mode == "dense":
        context_docs = vector_store.similarity_search(query, k=TOP_K_FINAL)
    else:
        context_docs = retrieve_hybrid(query)

    context_text = "\n\n".join(
        [f"Sản phẩm: {d.metadata['product_name']}\nThông tin: {d.metadata['original_content']}" for d in context_docs])

    # Giả lập gọi LLM (Bạn thay thế bằng API OpenAI/Gemini của bạn)
    prompt = f"Dựa trên các thông tin sau:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"

    # Ở đây trả về context để file eval.py có thể chấm điểm relevance
    return {
        "answer": f"[Simulated Answer for: {query}]",
        "context": context_text,
        "sources": [d.metadata['source'] for d in context_docs]
    }


if __name__ == "__main__":
    test_q = "Tai nghe ThieAudio Oracle MKIII có giá bao nhiêu và cấu hình thế nào?"
    res = rag_answer(test_q)
    print(f"Query: {test_q}\nContext found:\n{res['context']}")