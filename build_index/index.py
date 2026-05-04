import os
import pickle

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.config import INDEX_FOLDER, DOCSTORE_PATH, EMBED_MODEL_NAME, RERANK_MODEL_NAME, DEVICE
from build_index.prepare_data import prepare_data
from utils.sharedutil import check_file_exists, check_metadata_fields

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def build_index(documents):
    """
    SỬA ĐIỂM 7: Tách riêng bước build index.
    SỬA ĐIỂM 9: Log chi tiết từng bước.
    """
    if documents is None or len(documents) == 0:
        print("[ERROR] Không có documents để build index!")
        return None

    print(f"[RUNNING] Bắt đầu build index với {len(documents)} documents...")

    # SỬA ĐIỂM 1: Dùng DEVICE từ config
    lc_embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': DEVICE}
    )

    texts     = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    print(f"   Đang encode {len(texts)} texts bằng {DEVICE.upper()}...")
    fast_embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    batch = 256 if DEVICE == 'cuda' else 32
    embeddings_matrix = fast_embedder.encode(texts, batch_size=batch, show_progress_bar=True)

    print("\n[RUNNING] Đóng gói vào FAISS...")
    text_embedding_pairs = list(zip(texts, embeddings_matrix.tolist()))
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=lc_embedder,
        metadatas=metadatas
    )

    print(f"[INFO] Lưu index vào: {INDEX_FOLDER}")
    vector_store.save_local(INDEX_FOLDER)
    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"[INFO] Build index hoàn thành! ({len(documents)} vectors)")
    return vector_store

def load_index():
    """
    SỬA ĐIỂM 7: Tách riêng bước load.
    SỬA ĐIỂM 9: Guard kiểm tra file tồn tại.
    """
    if not check_file_exists(INDEX_FOLDER, "INDEX_FOLDER"):
        return None, None, None, None
    if not check_file_exists(DOCSTORE_PATH, "DOCSTORE"):
        return None, None, None, None

    print("\n⏳ Đang tải Embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': DEVICE}  # SỬA ĐIỂM 1
    )
    vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    with open(DOCSTORE_PATH, "rb") as f:
        all_docs = pickle.load(f)

    # SỬA ĐIỂM 9: Kiểm tra metadata đủ trường
    REQUIRED_FIELDS = ['product_name', 'brand', 'price', 'url', 'original_content']
    bad_docs = [i for i, d in enumerate(all_docs) if not check_metadata_fields(d, REQUIRED_FIELDS)]
    if bad_docs:
        print(f"[WARNING]  {len(bad_docs)} documents thiếu metadata — có thể ảnh hưởng reranking")
    else:
        print(f"[INFO] Tất cả {len(all_docs)} documents có đủ metadata")

    # SỬA ĐIỂM 8: BM25 tokenize chuẩn PyVi (page_content đã được tokenize khi build)
    print("[RUNNING] Đang khởi tạo BM25 (dùng page_content đã tokenize PyVi)...")
    tokenized_corpus = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   BM25 vocab size: {len(bm25.idf)} terms")

    print("[RUNNING] Đang tải Reranker...")
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)  # SỬA ĐIỂM 1

    print(f"\n[INFO] Hệ thống Retrieval sẵn sàng! ({len(all_docs)} docs trong index)")
    return vector_store, bm25, reranker, all_docs

def main():
    index_exists = os.path.exists(INDEX_FOLDER) and os.path.exists(DOCSTORE_PATH)
    if not index_exists:
        print("[WARNING] Chưa có Index — bắt đầu build từ đầu...")
        all_docs_prepared = prepare_data()
        vector_store = build_index(all_docs_prepared)
        vector_store, bm25, reranker, all_docs = load_index()
    else:
        print("[INFO] Index đã tồn tại — load trực tiếp...")
        vector_store, bm25, reranker, all_docs = load_index()

if __name__ == '__main__':
    main()