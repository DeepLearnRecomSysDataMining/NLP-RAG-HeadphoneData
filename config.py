import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Tự động chọn CPU nếu không có GPU (phù hợp khi chuyển từ Colab sang máy cá nhân)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Đang sử dụng thiết bị: {DEVICE}")

RAW_DATA_PATH = "full_xuanvu_database.csv"
INDEX_FOLDER = "faiss_index_v2"
DOCSTORE_PATH = "docstore_v2.pkl"
MODEL_NAME = 'keepitreal/vietnamese-sbert'
EVAL_RESULT_PATH = 'eval_results.csv'

EMBED_MODEL_NAME = 'bkai-foundation-models/vietnamese-bi-encoder'
RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'

TOP_K_RETRIEVE = 25
TOP_K_FINAL    = 5
CHUNK_IF_LONGER_THAN = 1000  # Chỉ chunk nếu review dài hơn N ký tự
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

BRAND_KEYWORDS = [
    "sony", "jbl", "sennheiser", "soundpeats", "anker", "edifier",
    "final", "fiio", "moondrop", "soundmagic", "akg", "bose",
    "apple", "marshall", "hifiman",
    "thieaudio", "beyerdynamic", "audio-technica", "audiotechnica",
    "unique melody", "pwaudio", "pw audio", "monster", "koss",
    "hifiman", "final", "fiio", "moondrop", "tangzu", "hiby",
    "dekoni", "hidizs", "campfire", "64 audio", "64audio",
    "empire ears", "meze", "grado", "focal", "audeze",
    "tin hifi", "tinhifi", "simgot", "dunu", "kinera",
    "letshuoer", "truthear", "7hz", "tripowin", "kz",
    "cca", "tanchjim", "yanyin", "softears"
]