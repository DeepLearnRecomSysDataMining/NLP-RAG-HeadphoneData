from config.config import RAW_DATA_PATH
import pandas as pd

# OPTIONAL DEBUG CELL
# Dùng để kiểm tra URL thực tế trong database nếu một relevant_url không match.
# Không bắt buộc chạy trong lúc báo cáo.

try:
    df_debug = pd.read_csv(RAW_DATA_PATH)
    search_terms = [
        "oracle mkiii",
        "beyerdynamic",
        "ath-m50",
        "boomerang",
        "legend iii",
        "pwaudio",
        "fiio jd7"
    ]

    print("=== URL THỰC TẾ TRONG DATABASE ===\n")
    for term in search_terms:
        matches = df_debug[df_debug['product_name'].astype(str).str.lower().str.contains(term, na=False)]
        for _, row in matches.head(10).iterrows():
            print(f"[{term}]")
            print(f"  product_name : {row.get('product_name', '')}")
            print(f"  url          : {row.get('url', '')}")
            print()
except Exception as e:
    print("[INFO] Bỏ qua debug URL cell:", e)
