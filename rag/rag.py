# ============================================================
# CELL: RAG ANSWER - RETRIEVAL + CONTEXT + TEMPLATE GENERATION
# Không dùng GPT/Claude/LLM API, phù hợp nếu môn học hạn chế dùng LLM pretrained
# ============================================================

import re

from config.config import INDEX_FOLDER, DOCSTORE_PATH, BRAND_KEYWORDS
from rag.retrieve_func import retrieve_hybrid, rerank_candidates, retrieve_dense

def format_price_vnd(price):
    try:
        price = int(price)
        if price <= 0:
            return "Liên hệ"
        return f"{price:,} VNĐ".replace(",", ".")
    except:
        return "Liên hệ"


def shorten_text(text, max_chars=450):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def parse_user_need(query):
    """ Bóc tách đơn giản nhu cầu người dùng để sinh câu trả lời tự nhiên hơn. """
    q = query.lower()

    need = {
        "has_price_constraint": False,
        "max_price": None,
        "brand_hint": None,
        "intent_type": "general"
    }

    # Giá: hỗ trợ dưới 500k, 1tr, 1 triệu, 2.5 triệu, 2,5 triệu
    price_patterns = [
        r"(?:dưới|duoi|nhỏ hơn|nho hon|không quá|khong qua|tối đa|toi da|đổ lại|do lai)\s*(\d+(?:[.,]\d+)?)\s*(triệu|tr|k|ngàn|nghìn)",
        r"(\d+(?:[.,]\d+)?)\s*(triệu|tr|k|ngàn|nghìn)\s*(?:đổ lại|do lai|trở xuống|tro xuong)"
    ]

    for pattern in price_patterns:
        m = re.search(pattern, q)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2)
            if unit in ["triệu", "tr"]:
                need["max_price"] = int(val * 1_000_000)
            elif unit in ["k", "ngàn", "nghìn"]:
                need["max_price"] = int(val * 1_000)
            need["has_price_constraint"] = True
            break

    # Brand hint đơn giản, lấy từ list nếu đã có BRAND_KEYWORDS trong notebook
    try:
        brands = BRAND_KEYWORDS
    except:
        brands = [
            "sony", "jbl", "fiio", "thieaudio", "pwaudio", "ddhifi",
            "monster", "sennheiser", "bose", "moondrop", "final",
            "soundpeats", "edifier", "anker", "akg"
        ]

    for b in brands:
        if b.lower() in q:
            need["brand_hint"] = b
            break

    # Intent đơn giản
    if any(x in q for x in ["tốt nhất", "đáng mua", "recommend", "gợi ý", "tư vấn", "nên mua"]):
        need["intent_type"] = "recommendation"
    elif any(x in q for x in ["thông tin", "mô tả", "chi tiết", "review"]):
        need["intent_type"] = "information"
    elif any(x in q for x in ["giá", "dưới", "triệu", "500k", "1tr", "2tr"]):
        need["intent_type"] = "budget"

    return need


def build_rag_context(context_docs, max_chars_per_doc=450):
    """ Tạo context từ các document đã retrieve/rerank. """
    context_blocks = []

    for rank, doc in enumerate(context_docs, start=1):
        meta = doc.metadata

        name = meta.get("product_name", "Không rõ tên")
        brand = meta.get("brand", "Không rõ")
        price = format_price_vnd(meta.get("price", 0))
        url = meta.get("url", "N/A")
        content = meta.get("original_content", doc.page_content)

        block = {
            "rank": rank,
            "product_name": name,
            "brand": brand,
            "price": price,
            "url": url,
            "summary": shorten_text(content, max_chars=max_chars_per_doc),
            "score": meta.get("rerank_score", None)
        }
        context_blocks.append(block)

    return context_blocks


def generate_template_answer(query, context_blocks):
    """
    Sinh câu trả lời tự nhiên dựa trên context retrieved.
    Đây là phần generation dạng template-based, không gọi LLM.
    """
    need = parse_user_need(query)

    if not context_blocks:
        return (
            "Mình chưa tìm thấy sản phẩm phù hợp với truy vấn này trong dữ liệu hiện có. "
            "Bạn có thể thử nhập truy vấn cụ thể hơn, ví dụ thêm thương hiệu, mức giá hoặc loại sản phẩm."
        )

    intro = "Dựa trên các sản phẩm được truy hồi từ dữ liệu hiện có, mình tìm được một số lựa chọn phù hợp như sau:"

    if need["intent_type"] == "budget" and need["max_price"]:
        intro = (
            f"Dựa trên điều kiện ngân sách khoảng dưới {format_price_vnd(need['max_price'])}, "
            "hệ thống truy hồi được một số sản phẩm phù hợp như sau:"
        )
    elif need["intent_type"] == "recommendation":
        intro = (
            "Dựa trên truy vấn mang tính tư vấn/gợi ý, hệ thống ưu tiên các sản phẩm có nội dung gần nhất "
            "với nhu cầu của bạn trong dữ liệu hiện có:"
        )
    elif need["intent_type"] == "information":
        intro = (
            "Dựa trên truy vấn cần thông tin sản phẩm, hệ thống tìm được các sản phẩm liên quan nhất như sau:"
        )

    lines = [intro, ""]

    for item in context_blocks:
        rank = item["rank"]
        name = item["product_name"]
        brand = item["brand"]
        price = item["price"]
        summary = item["summary"]
        url = item["url"]

        lines.append(f"{rank}. {name}")
        lines.append(f"   - Thương hiệu: {brand}")
        lines.append(f"   - Giá: {price}")
        lines.append(f"   - Thông tin liên quan: {summary}")
        lines.append(f"   - Nguồn: {url}")
        lines.append("")

    # Kết luận ngắn
    best = context_blocks[0]
    lines.append(
        f"Gợi ý nhanh: Nếu cần chọn một kết quả nổi bật nhất theo truy vấn hiện tại, "
        f"mình sẽ ưu tiên sản phẩm ở hạng 1: {best['product_name']}."
    )

    lines.append(
        "Lưu ý: Câu trả lời này được sinh dựa trên các kết quả truy hồi từ hệ thống, "
        "không dùng mô hình LLM bên ngoài."
    )

    return "\n".join(lines)


def rag_answer( query, vector_store, bm25, reranker, all_docs, retrieval_mode="hybrid", top_k=5, verbose=False ):
    """
    Pipeline RAG hiện tại:
    1. Retrieve candidates
    2. Rerank candidates
    3. Build context
    4. Generate answer từ context
    """

    # 1. Retrieval
    if retrieval_mode == "hybrid":
        candidates = retrieve_hybrid( query=query, vector_store=vector_store, bm25=bm25, all_docs=all_docs, k_retrieve=50, verbose=verbose )

        context_docs = rerank_candidates( query=query, candidates=candidates, reranker=reranker, top_k=top_k, verbose=verbose )

    else:
        context_docs = retrieve_dense( query=query, vector_store=vector_store, k=top_k )

    # 2. Build context
    context_blocks = build_rag_context(context_docs)

    # 3. Generate final answer
    answer = generate_template_answer(query, context_blocks)

    # 4. Format raw context để debug / báo cáo
    raw_context = []
    for item in context_blocks:
        raw_context.append(
            f"[{item['rank']}] {item['product_name']}\n"
            f"Brand: {item['brand']}\n"
            f"Price: {item['price']}\n"
            f"URL: {item['url']}\n"
            f"Context: {item['summary']}"
        )

    return {
        "query": query,
        "answer": answer,
        "context": "\n\n" + "=" * 60 + "\n\n".join(raw_context),
        "sources": [item["url"] for item in context_blocks],
        "retrieved_products": context_blocks,
        "retrieval_mode": retrieval_mode
    }