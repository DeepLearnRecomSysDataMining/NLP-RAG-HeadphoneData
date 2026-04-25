import pandas as pd
from rag_answer import rag_answer


def score_mock(query, context):
    """Hàm giả lập chấm điểm relevance (Trong thực tế dùng LLM-as-a-judge)"""
    # Nếu trong context có chứa từ khóa quan trọng từ query thì điểm cao
    keywords = [w for w in query.lower().split() if len(w) > 3]
    match_count = sum(1 for k in keywords if k in context.lower())
    return min(5, match_count + 1)


def run_evaluation():
    test_questions = [
        "Giá của Oracle MKIII là bao nhiêu?",
        "Tai nghe nào phù hợp để nghe nhạc chuyên nghiệp?",
        "Monster Boomerang có chống nước không?",
        "Dây cáp Entry Level Legend III làm từ chất liệu gì?"
    ]

    results = []
    print("--- Bắt đầu đánh giá hệ thống ---")

    for q in test_questions:
        # Chạy Baseline (Dense)
        res_dense = rag_answer(q, retrieval_mode="dense")
        score_dense = score_mock(q, res_dense['context'])

        # Chạy Variant (Hybrid + Rerank)
        res_hybrid = rag_answer(q, retrieval_mode="hybrid")
        score_hybrid = score_mock(q, res_hybrid['context'])

        results.append({
            "Question": q,
            "Dense_Score": score_dense,
            "Hybrid_Score": score_hybrid,
            "Gain": score_hybrid - score_dense
        })

    df = pd.DataFrame(results)
    print("\nBẢNG SO SÁNH KẾT QUẢ:")
    print(df.to_string(index=False))
    print(f"\nĐiểm trung bình Dense: {df['Dense_Score'].mean()}")
    print(f"Điểm trung bình Hybrid: {df['Hybrid_Score'].mean()}")

    df.to_csv("eval_results.csv", index=False)
    print("\nKết quả đã được lưu vào eval_results.csv")


if __name__ == "__main__":
    run_evaluation()