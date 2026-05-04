import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import concurrent.futures
import re
from config.config import RAW_DATA_PATH

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
BASE_DOMAIN = "https://tainghe.com.vn" # Web cào dữ liệu
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}
MAX_THREADS = 4 # Cào với 20 luồng

# BƯỚC 1: TÌM DANH MỤC TỪ TRANG CHỦ
def get_all_subcategories():
    print("--- ĐANG LỌC DANH MỤC TRÊN TRANG CHỦ ---")
    sub_categories = set()

    try:
        response = requests.get(BASE_DOMAIN, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Quét các thẻ <a> chứa link danh mục .html
        menu_items = soup.find_all('a')
        for item in menu_items:
            href = item.get('href', '')
            if href.endswith('.html') and not href.startswith('http'):
                full_link = BASE_DOMAIN + href if href.startswith('/') else BASE_DOMAIN + '/' + href
                # Chỉ lấy các danh mục liên quan đến thiết bị âm thanh
                if any(keyword in href for keyword in ['tai-nghe', 'may-nghe-nhac', 'dac-amp', 'loa', 'phu-kien', 'hi-end-cables']):
                    sub_categories.add(full_link)

        print(f"[OK] Đã chốt được {len(sub_categories)} link danh mục.")
        return list(sub_categories)
    except Exception as e:
        print(f"[LỖI] Lọc danh mục thất bại: {e}")
        return []

# BƯỚC 2: THU THẬP LINK SẢN PHẨM (TỪ TỪNG DANH MỤC)
def get_product_links_from_category(category_url, max_pages_per_cat=50):
    product_links = set()

    for page in range(1, max_pages_per_cat + 1):
        url = f"{category_url}?page={page}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Nếu hết sản phẩm thì chuyển sang danh mục khác
            items = soup.find_all('a', class_='p-name')
            if not items:
                break

            for item in items:
                link = item.get("href")
                if link:
                    link = link[0] if isinstance(link, list) else link
                    full_link = str(link) if "http" in str(link) else BASE_DOMAIN + str(link)
                    product_links.add(full_link)
            # Ngủ 1 nhịp để server khỏi chú ý
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"[LỖI PAGE] {url} -> {e}")
            break

    return list(product_links)

# BƯỚC 3: BÓC TÁCH CHI TIẾT SẢN PHẨM (ĐA LUỒNG)
def fetch_product_details(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. Tên sản phẩm
        name_tag = soup.find('h1', id='detail-name')
        name = name_tag.text.strip() if name_tag else "Unknown"

        # 2. Giá
        price_tag = soup.find('span', class_='price_config')
        price = price_tag.text.strip() if price_tag else "Contact"

        # 3. Thương hiệu (Brand) - Thuật toán Quét đa tầng (Bottom-up)
        brand = "Unknown"

        # Lấy TẤT CẢ các thẻ có chứa cụm từ "thương hiệu"
        candidates = soup.find_all(lambda t: t.name in ['li', 'p', 'div', 'tr', 'span', 'b', 'strong']
                                   and t.get_text()
                                   and 'thương hiệu' in t.get_text().lower())

        if candidates:
            # Sắp xếp các thẻ theo độ dài chữ tăng dần (để đọc từ trong lõi đọc ra ngoài)
            candidates_sorted = sorted(candidates, key=lambda t: len(t.get_text(strip=True)))

            for tag in candidates_sorted:
                # Ép thẻ thành chữ, các thẻ con bên trong cách nhau bằng khoảng trắng
                text = tag.get_text(separator=' ', strip=True)

                # Regex tìm chữ "Thương hiệu", bỏ qua dấu : hoặc - và lấy toàn bộ chữ phía sau
                match = re.search(r'thương\s*hiệu\s*[:\-]?\s*(.+)', text, re.IGNORECASE)

                if match:
                    extracted = match.group(1).strip()
                    # Kiểm tra tính hợp lệ: Tên hãng phải có chữ và thường không dài quá 30 ký tự
                    if extracted and len(extracted) < 30:
                        brand = extracted
                        break # Tìm thấy tên hãng rồi thì khóa mục tiêu, thoát vòng lặp!

        # 4. Review / Mô tả
        content_tag = soup.find('div', class_='emtry_content')
        content = content_tag.text.strip() if content_tag else ""

        # Thời gian nghỉ ngẫu nhiên cho mỗi luồng
        time.sleep(random.uniform(1.0, 2.5))

        # Chỉ lưu nếu sản phẩm có tên và có bài review
        if content and name != "Unknown":
            print(f"[XONG] {name[:30]:<30} | Hãng: {brand:<15} | Giá: {price}")
            return {
                "url": url,
                "product_name": name,
                "brand": brand,
                "price": price,
                "review_content": content
            }
        return None
    except Exception as e:
        print(f"[LỖI] {url} -> {e}")
        return None

# MAIN
def run_full_scraper():
    start_time = time.time()

    # Bước 1
    categories = get_all_subcategories()
    if not categories:
        return

    # Bước 2
    print(f"\n--- BƯỚC 2: ĐANG GOM LINK TỪ {len(categories)} DANH MỤC ---")
    all_product_links = set()
    for i, cat_url in enumerate(categories):
        print(f"Đang lật trang danh mục [{i+1}/{len(categories)}]: {cat_url}")
        links = get_product_links_from_category(cat_url)
        all_product_links.update(links)

    product_links_list = list(all_product_links)
    print(f"\n✅ TỔNG CỘNG ĐÃ GOM ĐƯỢC: {len(product_links_list)} LINK SẢN PHẨM KHÁC NHAU.")

    # Bước 3
    print(f"\n--- BƯỚC 3: BẮT ĐẦU BÓC TÁCH CHI TIẾT BẰNG {MAX_THREADS} THREADS ---")
    final_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {executor.submit(fetch_product_details, url): url for url in product_links_list}

        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            if data:
                final_data.append(data)

    # Lưu ra CSV trên Google Drive
    local_save_path = RAW_DATA_PATH
    df = pd.DataFrame(final_data)
    df.to_csv(local_save_path, index=False, encoding='utf-8-sig')

    end_time = time.time()
    print("\n" + "="*50)
    print(f"HOÀN THÀNH CÀO DỮ LIỆU!")
    print(f"Thu thập thành công: {len(final_data)} sản phẩm.")
    print(f"Thời gian chạy: {round((end_time - start_time)/60, 2)} phút.")
    print(f"File lưu tại : {local_save_path}")
    print("="*50)

if __name__ == "__main__":
    run_full_scraper()