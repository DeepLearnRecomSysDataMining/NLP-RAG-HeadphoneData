import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.config import RAW_DATA_PATH
from utils.sharedutil import check_file_exists, extract_brand


def run_eda():
    print(" BẮT ĐẦU KHÁM PHÁ DỮ LIỆU (EDA)")
    if not check_file_exists(RAW_DATA_PATH, "RAW_DATA"):
        return
    df = pd.read_csv(RAW_DATA_PATH)
    df['brand'] = df['product_name'].apply(extract_brand)
    print(f"   Tổng số sản phẩm: {len(df)}")
    print(f"   Số thương hiệu nhận diện được: {df[df['brand'] != 'khác']['brand'].nunique()}")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    top_brands = df[df['brand'] != 'khác']['brand'].value_counts().head(10)
    sns.barplot(x=top_brands.values, y=top_brands.index, hue=top_brands.index, palette='viridis', legend=False)
    plt.title('Top 10 Thương hiệu sản phẩm')
    plt.show()