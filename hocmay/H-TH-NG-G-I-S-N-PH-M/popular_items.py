"""
Popular Items Generator
Tạo danh sách sản phẩm phổ biến theo ngữ cảnh và toàn cục
"""

import pandas as pd
import numpy as np
import os
import ast

# Đường dẫn
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'
MODELS_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models'

# Tạo thư mục nếu chưa có
os.makedirs(MODELS_DIR, exist_ok=True)

# Các ngữ cảnh
CONTEXTS = [
    'Weekday_Morning', 'Weekday_Afternoon', 'Weekday_Evening', 'Weekday_Night',
    'Weekend_Morning', 'Weekend_Afternoon', 'Weekend_Evening', 'Weekend_Night'
]


def parse_product_list(product_str):
    """Parse chuỗi list sản phẩm thành Python list"""
    try:
        return ast.literal_eval(product_str)
    except:
        return []


def generate_popular_items():
    """
    Tạo danh sách sản phẩm phổ biến:
    1. Global popular: Top 100 sản phẩm phổ biến nhất
    2. Context popular: Top 100 sản phẩm phổ biến theo từng context
    """
    print("="*60)
    print("GENERATING POPULAR ITEMS")
    print("="*60)
    
    # Load transactions
    trans_path = os.path.join(PROCESSED_DIR, 'transactions_by_context.csv')
    print(f"\nLoading transactions from {trans_path}...")
    df = pd.read_csv(trans_path)
    print(f"Total transactions: {len(df)}")
    
    # Parse product lists
    print("Parsing product lists...")
    df['products'] = df['product_name'].apply(parse_product_list)
    
    # ========================================
    # 1. Global Popular Items
    # ========================================
    print("\n" + "-"*40)
    print("Computing Global Popular Items...")
    print("-"*40)
    
    # Đếm tần suất xuất hiện của mỗi sản phẩm
    all_products = []
    for products in df['products']:
        all_products.extend(products)
    
    product_counts = pd.Series(all_products).value_counts()
    
    # Top 100 global
    global_popular = product_counts.head(100)
    
    # Lưu file
    global_df = pd.DataFrame({
        'product_name': global_popular.index,
        'count': global_popular.values,
        'rank': range(1, len(global_popular) + 1)
    })
    
    global_path = os.path.join(MODELS_DIR, 'global_popular.csv')
    global_df.to_csv(global_path, index=False)
    print(f"Saved {len(global_df)} global popular items to {global_path}")
    
    print("\nTop 10 Global Popular:")
    for i, row in global_df.head(10).iterrows():
        print(f"  {row['rank']}. {row['product_name']} ({row['count']:,} transactions)")
    
    # ========================================
    # 2. Context-Aware Popular Items
    # ========================================
    print("\n" + "-"*40)
    print("Computing Context-Aware Popular Items...")
    print("-"*40)
    
    context_popular_list = []
    
    for context in CONTEXTS:
        # Lọc theo context
        df_context = df[df['context_label'] == context]
        
        if len(df_context) == 0:
            print(f"  {context}: No data")
            continue
        
        # Đếm tần suất trong context này
        context_products = []
        for products in df_context['products']:
            context_products.extend(products)
        
        context_counts = pd.Series(context_products).value_counts()
        
        # Top 100 cho context này
        top_100 = context_counts.head(100)
        
        for rank, (product, count) in enumerate(top_100.items(), 1):
            context_popular_list.append({
                'context': context,
                'product_name': product,
                'count': count,
                'rank': rank
            })
        
        print(f"  {context}: {len(df_context)} transactions, top product: {top_100.index[0]}")
    
    # Lưu file
    context_df = pd.DataFrame(context_popular_list)
    context_path = os.path.join(MODELS_DIR, 'context_popular.csv')
    context_df.to_csv(context_path, index=False)
    print(f"\nSaved {len(context_df)} context popular items to {context_path}")
    
    # ========================================
    # 3. Co-occurrence Matrix (Optional - cho tương lai)
    # ========================================
    print("\n" + "-"*40)
    print("Computing Product Co-occurrence (top 500 products)...")
    print("-"*40)
    
    # Lấy top 500 sản phẩm phổ biến
    top_500_products = set(product_counts.head(500).index)
    
    # Đếm co-occurrence
    from collections import defaultdict
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for products in df['products']:
        # Chỉ xét sản phẩm trong top 500
        filtered = [p for p in products if p in top_500_products]
        
        for i, p1 in enumerate(filtered):
            for p2 in filtered[i+1:]:
                cooccurrence[p1][p2] += 1
                cooccurrence[p2][p1] += 1
    
    # Convert to DataFrame và lưu top co-occurrences cho mỗi sản phẩm
    cooc_list = []
    for product, related in cooccurrence.items():
        # Top 20 related products
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)[:20]
        for related_product, count in sorted_related:
            cooc_list.append({
                'product': product,
                'related_product': related_product,
                'cooccurrence_count': count
            })
    
    cooc_df = pd.DataFrame(cooc_list)
    cooc_path = os.path.join(MODELS_DIR, 'product_cooccurrence.csv')
    cooc_df.to_csv(cooc_path, index=False)
    print(f"Saved {len(cooc_df)} co-occurrence pairs to {cooc_path}")
    
    print("\n" + "="*60)
    print("POPULAR ITEMS GENERATION COMPLETED!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {global_path}")
    print(f"  - {context_path}")
    print(f"  - {cooc_path}")


if __name__ == "__main__":
    generate_popular_items()
