import pandas as pd
import numpy as np
import os
import ast
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    print("Please install mlxtend: pip install mlxtend")
    exit(1)

DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'
MODELS_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models'
RULES_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/rules'

os.makedirs(RULES_DIR, exist_ok=True)

CONTEXTS = [
    'Weekday_Morning', 'Weekday_Afternoon', 'Weekday_Evening', 'Weekday_Night',
    'Weekend_Morning', 'Weekend_Afternoon', 'Weekend_Evening', 'Weekend_Night'
]


def load_transactions():
    """
    Load transactions từ file đã tiền xử lý
    """
    trans_path = os.path.join(PROCESSED_DIR, 'transactions_by_context.csv')
    print(f"Loading transactions from {trans_path}...")
    
    df = pd.read_csv(trans_path)
    print(f"Total transactions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def parse_product_list(product_str):
    """
    Parse chuỗi list sản phẩm thành Python list
    """
    try:
        return ast.literal_eval(product_str)
    except:
        return []


def prepare_transactions_for_context(df, context_label):
    """
    Lọc và chuẩn bị transactions cho một ngữ cảnh cụ thể
    """
    # Lọc theo context
    df_context = df[df['context_label'] == context_label].copy()
    
    if len(df_context) == 0:
        print(f"  No transactions found for context: {context_label}")
        return None
    
    print(f"  Found {len(df_context)} transactions for {context_label}")
    
    # Parse product lists
    transactions = df_context['product_name'].apply(parse_product_list).tolist()
    
    # Lọc bỏ transactions rỗng
    transactions = [t for t in transactions if len(t) >= 2]
    
    print(f"  Valid transactions (>= 2 items): {len(transactions)}")
    
    return transactions


def mine_rules_fpgrowth(transactions, min_support=0.005, min_confidence=0.2, min_lift=1.0,
                         max_transactions=50000, top_n_products=1000):
    """
    Khai phá luật kết hợp sử dụng FP-Growth
    
    Parameters:
    -----------
    transactions: list of lists - Danh sách các giao dịch
    min_support: float - Ngưỡng support tối thiểu (0.005 = 0.5%)
    min_confidence: float - Ngưỡng confidence tối thiểu
    min_lift: float - Ngưỡng lift tối thiểu
    max_transactions: int - Số transaction tối đa để tránh OOM
    top_n_products: int - Chỉ giữ top N sản phẩm phổ biến nhất
    
    Returns:
    --------
    rules: DataFrame - Bảng các luật kết hợp
    """
    if len(transactions) < 100:
        print(f"  Too few transactions ({len(transactions)}), skipping...")
        return None
   
    print(f"  Original: {len(transactions)} transactions")
    
    # Đếm tần suất sản phẩm
    from collections import Counter
    product_counts = Counter()
    for trans in transactions:
        product_counts.update(trans)
    
    # Lấy top N sản phẩm
    top_products = set([p for p, _ in product_counts.most_common(top_n_products)])
    print(f"  Filtering to top {top_n_products} products...")
    
    # Lọc transactions chỉ giữ top products
    filtered_transactions = []
    for trans in transactions:
        filtered = [p for p in trans if p in top_products]
        if len(filtered) >= 2:  # Chỉ giữ giao dịch có >= 2 sản phẩm
            filtered_transactions.append(filtered)
    
    print(f"  After product filter: {len(filtered_transactions)} transactions")
    
   
    if len(filtered_transactions) > max_transactions:
        import random
        random.seed(42)
        filtered_transactions = random.sample(filtered_transactions, max_transactions)
        print(f"  Sampled to: {max_transactions} transactions")
    
    if len(filtered_transactions) < 100:
        print(f"  Too few transactions after filtering, skipping...")
        return None
    
 
    te = TransactionEncoder()
    te_ary = te.fit(filtered_transactions).transform(filtered_transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"  Encoded shape: {df_encoded.shape}")
    
  
    print(f"  Running FP-Growth (min_support={min_support})...")
    try:
        frequent_items = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    except Exception as e:
        print(f"  FP-Growth failed: {e}")
        return None
    
    if len(frequent_items) == 0:
        print("  No frequent itemsets found. Try lowering min_support.")
        return None
    
    print(f"  Found {len(frequent_items)} frequent itemsets")
    
    print(f"  Generating rules (min_confidence={min_confidence})...")
    try:
        rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
    except Exception as e:
        print(f"  Rule generation failed: {e}")
        return None
    
    if len(rules) == 0:
        print("  No rules found. Try lowering min_confidence.")
        return None
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    print(f"  Final rules (lift >= {min_lift}): {len(rules)}")
    
    # Convert frozensets to strings for saving
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules


def mine_all_contexts(df, min_support=0.005, min_confidence=0.2, min_lift=1.0):
    """
    Khai phá luật cho tất cả các ngữ cảnh
    """
    all_rules = {}
    
    for context in CONTEXTS:
        print(f"\n{'='*50}")
        print(f"Processing: {context}")
        print('='*50)
        
        # Chuẩn bị transactions
        transactions = prepare_transactions_for_context(df, context)
        
        if transactions is None or len(transactions) < 100:
            print(f"  Skipping {context} (insufficient data)")
            continue
        
        # Khai phá luật
        rules = mine_rules_fpgrowth(
            transactions, 
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift
        )
        
        if rules is not None and len(rules) > 0:
            all_rules[context] = rules
            
            # Lưu rules ra file
            output_path = os.path.join(RULES_DIR, f'{context}_rules.csv')
            rules.to_csv(output_path, index=False)
            print(f"  Saved to {output_path}")
    
    return all_rules


def print_top_rules(rules_dict, top_n=5):
    """
    In các luật tốt nhất cho mỗi ngữ cảnh
    """
    print("\n" + "="*60)
    print("TOP ASSOCIATION RULES BY CONTEXT")
    print("="*60)
    
    for context, rules in rules_dict.items():
        print(f"\n{context}:")
        print("-" * 40)
        
        top_rules = rules.nlargest(top_n, 'lift')
        
        for idx, row in top_rules.iterrows():
            ant = row['antecedents']
            con = row['consequents']
            conf = row['confidence']
            lift = row['lift']
            
            # Format đẹp hơn
            ant_str = ', '.join(ant[:2]) + ('...' if len(ant) > 2 else '')
            con_str = ', '.join(con[:2]) + ('...' if len(con) > 2 else '')
            
            print(f"  {ant_str} → {con_str}")
            print(f"     confidence={conf:.3f}, lift={lift:.3f}")


def main():
    print("="*60)
    print("ASSOCIATION RULES MINING WITH FP-GROWTH")
    print("="*60)
    
    # 1. Load transactions
    df = load_transactions()
    
    print("\nContexts in data:")
    print(df['context_label'].value_counts())
    
    # 3. Khai phá luật cho tất cả contexts
    # Có thể điều chỉnh các tham số tùy theo data
    rules_dict = mine_all_contexts(
        df,
        min_support=0.005,    # 0.5% - có thể giảm nếu ít luật
        min_confidence=0.2,   # 20%
        min_lift=1.0          # Chỉ giữ luật có correlation dương
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_rules = 0
    for context, rules in rules_dict.items():
        n_rules = len(rules)
        total_rules += n_rules
        print(f"  {context}: {n_rules} rules")
    
    print(f"\n  TOTAL: {total_rules} rules across {len(rules_dict)} contexts")
    
    if rules_dict:
        print_top_rules(rules_dict, top_n=3)
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print(f"Rules saved to: {RULES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
