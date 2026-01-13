import pandas as pd
import numpy as np
import os

# Đường dẫn dữ liệu
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
OUTPUT_DIR = '/Users/huudat/hocmay/processed_data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    print("Loading data...")
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    departments = pd.read_csv(os.path.join(DATA_DIR, 'departments.csv'))
    
    # Chỉ dùng tập 'prior' cho việc training/learning patterns
    order_products_prior = pd.read_csv(os.path.join(DATA_DIR, 'order_products__prior.csv'))
    
    return orders, products, departments, order_products_prior

def clean_data(orders):
    print("Cleaning data...")
    # Điền giá trị 0 cho đơn hàng đầu tiên (NaN caused by SQL lag)
    orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
    return orders

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

def get_day_type(dow):
    # Giả định: 0=Saturday/Sunday, 1=Sunday/Monday. 
    # Theo document Instacart, 0 và 1 thường là cuối tuần.
    if dow <= 1:
        return 'Weekend'
    else:
        return 'Weekday'

def create_user_features(orders, order_products_prior, products, departments):
    print("Creating User Features for KMeans...")
    
    # 1. Merge để có thông tin đầy đủ
    # Chỉ lấy các order thuộc tập 'prior'
    prior_orders = orders[orders['eval_set'] == 'prior'].copy()
    
    # Tạo các cột Time Context ngay trên bảng orders để tính toán cho nhanh
    prior_orders['time_of_day'] = prior_orders['order_hour_of_day'].apply(get_time_of_day)
    prior_orders['day_type'] = prior_orders['order_dow'].apply(get_day_type)
    
    # 2. Tính toán các đặc trưng hành vi và thời gian (User Context)
    # Group theo user_id tren bang ORDERS (vì tính theo đơn hàng)
    user_context = prior_orders.groupby('user_id').agg({
        'order_id': 'count', # Tổng số đơn hàng (Frequency)
        'days_since_prior_order': 'mean', # Recency (trung bình)
    }).rename(columns={'order_id': 'total_orders', 'days_since_prior_order': 'avg_frequency'})
    
    # Tính Ratios cho Thời gian (Time Ratios)
    # Dummy encoding cho time_of_day và day_type
    time_dummies = pd.get_dummies(prior_orders['time_of_day'], prefix='ratio')
    day_dummies = pd.get_dummies(prior_orders['day_type'], prefix='ratio')
    
    # Gán vào bảng orders tạm thời
    prior_orders_dummies = pd.concat([prior_orders['user_id'], time_dummies, day_dummies], axis=1)
    
    # Tính trung bình theo user (Mean của cột 0/1 chính là Tỷ lệ/Ratio)
    time_ratios = prior_orders_dummies.groupby('user_id').mean()
    
    # 3. Tính toán Basket Size và Reorder (Tổng hợp từ order_products)
    merged = pd.merge(order_products_prior, prior_orders[['order_id', 'user_id']], on='order_id')
    
    product_stats = merged.groupby('user_id').agg({
        'reordered': 'mean', # Reorder Ratio
        'product_id': 'count' # Tạm tính để chia cho total_orders
    }).rename(columns={'reordered': 'reorder_ratio', 'product_id': 'total_items_tmp'})
    
    # Ghép các bảng lại
    user_profiles = user_context.join(time_ratios).join(product_stats)
    
    # Tính Avg Basket Size và BỎ cột total_items_tmp (Tránh đa cộng tuyến)
    user_profiles['avg_basket_size'] = user_profiles['total_items_tmp'] / user_profiles['total_orders']
    user_profiles = user_profiles.drop(columns=['total_items_tmp'])
    
    # 4. Đặc trưng sở thích theo Department (Tỷ lệ phần trăm) - Product Preferences
    merged_dept = pd.merge(merged, products[['product_id', 'department_id']], on='product_id')
    dept_pivot = pd.crosstab(merged_dept['user_id'], merged_dept['department_id'])
    # Chuyển đổi sang tỷ lệ phần trăm (chia cho tổng số sản phẩm user đã mua)
    dept_ratios = dept_pivot.div(dept_pivot.sum(axis=1), axis=0)
    dept_ratios.columns = [f'dept_{col}_ratio' for col in dept_ratios.columns]
    
    # Ghép tất cả lại: Context + Product Prefs
    final_user_features = user_profiles.join(dept_ratios)
    
    # FillNA nếu có (ví dụ user chưa bao giờ mua buổi sáng thì ratio là NaN hoặc 0)
    final_user_features = final_user_features.fillna(0)
    
    return final_user_features

def create_transactions(orders, order_products_prior, products):
    print("Creating Contextual Transactions for Association Rules...")
    
    # Chỉ lấy dữ liệu Prior
    prior_orders = orders[orders['eval_set'] == 'prior'].copy()
    
    # Tạo nhãn Context
    prior_orders['time_of_day'] = prior_orders['order_hour_of_day'].apply(get_time_of_day)
    prior_orders['day_type'] = prior_orders['order_dow'].apply(get_day_type)
    prior_orders['context_label'] = prior_orders['day_type'] + '_' + prior_orders['time_of_day']
    
    # Merge để lấy tên sản phẩm
    merged = pd.merge(order_products_prior, prior_orders[['order_id', 'context_label']], on='order_id')
    merged = pd.merge(merged, products[['product_id', 'product_name']], on='product_id')
    
    # Group by Order ID và Context để tạo list sản phẩm
    transactions = merged.groupby(['order_id', 'context_label'])['product_name'].apply(list).reset_index()
    
    return transactions

def main():
    # 1. Load Data
    orders, products, departments, order_products_prior = load_data()
    
    # 2. Clean Data
    orders = clean_data(orders)
    
    # 3. Create User Features (KMeans Input)
    user_features = create_user_features(orders, order_products_prior, products, departments)
    print(f"User features created shape: {user_features.shape}")
    user_features_path = os.path.join(OUTPUT_DIR, 'user_features.csv')
    user_features.to_csv(user_features_path)
    print(f"Saved user features to {user_features_path}")
    
    # 4. Create Contextual Transactions (Assoc Rules Input)
    transactions = create_transactions(orders, order_products_prior, products)
    print(f"Transactions created shape: {transactions.shape}")
    
    # Lưu transactions thành file riêng (có thể dùng CSV hoặc định dạng khác xử lý text)
    # Ở đây lưu CSV để dễ đọc, mỗi dòng 1 list sản phẩm
    trans_path = os.path.join(OUTPUT_DIR, 'transactions_by_context.csv')
    transactions.to_csv(trans_path, index=False)
    print(f"Saved transactions to {trans_path}")
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
