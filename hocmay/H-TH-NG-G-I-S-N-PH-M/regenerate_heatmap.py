"""
Regenerate Heatmap with Department Names and Cluster Labels
Vẽ lại heatmap với tên department và nhãn cluster có ý nghĩa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Đường dẫn
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'
MODELS_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models'

# Tên các cụm dựa trên đặc điểm thực tế từ heatmap
CLUSTER_NAMES = {
    0: 'Đồ uống &\nGia dụng',         # beverages_ratio=1.26, household_ratio=1.01, personal_care=0.78
    1: 'Mua sắm\nCuối tuần',          # ratio_Weekend=1.23, ratio_Weekday=-1.23
    2: 'Khách hàng\nTrung thành',     # reorder_ratio=1.12, total_orders=1.29, avg_frequency=-0.88
    3: 'Khách hàng\nMới/Thỉnh thoảng' # reorder_ratio=-0.46, total_orders=-0.41, ratio_Weekday=0.63
}

# Tên tiếng Việt cho cụm (dùng trong báo cáo)
CLUSTER_NAMES_VI = {
    0: 'Nhóm Đồ uống & Gia dụng',
    1: 'Nhóm Mua sắm Cuối tuần', 
    2: 'Nhóm Khách hàng Trung thành',
    3: 'Nhóm Khách hàng Mới/Thỉnh thoảng'
}


def load_cluster_labels():
    """Load cluster labels từ file đã lưu hoặc tính lại"""
    user_ids_path = os.path.join(PROCESSED_DIR, 'user_ids.npy')
    
    # Thử load model KMeans nếu có
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
        df_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'user_features_scaled.csv'))
        # Bỏ cột user_id nếu có
        feature_cols = [c for c in df_scaled.columns if c != 'user_id']
        return kmeans.labels_
    else:
        print("Warning: KMeans model not found. Please ensure kmeans_model.pkl exists.")
        return None


def create_heatmap(df_scaled, cluster_labels, output_path):
    """
    Tạo heatmap hiển thị đặc điểm trung bình của mỗi cluster với tên có ý nghĩa
    """
    # Thêm cluster labels
    df_scaled['cluster'] = cluster_labels
    
    # Lấy các cột feature (bỏ user_id và cluster)
    feature_cols = [c for c in df_scaled.columns if c not in ['user_id', 'cluster']]
    
    # Tính giá trị trung bình theo cluster
    cluster_means = df_scaled.groupby('cluster')[feature_cols].mean()
    
    # Đổi tên index thành tên cluster có ý nghĩa
    cluster_means.index = [CLUSTER_NAMES.get(i, f'Cụm {i}') for i in cluster_means.index]
    
    # Tạo heatmap
    plt.figure(figsize=(18, 12))
    
    # Tạo heatmap với tên cluster
    ax = sns.heatmap(
        cluster_means.T,  # Transpose để features nằm dọc
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Giá trị trung bình (chuẩn hóa)'},
        annot_kws={'size': 9}
    )
    
    plt.title('Heatmap - Đặc điểm trung bình của các nhóm khách hàng', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Nhóm khách hàng', fontsize=13)
    plt.ylabel('Đặc trưng', fontsize=13)
    
    # Xoay nhãn x-axis để dễ đọc
    plt.xticks(rotation=0, ha='center', fontsize=11)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")
    
    return cluster_means


def main():
    print("=" * 50)
    print("REGENERATE HEATMAP WITH DEPARTMENT NAMES")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading scaled features...")
    df_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'user_features_scaled.csv'))
    print(f"   Shape: {df_scaled.shape}")
    
    # 2. Load hoặc tính cluster labels
    print("\n2. Loading cluster labels...")
    cluster_labels = load_cluster_labels()
    
    if cluster_labels is None:
        print("   Cannot find cluster labels. Running KMeans...")
        from sklearn.cluster import KMeans
        
        feature_cols = [c for c in df_scaled.columns if c != 'user_id']
        X = df_scaled[feature_cols].values
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Lưu model
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
        print("   Saved KMeans model.")
    
    print(f"   Found {len(np.unique(cluster_labels))} clusters")
    
    # 3. Tạo heatmap mới
    print("\n3. Creating new heatmap...")
    output_path = os.path.join(DATA_DIR, 'cluster_heatmap_named.png')
    cluster_means = create_heatmap(df_scaled, cluster_labels, output_path)
    
    # 4. In thống kê
    print("\n4. Cluster characteristics:")
    print("-" * 40)
    for i, cluster_name in enumerate(cluster_means.index):
        # Use ASCII-safe version for console
        safe_name = f"Cluster {i}"
        print(f"\n{safe_name}:")
        top_features = cluster_means.loc[cluster_name].nlargest(3)
        for feat, val in top_features.items():
            print(f"   + {feat}: {val:.3f}")
        bottom_features = cluster_means.loc[cluster_name].nsmallest(3)
        for feat, val in bottom_features.items():
            print(f"   - {feat}: {val:.3f}")
    
    print("\n" + "=" * 50)
    print("COMPLETED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
