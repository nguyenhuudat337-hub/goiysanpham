import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "header",
            "metadata": {},
            "source": [
                "# Khám Phá & Tiền Xử Lý Dữ Liệu: user_features.csv\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "imports",
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "\n",
                "# Cấu hình hiển thị\n",
                "pd.set_option('display.max_columns', None)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "sns.set_palette('husl')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "load_header",
            "metadata": {},
            "source": ["## 1. Load Dữ Liệu"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "load_data",
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('D:\\\\MachineLearning\\\\processed_data\\\\user_features.csv')\n",
                "print(f'Shape: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "null_header",
            "metadata": {},
            "source": ["## 2. Kiểm Tra Giá Trị Null"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "null_check",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kiểm tra null\n",
                "null_counts = df.isnull().sum()\n",
                "print(f'Tổng số null: {null_counts.sum()}')\n",
                "print('\\nNull theo cột:')\n",
                "print(null_counts[null_counts > 0] if null_counts.sum() > 0 else 'Không có null!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "info",
            "metadata": {},
            "outputs": [],
            "source": ["df.info()"]
        },
        {
            "cell_type": "markdown",
            "id": "stats_header",
            "metadata": {},
            "source": ["## 3. Thống Kê Mô Tả"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "describe",
            "metadata": {},
            "outputs": [],
            "source": ["df.describe().T.round(4)"]
        },
        {
            "cell_type": "markdown",
            "id": "dist_header",
            "metadata": {},
            "source": ["## 4. Phân Phối Các Đặc Trưng Chính"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "main_features_hist",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Các đặc trưng chính (không phải ratio)\n",
                "main_features = ['total_orders', 'avg_frequency', 'reorder_ratio', 'avg_basket_size']\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
                "for ax, col in zip(axes.flatten(), main_features):\n",
                "    ax.hist(df[col], bins=50, edgecolor='white', alpha=0.7)\n",
                "    ax.set_title(f'Phân phối {col}')\n",
                "    ax.set_xlabel(col)\n",
                "    ax.set_ylabel('Số lượng')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "outlier_header",
            "metadata": {},
            "source": ["## 5. Phát Hiện Outliers (Boxplot)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "boxplot",
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
                "for ax, col in zip(axes, main_features):\n",
                "    ax.boxplot(df[col])\n",
                "    ax.set_title(col)\n",
                "plt.suptitle('Boxplot - Phát hiện Outliers', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "outlier_stats",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Thống kê outliers theo IQR\n",
                "def count_outliers(series):\n",
                "    Q1, Q3 = series.quantile([0.25, 0.75])\n",
                "    IQR = Q3 - Q1\n",
                "    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR\n",
                "    return ((series < lower) | (series > upper)).sum()\n",
                "\n",
                "outlier_counts = {col: count_outliers(df[col]) for col in main_features}\n",
                "print('Số lượng outliers theo IQR:')\n",
                "for col, count in outlier_counts.items():\n",
                "    print(f'  {col}: {count:,} ({count/len(df)*100:.2f}%)')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "corr_header",
            "metadata": {},
            "source": ["## 6. Ma Trận Tương Quan"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "correlation",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Chỉ lấy các cột số (loại bỏ user_id nếu có)\n",
                "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
                "if 'user_id' in numeric_cols:\n",
                "    numeric_cols.remove('user_id')\n",
                "\n",
                "corr_matrix = df[numeric_cols].corr()\n",
                "\n",
                "plt.figure(figsize=(14, 10))\n",
                "sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, \n",
                "            linewidths=0.5, square=True)\n",
                "plt.title('Ma Trận Tương Quan', fontsize=14)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "high_corr",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tìm các cặp có tương quan cao (> 0.8)\n",
                "high_corr_pairs = []\n",
                "for i in range(len(corr_matrix.columns)):\n",
                "    for j in range(i+1, len(corr_matrix.columns)):\n",
                "        if abs(corr_matrix.iloc[i, j]) > 0.8:\n",
                "            high_corr_pairs.append((\n",
                "                corr_matrix.columns[i], \n",
                "                corr_matrix.columns[j], \n",
                "                corr_matrix.iloc[i, j]\n",
                "            ))\n",
                "\n",
                "print(f'Các cặp có tương quan > 0.8: {len(high_corr_pairs)}')\n",
                "for c1, c2, val in high_corr_pairs:\n",
                "    print(f'  {c1} <-> {c2}: {val:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "preprocess_header",
            "metadata": {},
            "source": ["## 7. Tiền Xử Lý Cho Mô Hình"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "prepare_features",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Chuẩn bị dữ liệu cho mô hình\n",
                "# Loại bỏ user_id (chỉ là định danh, không phải feature)\n",
                "feature_cols = [col for col in df.columns if col != 'user_id']\n",
                "X = df[feature_cols].copy()\n",
                "\n",
                "print(f'Features shape: {X.shape}')\n",
                "print(f'Các features: {list(X.columns)}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "scaling",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Chuẩn hóa dữ liệu với StandardScaler\n",
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "\n",
                "# Tạo DataFrame với dữ liệu đã scale\n",
                "X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)\n",
                "\n",
                "print('Dữ liệu sau khi chuẩn hóa:')\n",
                "X_scaled_df.describe().T[['mean', 'std', 'min', 'max']].round(4)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "save_scaled",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Lưu dữ liệu đã xử lý\n",
                "output_path = 'D:\\\\MachineLearning\\\\processed_data\\\\user_features_scaled.csv'\n",
                "X_scaled_df.to_csv(output_path, index=False)\n",
                "print(f'Đã lưu dữ liệu đã chuẩn hóa tại: {output_path}')\n",
                "\n",
                "# Lưu cả user_id để mapping sau này\n",
                "user_ids = df['user_id'].values\n",
                "np.save('D:\\\\MachineLearning\\\\processed_data\\\\user_ids.npy', user_ids)\n",
                "print('Đã lưu user_ids mapping.')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "summary_header",
            "metadata": {},
            "source": ["## 8. Tổng Kết"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "summary",
            "metadata": {},
            "outputs": [],
            "source": [
                "print('='*50)\n",
                "print('TỔNG KẾT KHÁM PHÁ & TIỀN XỬ LÝ DỮ LIỆU')\n",
                "print('='*50)\n",
                "print(f'Số lượng users: {len(df):,}')\n",
                "print(f'Số lượng features: {len(feature_cols)}')\n",
                "print(f'Giá trị null: {df.isnull().sum().sum()}')\n",
                "print(f'Scaling method: StandardScaler')\n",
                "print(f'Output: user_features_scaled.csv')\n",
                "print('='*50)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('D:/MachineLearning/xuly.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook created successfully!")
