"""
Rename Department Columns
Đổi tên cột dept_X_ratio thành tên department thực tế trong user_features_scaled.csv
"""

import pandas as pd
import numpy as np
import os

# Đường dẫn
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'

# Bảng ánh xạ department_id -> tên
DEPT_MAPPING = {
    1: 'frozen',
    2: 'other', 
    3: 'bakery',
    4: 'produce',
    5: 'alcohol',
    6: 'international',
    7: 'beverages',
    8: 'pets',
    9: 'dry_goods_pasta',
    10: 'bulk',
    11: 'personal_care',
    12: 'meat_seafood',
    13: 'pantry',
    14: 'breakfast',
    15: 'canned_goods',
    16: 'dairy_eggs',
    17: 'household',
    18: 'babies',
    19: 'snacks',
    20: 'deli',
    21: 'missing'
}

def rename_department_columns(df):
    """
    Đổi tên cột từ dept_X_ratio sang {department_name}_ratio
    """
    rename_dict = {}
    for dept_id, dept_name in DEPT_MAPPING.items():
        old_col = f'dept_{dept_id}_ratio'
        new_col = f'{dept_name}_ratio'
        if old_col in df.columns:
            rename_dict[old_col] = new_col
    
    return df.rename(columns=rename_dict)


def main():
    print("=" * 50)
    print("RENAME DEPARTMENT COLUMNS")
    print("=" * 50)
    
    # 1. Load user_features_scaled.csv
    scaled_path = os.path.join(PROCESSED_DIR, 'user_features_scaled.csv')
    print(f"\n1. Loading {scaled_path}...")
    df_scaled = pd.read_csv(scaled_path)
    print(f"   Shape: {df_scaled.shape}")
    print(f"   Columns before: {[c for c in df_scaled.columns if 'dept' in c][:5]}...")
    
    # 2. Rename columns
    print("\n2. Renaming columns...")
    df_scaled = rename_department_columns(df_scaled)
    print(f"   Columns after: {[c for c in df_scaled.columns if '_ratio' in c and 'dept' not in c][:5]}...")
    
    # 3. Save back
    print(f"\n3. Saving to {scaled_path}...")
    df_scaled.to_csv(scaled_path, index=False)
    print("   Done!")
    
    # 4. Also update user_features.csv (bản gốc chưa scale)
    features_path = os.path.join(PROCESSED_DIR, 'user_features.csv')
    print(f"\n4. Loading {features_path}...")
    df_features = pd.read_csv(features_path)
    df_features = rename_department_columns(df_features)
    df_features.to_csv(features_path, index=False)
    print("   Done!")
    
    print("\n" + "=" * 50)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    # Print sample của các cột mới
    print("\nSample columns with new names:")
    ratio_cols = [c for c in df_scaled.columns if '_ratio' in c]
    for col in ratio_cols[:10]:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
