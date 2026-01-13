"""
Evaluation Module - Hybrid Recommender
Đánh giá hiệu quả hệ thống gợi ý sản phẩm Hybrid
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

# Đường dẫn
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'

# Import recommender
try:
    from recommendation_engine import HybridRecommender
except ImportError:
    print("Please ensure recommendation_engine.py is in the same directory")


def get_time_of_day(hour):
    """Phân loại khoảng thời gian"""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


def get_day_type(dow):
    """Phân loại ngày trong tuần"""
    return 'Weekend' if dow <= 1 else 'Weekday'


def load_test_data():
    """Load test data (tập train của Instacart)"""
    print("Loading test data...")
    
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    test_orders = orders[orders['eval_set'] == 'train'].copy()
    
    order_products = pd.read_csv(os.path.join(DATA_DIR, 'order_products__train.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    
    merged = pd.merge(order_products, test_orders, on='order_id')
    merged = pd.merge(merged, products[['product_id', 'product_name']], on='product_id')
    
    merged['time_of_day'] = merged['order_hour_of_day'].apply(get_time_of_day)
    merged['day_type'] = merged['order_dow'].apply(get_day_type)
    merged['context'] = merged['day_type'] + '_' + merged['time_of_day']
    
    print(f"Test orders: {len(test_orders)}")
    print(f"Test products: {len(merged)}")
    
    return merged


def precision_at_k(recommended, actual, k):
    """Precision@K = |recommended ∩ actual| / K"""
    if k == 0:
        return 0.0
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    return len(recommended_k & actual_set) / k


def recall_at_k(recommended, actual, k):
    """Recall@K = |recommended ∩ actual| / |actual|"""
    if len(actual) == 0:
        return 0.0
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    return len(recommended_k & actual_set) / len(actual_set)


def hit_rate(recommended, actual):
    """Hit Rate = 1 if có ít nhất 1 sản phẩm đúng"""
    return 1.0 if len(set(recommended) & set(actual)) > 0 else 0.0


def mrr(recommended, actual):
    """Mean Reciprocal Rank"""
    actual_set = set(actual)
    for rank, item in enumerate(recommended, 1):
        if item in actual_set:
            return 1.0 / rank
    return 0.0


def evaluate_hybrid_recommender(recommender, test_data, k=10, sample_size=1000):
    """
    Đánh giá Hybrid Recommender với coverage tracking
    """
    print(f"\nEvaluating HYBRID Recommender on {sample_size} samples...")
    
    # Group by order
    orders_products = test_data.groupby(['order_id', 'context', 'user_id']).agg({
        'product_name': list
    }).reset_index()
    
    if len(orders_products) > sample_size:
        orders_products = orders_products.sample(n=sample_size, random_state=42)
    
    # Metrics
    precisions, recalls, hit_rates, mrrs = [], [], [], []
    source_counts = defaultdict(int)
    orders_with_recs = 0
    orders_with_rules = 0
    
    for _, row in orders_products.iterrows():
        actual_products = row['product_name']
        context = row['context']
        
        if len(actual_products) < 2:
            continue
        
        mid = len(actual_products) // 2
        basket = actual_products[:mid]
        target = actual_products[mid:]
        
        # Get recommendations
        recs = recommender.recommend(
            current_basket=basket,
            context=context,
            top_n=k
        )
        
        recommended = [r['product'] for r in recs]
        
        if len(recommended) > 0:
            orders_with_recs += 1
            
            # Track sources
            for r in recs:
                source_counts[r['source']] += 1
            
            if any(r['source'] == 'association_rules' for r in recs):
                orders_with_rules += 1
        
        # Calculate metrics (even if no recs, count as 0)
        precisions.append(precision_at_k(recommended, target, k))
        recalls.append(recall_at_k(recommended, target, k))
        hit_rates.append(hit_rate(recommended, target))
        mrrs.append(mrr(recommended, target))
    
    n_evaluated = len(precisions)
    coverage = orders_with_recs / n_evaluated if n_evaluated > 0 else 0
    rules_coverage = orders_with_rules / n_evaluated if n_evaluated > 0 else 0
    
    metrics = {
        'precision@k': np.mean(precisions) if precisions else 0,
        'recall@k': np.mean(recalls) if recalls else 0,
        'hit_rate': np.mean(hit_rates) if hit_rates else 0,
        'mrr': np.mean(mrrs) if mrrs else 0,
        'coverage': coverage,
        'rules_coverage': rules_coverage,
        'n_evaluated': n_evaluated,
        'k': k,
        'source_counts': dict(source_counts)
    }
    
    if metrics['precision@k'] + metrics['recall@k'] > 0:
        metrics['f1@k'] = 2 * metrics['precision@k'] * metrics['recall@k'] / \
                         (metrics['precision@k'] + metrics['recall@k'])
    else:
        metrics['f1@k'] = 0
    
    return metrics


def evaluate_popular_baseline(test_data, k=10, sample_size=1000):
    """
    Baseline: Popular items - FAIR comparison (same target)
    """
    print(f"\nEvaluating Popular Items BASELINE on {sample_size} samples...")
    
    # Top-k popular products
    product_counts = test_data['product_name'].value_counts()
    popular_products = product_counts.head(k).index.tolist()
    
    print(f"Top {k} popular products:")
    for i, prod in enumerate(popular_products[:5], 1):
        print(f"  {i}. {prod}")
    
    # Group by order
    orders_products = test_data.groupby(['order_id']).agg({
        'product_name': list
    }).reset_index()
    
    if len(orders_products) > sample_size:
        orders_products = orders_products.sample(n=sample_size, random_state=42)
    
    precisions, recalls, hit_rates_list, mrrs = [], [], [], []
    
    for _, row in orders_products.iterrows():
        actual_products = row['product_name']
        
        # FAIR: Use same split as hybrid evaluation
        if len(actual_products) < 2:
            continue
        
        mid = len(actual_products) // 2
        basket = actual_products[:mid]
        target = actual_products[mid:]  # Same target as hybrid!
        
        # Filter popular that's not in basket
        filtered_popular = [p for p in popular_products if p not in basket][:k]
        
        precisions.append(precision_at_k(filtered_popular, target, k))
        recalls.append(recall_at_k(filtered_popular, target, k))
        hit_rates_list.append(hit_rate(filtered_popular, target))
        mrrs.append(mrr(filtered_popular, target))
    
    metrics = {
        'precision@k': np.mean(precisions),
        'recall@k': np.mean(recalls),
        'hit_rate': np.mean(hit_rates_list),
        'mrr': np.mean(mrrs),
        'coverage': 1.0,  # Always 100% by design
        'n_evaluated': len(precisions),
        'k': k
    }
    
    if metrics['precision@k'] + metrics['recall@k'] > 0:
        metrics['f1@k'] = 2 * metrics['precision@k'] * metrics['recall@k'] / \
                         (metrics['precision@k'] + metrics['recall@k'])
    else:
        metrics['f1@k'] = 0
    
    return metrics


def print_metrics(metrics, title):
    """In metrics đẹp"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"  Samples evaluated: {metrics['n_evaluated']}")
    print(f"  K = {metrics['k']}")
    print(f"  ---")
    print(f"  Precision@{metrics['k']}: {metrics['precision@k']:.4f}")
    print(f"  Recall@{metrics['k']}:    {metrics['recall@k']:.4f}")
    print(f"  F1@{metrics['k']}:        {metrics['f1@k']:.4f}")
    print(f"  Hit Rate:         {metrics['hit_rate']:.4f}")
    print(f"  MRR:              {metrics['mrr']:.4f}")
    print(f"  Coverage:         {metrics.get('coverage', 1.0)*100:.1f}%")
    
    if 'rules_coverage' in metrics:
        print(f"  Rules Coverage:   {metrics['rules_coverage']*100:.1f}%")
    
    if 'source_counts' in metrics:
        print(f"\n  Recommendation Sources:")
        total = sum(metrics['source_counts'].values())
        for source, count in sorted(metrics['source_counts'].items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"    - {source}: {count} ({pct:.1f}%)")


def main():
    print("="*60)
    print("HYBRID RECOMMENDATION SYSTEM EVALUATION")
    print("="*60)
    
    # 1. Load test data
    test_data = load_test_data()
    
    # 2. Initialize hybrid recommender
    print("\nInitializing Hybrid Recommender...")
    recommender = HybridRecommender()
    
    k = 10
    sample_size = 1000
    
    # 3. Evaluate hybrid recommender
    hybrid_metrics = evaluate_hybrid_recommender(
        recommender, test_data, k=k, sample_size=sample_size
    )
    print_metrics(hybrid_metrics, "HYBRID RECOMMENDER")
    
    # 4. Evaluate baseline (fair comparison)
    baseline_metrics = evaluate_popular_baseline(test_data, k=k, sample_size=sample_size)
    print_metrics(baseline_metrics, "POPULAR ITEMS BASELINE (Fair)")
    
    # 5. Comparison
    print("\n" + "="*60)
    print("COMPARISON: Hybrid vs Baseline")
    print("="*60)
    
    for metric in ['precision@k', 'hit_rate', 'mrr']:
        hybrid_val = hybrid_metrics[metric]
        baseline_val = baseline_metrics[metric]
        
        if baseline_val > 0:
            improvement = (hybrid_val - baseline_val) / baseline_val * 100
            symbol = "+" if improvement > 0 else ""
            print(f"  {metric}: {hybrid_val:.4f} vs {baseline_val:.4f} ({symbol}{improvement:.1f}%)")
        else:
            print(f"  {metric}: {hybrid_val:.4f} vs {baseline_val:.4f}")
    
    print(f"\n  Hybrid Coverage: {hybrid_metrics['coverage']*100:.1f}%")
    print(f"  Rules Used: {hybrid_metrics['rules_coverage']*100:.1f}% of recommendations")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
