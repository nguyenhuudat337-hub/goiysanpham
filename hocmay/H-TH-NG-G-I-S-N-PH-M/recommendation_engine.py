"""
Contextual Hybrid Recommendation Engine
Hệ thống gợi ý sản phẩm kết hợp phân cụm, luật kết hợp và sản phẩm phổ biến
"""

import pandas as pd
import numpy as np
import os
import ast
import joblib
from collections import defaultdict

# Đường dẫn
DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'
MODELS_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models'
RULES_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models/rules'


class HybridRecommender:
    """
    Hệ thống gợi ý sản phẩm Hybrid kết hợp 3 lớp:
    
    Layer 1: Association Rules (weight cao nhất)
             - Luật kết hợp FP-Growth theo ngữ cảnh
             - Chính xác nhất nhưng coverage thấp
    
    Layer 2: Context-Aware Popular Items (weight trung bình)
             - Sản phẩm phổ biến theo ngữ cảnh thời gian
             - Coverage tốt, có tính cá nhân hóa theo context
    
    Layer 3: Global Popular Items (weight thấp nhất)
             - Fallback cuối cùng, đảm bảo luôn có recommendations
    """
    
    # Weights cho từng layer
    WEIGHT_ASSOCIATION_RULES = 1.0
    WEIGHT_CONTEXT_POPULAR = 0.5
    WEIGHT_GLOBAL_POPULAR = 0.3
    WEIGHT_COOCCURRENCE = 0.4
    
    def __init__(self, kmeans_model=None, scaler=None, rules_dict=None):
        """
        Initialize hybrid recommender
        """
        self.kmeans = self._load_model(kmeans_model, 'kmeans_model.pkl')
        self.scaler = self._load_model(scaler, 'scaler.pkl')
        self.rules = self._load_rules(rules_dict)
        
        # Load popular items
        self.global_popular = self._load_global_popular()
        self.context_popular = self._load_context_popular()
        self.cooccurrence = self._load_cooccurrence()
        
        # Stats
        self._print_stats()
    
    def _print_stats(self):
        """In thống kê về data đã load"""
        print(f"Loaded {len(self.rules)} context rules")
        print(f"Loaded {len(self.global_popular)} global popular items")
        print(f"Loaded {len(self.context_popular)} context popular configs")
        print(f"Loaded {len(self.cooccurrence)} product co-occurrence entries")
        
    def _load_model(self, model_or_path, default_filename):
        """Load model từ object hoặc file path"""
        if model_or_path is None:
            path = os.path.join(MODELS_DIR, default_filename)
            if os.path.exists(path):
                return joblib.load(path)
            return None
        elif isinstance(model_or_path, str):
            return joblib.load(model_or_path)
        return model_or_path
    
    def _load_rules(self, rules_or_path):
        """Load rules từ dict hoặc thư mục"""
        if rules_or_path is None:
            rules_or_path = RULES_DIR
            
        if isinstance(rules_or_path, str):
            rules_dict = {}
            if os.path.exists(rules_or_path):
                for filename in os.listdir(rules_or_path):
                    if filename.endswith('_rules.csv'):
                        context = filename.replace('_rules.csv', '')
                        path = os.path.join(rules_or_path, filename)
                        rules_dict[context] = pd.read_csv(path)
            return rules_dict
        return rules_or_path
    
    def _load_global_popular(self):
        """Load global popular items"""
        path = os.path.join(MODELS_DIR, 'global_popular.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df['product_name'].tolist()
        return []
    
    def _load_context_popular(self):
        """Load context-aware popular items"""
        path = os.path.join(MODELS_DIR, 'context_popular.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Group by context
            context_dict = {}
            for context in df['context'].unique():
                context_df = df[df['context'] == context].sort_values('rank')
                context_dict[context] = context_df['product_name'].tolist()
            return context_dict
        return {}
    
    def _load_cooccurrence(self):
        """Load product co-occurrence data"""
        path = os.path.join(MODELS_DIR, 'product_cooccurrence.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Group by product
            cooc_dict = {}
            for product in df['product'].unique():
                product_df = df[df['product'] == product].sort_values('cooccurrence_count', ascending=False)
                cooc_dict[product] = list(zip(
                    product_df['related_product'].tolist(),
                    product_df['cooccurrence_count'].tolist()
                ))
            return cooc_dict
        return {}
    
    def get_context(self, hour, day_of_week):
        """
        Xác định ngữ cảnh từ giờ và ngày trong tuần
        """
        if day_of_week <= 1:
            day_type = 'Weekend'
        else:
            day_type = 'Weekday'
        
        if 5 <= hour < 12:
            time_of_day = 'Morning'
        elif 12 <= hour < 18:
            time_of_day = 'Afternoon'
        elif 18 <= hour < 24:
            time_of_day = 'Evening'
        else:
            time_of_day = 'Night'
        
        return f'{day_type}_{time_of_day}'
    
    def _parse_itemset(self, itemset_str):
        """Parse itemset từ string về list"""
        if isinstance(itemset_str, list):
            return itemset_str
        try:
            return ast.literal_eval(itemset_str)
        except:
            return []
    
    def _get_association_rules_recs(self, current_basket, context):
        """
        Layer 1: Gợi ý từ Association Rules
        
        Returns:
        --------
        dict: {product: score}
        """
        if context not in self.rules or len(self.rules[context]) == 0:
            return {}
        
        rules = self.rules[context]
        current_basket_set = set(current_basket)
        
        product_scores = {}
        
        for _, rule in rules.iterrows():
            antecedents = set(self._parse_itemset(rule['antecedents']))
            consequents = self._parse_itemset(rule['consequents'])
            
            # Kiểm tra nếu antecedents là tập con của giỏ hàng
            if antecedents.issubset(current_basket_set):
                for product in consequents:
                    if product not in current_basket_set:
                        # Score = confidence * lift (normalized)
                        score = rule['confidence'] * min(rule['lift'], 10) / 10
                        product_scores[product] = max(product_scores.get(product, 0), score)
        
        return product_scores
    
    def _get_cooccurrence_recs(self, current_basket):
        """
        Gợi ý từ Co-occurrence (sản phẩm thường mua cùng)
        
        Returns:
        --------
        dict: {product: score}
        """
        current_basket_set = set(current_basket)
        product_scores = defaultdict(float)
        
        for product in current_basket:
            if product in self.cooccurrence:
                related = self.cooccurrence[product]
                max_count = related[0][1] if related else 1
                
                for related_product, count in related[:20]:
                    if related_product not in current_basket_set:
                        # Normalize score
                        score = count / max_count
                        product_scores[related_product] = max(product_scores[related_product], score)
        
        return dict(product_scores)
    
    def _get_context_popular_recs(self, context, exclude_products):
        """
        Layer 2: Gợi ý sản phẩm phổ biến theo context
        
        Returns:
        --------
        dict: {product: score}
        """
        if context not in self.context_popular:
            return {}
        
        popular_list = self.context_popular[context]
        product_scores = {}
        
        for rank, product in enumerate(popular_list):
            if product not in exclude_products:
                # Score giảm dần theo rank (rank 1 = score 1.0, rank 100 = score ~0)
                score = 1.0 - (rank / len(popular_list))
                product_scores[product] = score
        
        return product_scores
    
    def _get_global_popular_recs(self, exclude_products):
        """
        Layer 3: Gợi ý sản phẩm phổ biến toàn cục (fallback)
        
        Returns:
        --------
        dict: {product: score}
        """
        product_scores = {}
        
        for rank, product in enumerate(self.global_popular):
            if product not in exclude_products:
                score = 1.0 - (rank / len(self.global_popular))
                product_scores[product] = score
        
        return product_scores
    
    def recommend(self, user_id=None, current_basket=None, hour=None, day_of_week=None, 
                  context=None, top_n=10):
        """
        Hybrid Multi-Layer Recommendation
        
        Parameters:
        -----------
        user_id: int - ID người dùng (optional, cho tương lai)
        current_basket: list - Sản phẩm trong giỏ hàng
        hour: int - Giờ hiện tại (0-23)
        day_of_week: int - Ngày trong tuần (0-6)
        context: str - Ngữ cảnh (nếu đã biết)
        top_n: int - Số lượng gợi ý
        
        Returns:
        --------
        recommendations: list of dicts
        """
        # Xác định context
        if context is None and hour is not None and day_of_week is not None:
            context = self.get_context(hour, day_of_week)
        elif context is None:
            context = 'Weekday_Afternoon'  # Default
        
        current_basket = current_basket or []
        current_basket_set = set(current_basket)
        
        # ========================================
        # Layer 1: Association Rules
        # ========================================
        rules_scores = self._get_association_rules_recs(current_basket, context)
        
        # ========================================
        # Layer 1.5: Co-occurrence (bonus)
        # ========================================
        cooc_scores = self._get_cooccurrence_recs(current_basket)
        
        # ========================================
        # Layer 2: Context Popular
        # ========================================
        context_pop_scores = self._get_context_popular_recs(context, current_basket_set)
        
        # ========================================
        # Layer 3: Global Popular
        # ========================================
        global_pop_scores = self._get_global_popular_recs(current_basket_set)
        
        # ========================================
        # Combine scores with weights
        # ========================================
        final_scores = defaultdict(lambda: {'score': 0, 'sources': []})
        
        # Apply rules scores
        for product, score in rules_scores.items():
            weighted = score * self.WEIGHT_ASSOCIATION_RULES
            final_scores[product]['score'] += weighted
            final_scores[product]['sources'].append(f'rules({score:.3f})')
        
        # Apply co-occurrence scores
        for product, score in cooc_scores.items():
            weighted = score * self.WEIGHT_COOCCURRENCE
            final_scores[product]['score'] += weighted
            if 'cooc' not in str(final_scores[product]['sources']):
                final_scores[product]['sources'].append(f'cooc({score:.3f})')
        
        # Apply context popular scores (only if not in rules)
        for product, score in context_pop_scores.items():
            weighted = score * self.WEIGHT_CONTEXT_POPULAR
            if product not in rules_scores:
                final_scores[product]['score'] += weighted
                final_scores[product]['sources'].append(f'ctx_pop({score:.3f})')
        
        # Apply global popular scores (only if not in rules or context)
        for product, score in global_pop_scores.items():
            weighted = score * self.WEIGHT_GLOBAL_POPULAR
            if product not in rules_scores and product not in context_pop_scores:
                final_scores[product]['score'] += weighted
                final_scores[product]['sources'].append(f'global({score:.3f})')
        
        # ========================================
        # Sort and format output
        # ========================================
        sorted_products = sorted(
            final_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:top_n]
        
        recommendations = []
        for product, data in sorted_products:
            # Determine primary source
            sources = data['sources']
            if any('rules' in s for s in sources):
                primary_source = 'association_rules'
            elif any('cooc' in s for s in sources):
                primary_source = 'cooccurrence'
            elif any('ctx_pop' in s for s in sources):
                primary_source = 'context_popular'
            else:
                primary_source = 'global_popular'
            
            recommendations.append({
                'product': product,
                'score': round(data['score'], 4),
                'source': primary_source,
                'context': context,
                'details': ', '.join(sources)
            })
        
        return recommendations


# Backward compatibility alias
class ContextualRecommender(HybridRecommender):
    """Alias for backward compatibility"""
    pass


def demo():
    """Demo sử dụng hybrid recommender"""
    print("="*60)
    print("HYBRID RECOMMENDER DEMO")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing hybrid recommender...")
    recommender = HybridRecommender()
    
    # Demo 1: Với basket có association rules
    print("\n" + "-"*40)
    print("Demo 1: Basket with Association Rules Coverage")
    print("-"*40)
    
    sample_basket = ['Banana', 'Organic Strawberries']
    context = 'Weekday_Morning'
    
    print(f"Basket: {sample_basket}")
    print(f"Context: {context}")
    
    recs = recommender.recommend(
        current_basket=sample_basket,
        context=context,
        top_n=10
    )
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['product']}")
        print(f"     Score: {rec['score']:.4f} | Source: {rec['source']}")
        print(f"     Details: {rec['details']}")
    
    # Demo 2: Với basket không có association rules
    print("\n" + "-"*40)
    print("Demo 2: Basket without Association Rules (fallback)")
    print("-"*40)
    
    sample_basket = ['Very Rare Product XYZ', 'Another Uncommon Item']
    context = 'Weekend_Evening'
    
    print(f"Basket: {sample_basket}")
    print(f"Context: {context}")
    
    recs = recommender.recommend(
        current_basket=sample_basket,
        context=context,
        top_n=5
    )
    
    print(f"\nRecommendations (should use fallback):")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['product']}")
        print(f"     Score: {rec['score']:.4f} | Source: {rec['source']}")
    
    # Demo 3: Empty basket
    print("\n" + "-"*40)
    print("Demo 3: Empty Basket")
    print("-"*40)
    
    recs = recommender.recommend(
        current_basket=[],
        context='Weekday_Afternoon',
        top_n=5
    )
    
    print("Recommendations for empty basket:")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['product']} (Source: {rec['source']})")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    demo()
