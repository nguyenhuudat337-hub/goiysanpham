
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime


DATA_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M'
PROCESSED_DIR = '/Users/huudat/hocmay/processed_data'
MODELS_DIR = '/Users/huudat/hocmay/H-TH-NG-G-I-S-N-PH-M/models'

# Import recommender
try:
    from recommendation_engine import ContextualRecommender
except ImportError:
    st.error("Please ensure recommendation_engine.py is in the same directory")
    st.stop()


@st.cache_resource
def load_recommender():
    """Load recommender (cached)"""
    return ContextualRecommender()


@st.cache_data
def load_products():
    """Load danh sách sản phẩm"""
    products_path = os.path.join(DATA_DIR, 'products.csv')
    if os.path.exists(products_path):
        df = pd.read_csv(products_path)
        return df['product_name'].tolist()
    return []


def main():
    # Page config
    st.set_page_config(
        page_title="Smart Product Recommender",
        page_icon="🛒",
        layout="wide"
    )
   
    # Title
    st.title("🛒 Hệ thống gợi ý sản phẩm thông minh")
    st.markdown("*Kết hợp phân cụm và luật kết hợp theo ngữ cảnh*")

    # Sidebar
    st.sidebar.header("⚙️ Cài đặt")
    
    # Context selection
    st.sidebar.subheader("📅 Ngữ cảnh thời gian")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        day_type = st.selectbox(
            "Loại ngày",
            options=['Weekday', 'Weekend'],
            index=0
        )
    
    with col2:
        time_of_day = st.selectbox(
            "Thời gian",
            options=['Morning', 'Afternoon', 'Evening', 'Night'],
            index=1
        )
    
    context = f"{day_type}_{time_of_day}"
    st.sidebar.info(f"**Context:** {context}")
    
    # Top-N
    top_n = st.sidebar.slider("Số sản phẩm gợi ý (Top-N)", 5, 20, 10)
    
    # Load recommender
    recommender = load_recommender()
    
    # Check rules
    if len(recommender.rules) == 0:
        st.warning("⚠️ Chưa có luật kết hợp. Vui lòng chạy `association_rules.py` trước.")
    else:
        st.sidebar.success(f"Đã load {len(recommender.rules)} contexts")
    
    # Main content
    st.header("🛍️ Giỏ hàng của bạn")
    
    # Product input
    products = load_products()
    
    if products:
        # Multi-select cho sản phẩm
        selected_products = st.multiselect(
            "Chọn sản phẩm trong giỏ hàng:",
            options=products[:1000],  # Giới hạn 1000 sản phẩm đầu
            default=[],
            help="Chọn các sản phẩm đã có trong giỏ hàng"
        )
    else:
        # Manual input
        product_input = st.text_input(
            "Nhập tên sản phẩm (phân cách bằng dấu phẩy):",
            value="Banana, Organic Strawberries",
            help="Ví dụ: Banana, Milk, Bread"
        )
        selected_products = [p.strip() for p in product_input.split(',') if p.strip()]
    
    # Display basket
    if selected_products:
        st.write("**Sản phẩm trong giỏ:**")
        cols = st.columns(min(len(selected_products), 5))
        for i, product in enumerate(selected_products[:5]):
            with cols[i]:
                st.markdown(f"🥬 {product}")
        if len(selected_products) > 5:
            st.write(f"... và {len(selected_products) - 5} sản phẩm khác")
    
    # Get recommendations button
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 5px 12px;
        border: 2px solid #1E90FF !important;
        float: right;
        width: auto;
    }
    div.stButton > button:first-child:hover {
        background-color: #0b66c3 !important;
        border: 2px solid #0b66c3 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Gợi ý sản phẩm", type="primary", use_container_width=True):
        if not selected_products:
            st.warning("Vui lòng chọn ít nhất 1 sản phẩm trong giỏ hàng")
        elif len(recommender.rules) == 0:
            st.error("Không có luật kết hợp. Vui lòng chạy association_rules.py trước.")
        else:
            with st.spinner("Đang phân tích..."):
                # Get recommendations
                recommendations = recommender.recommend(
                    current_basket=selected_products,
                    context=context,
                    top_n=top_n
                )
            
            # Display results
            st.header("✨ Sản phẩm gợi ý")
            
            if recommendations:
                # Create DataFrame for display
                rec_df = pd.DataFrame(recommendations)
                rec_df.index = range(1, len(rec_df) + 1)
                rec_df.columns = ['Sản phẩm', 'Điểm', 'Nguồn', 'Ngữ cảnh', 'Chi tiết']
                
                # Display as table
                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=False
                )
                
                # Display as cards
                st.subheader("Chi tiết gợi ý:")
                
                for i, rec in enumerate(recommendations[:5], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}. {rec['product']}**")
                    with col2:
                        st.metric("Score", f"{rec['score']:.3f}")
                    st.markdown("---")
            else:
                st.info("Không tìm thấy gợi ý phù hợp với giỏ hàng hiện tại.")
                st.write("Thử:")
                st.write("- Thêm sản phẩm vào giỏ hàng")
                st.write("- Thay đổi ngữ cảnh thời gian")
                st.write("- Giảm ngưỡng min_support khi chạy association_rules.py")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Hệ thống gợi ý sản phẩm thông minh | 
            Kết hợp K-Means Clustering + FP-Growth Association Rules</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
