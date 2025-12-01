import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ========================================
# CẤU HÌNH TRANG
# ========================================
st.set_page_config(
    page_title="Phân Loại Hoa",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom - Pastel Theme (màu đậm hơn)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffe8f5 0%, #d9ecff 50%, #ffeacc 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b9d 0%, #ff85b3 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 14px;
        border: none;
        font-size: 16px;
        box-shadow: 0 3px 10px rgba(255, 107, 157, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff4d8f 0%, #ff6b9d 100%);
        box-shadow: 0 5px 15px rgba(255, 77, 143, 0.6);
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffb3d9 0%, #ffc9e3 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 107, 157, 0.3);
        margin: 10px 0;
    }
    .metric-card h3 {
        color: #e91e63;
        font-size: 1em;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .metric-card h1 {
        color: #c2185b;
        font-size: 2.5em;
        margin: 0;
        font-weight: bold;
    }
    .info-box {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #ff6b9d;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 10px 0;
    }
    .flower-card {
        background: white;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .flower-card:hover {
        box-shadow: 0 4px 12px rgba(255, 107, 157, 0.4);
        transform: translateY(-3px);
    }
    .prediction-card {
        background: linear-gradient(135deg, #ffe0f0 0%, #ffeaf5 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);
        margin: 15px 0;
        border: 2px solid #ffb3d9;
    }
    .prediction-card h2 {
        color: #c2185b;
        font-size: 3em;
        margin: 10px 0;
        font-weight: bold;
    }
    h1 {
        color: #c2185b;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #e91e63;
        font-size: 1.1em;
        margin-bottom: 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        color: #e91e63;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff6b9d 0%, #ff85b3 100%);
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b9d 0%, #ff85b3 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model_and_metadata():
    """Load model và metadata"""
    try:
        model = tf.saved_model.load('vit_flower_model')
        
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, metadata
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        st.info("💡 Đảm bảo các file sau tồn tại:\n- vit_flower_model/\n- model_metadata.pkl")
        return None, None

# ========================================
# HÀM PREPROCESS & PREDICT
# ========================================
def preprocess_image(image):
    """Tiền xử lý ảnh"""
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Đảm bảo float32
    img_array = np.expand_dims(img_array, 0)
    return img_array

def predict_flower(model, image, class_names, confidence_threshold=50.0):
    """Dự đoán loại hoa với ngưỡng tin cậy"""
    img_array = preprocess_image(image)
    
    infer = model.signatures['serving_default']
    input_name = list(infer.structured_input_signature[1].keys())[0]
    
    # Đảm bảo input là float32 tensor
    predictions = infer(**{input_name: tf.constant(img_array, dtype=tf.float32)})
    
    output_name = list(predictions.keys())[0]
    predictions = predictions[output_name].numpy()[0]
    
    results = []
    for i, conf in enumerate(predictions):
        results.append({
            'class': class_names[i],
            'confidence': float(conf * 100)
        })
    
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Kiểm tra ngưỡng tin cậy
    max_confidence = results[0]['confidence']
    is_valid = max_confidence >= confidence_threshold
    
    return results, is_valid, max_confidence

# ========================================
# VẼ BIỂU ĐỒ
# ========================================
def plot_predictions(predictions):
    """Vẽ biểu đồ độ tin cậy"""
    classes = [p['class'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    # Màu gradient đậm hơn
    colors = ['#ff4d8f', '#ff6b9d', '#ff85b3', '#ff9ec7', '#ffb3d9', '#ffc9e3', '#ffd8ea']
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=confidences,
            orientation='h',
            marker=dict(
                color=colors[:len(predictions)],
                line=dict(color='white', width=2)
            ),
            text=[f"{c:.1f}%" for c in confidences],
            textposition='outside',
            textfont=dict(size=13, color='#c2185b'),
            hovertemplate='<b>%{y}</b><br>Độ tin cậy: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Độ Tin Cậy Các Loại Hoa",
            'font': {'size': 20, 'color': '#c2185b'}
        },
        xaxis_title="Độ tin cậy (%)",
        yaxis_title="",
        height=350,
        template="plotly_white",
        showlegend=False,
        xaxis=dict(range=[0, 105]),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(255, 240, 245, 0.5)'
    )
    
    return fig

def plot_top_prediction_gauge(confidence):
    """Vẽ đồng hồ đo độ tin cậy"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Độ Tin Cậy", 'font': {'size': 22, 'color': '#c2185b', 'weight': 'bold'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#c2185b'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#e91e63"},
            'bar': {'color': "#c2185b", 'thickness': 0.7},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#ffb3d9",
            'steps': [
                {'range': [0, 40], 'color': '#ffe0f0'},
                {'range': [40, 60], 'color': '#ffc9e3'},
                {'range': [60, 80], 'color': '#ffb3d9'},
                {'range': [80, 100], 'color': '#ff9ec7'}
            ],
            'threshold': {
                'line': {'color': "#c2185b", 'width': 4},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(255, 240, 245, 0.3)'
    )
    
    return fig

# ========================================
# MAIN APP
# ========================================
def main():
    # Header
    st.markdown("<h1>🌸 Nhận Diện Loài Hoa</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sử dụng công nghệ Vision Transformer & SVM</p>", unsafe_allow_html=True)
    
    # Load model
    with st.spinner("⏳ Đang tải mô hình..."):
        model, metadata = load_model_and_metadata()
    
    if model is None or metadata is None:
        st.stop()
    
    class_names = metadata['class_names']
    test_accuracy = metadata.get('test_accuracy', 0)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Thông Tin Mô Hình")
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Độ Chính Xác</h3>
            <h1>{test_accuracy*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Cài Đặt")
        confidence_threshold = st.slider(
            "Ngưỡng tin cậy tối thiểu (%)",
            min_value=30.0,
            max_value=90.0,
            value=60.0,
            step=5.0,
            help="Ngưỡng để phát hiện ảnh không hợp lệ hoặc hoa ngoài 7 loài"
        )
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                    padding: 12px; border-radius: 10px; border-left: 4px solid #ff6b9d;'>
            <p style='margin: 0; color: #c2185b; font-weight: 600;'>
                🎯 Ngưỡng: <strong>{confidence_threshold:.0f}%</strong>
            </p>
            <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #666;'>
                Ảnh có độ tin cậy < {confidence_threshold:.0f}% sẽ bị từ chối
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ Ngưỡng hoạt động thế nào?"):
            st.markdown("""
            **Ngưỡng cao (70-90%):** Chặt chẽ
            - ✅ Chỉ chấp nhận hoa rõ ràng trong 7 loài
            - ❌ Từ chối: hoa khác, động vật, đồ vật
            
            **Ngưỡng trung bình (50-70%):** Cân bằng
            - ✅ Chấp nhận hoa không quá rõ
            - ⚠️ Có thể nhầm hoa tương tự
            
            **Ngưỡng thấp (30-50%):** Dễ dãi
            - ✅ Chấp nhận nhiều trường hợp
            - ⚠️ Dễ nhận nhầm
            
            **Khuyến nghị:** 60-70% cho độ chính xác tốt
            """)
        
        st.markdown("---")
        
        st.markdown("### 🌺 Các Loài Hoa")
        for i, flower in enumerate(class_names, 1):
            emoji = ['🌼', '🌻', '🌹', '🌻', '🌷', '🌸', '🏵️'][i-1]
            st.markdown(f"{emoji} **{flower}**")
        
        st.markdown("---")
        
        with st.expander("ℹ️ Chi Tiết Kỹ Thuật"):
            st.markdown("""
            **Kiến trúc:** Vision Transformer  
            **Pretrained:** google/vit-base-patch16-224  
            **Phân loại:** Support Vector Machine  
            **Kích thước:** 224×224 pixels  
            **Số lớp:** 7 loài hoa
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Tải Ảnh Lên")
        uploaded_file = st.file_uploader(
            "Chọn ảnh hoa của bạn",
            type=['jpg', 'jpeg', 'png'],
            help="Hỗ trợ định dạng: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            
            if st.button("🔍 Nhận Diện", type="primary"):
                with st.spinner("🤔 Đang phân tích..."):
                    predictions, is_valid, max_conf = predict_flower(
                        model, image, class_names, confidence_threshold
                    )
                    st.session_state['predictions'] = predictions
                    st.session_state['is_valid'] = is_valid
                    st.session_state['max_confidence'] = max_conf
                    st.session_state['threshold'] = confidence_threshold
                    
                    if is_valid:
                        st.success("✅ Hoàn thành!")
                    else:
                        st.warning(f"⚠️ Phát hiện: Không phải hoa / Hoa không nằm trong danh sách 7 loài đã học (độ tin cậy {max_conf:.1f}%)")
        else:
            st.info("👆 Vui lòng tải lên ảnh hoa để bắt đầu nhận diện")
    
    with col2:
        st.markdown("### 🎯 Kết Quả Dự Đoán")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            is_valid = st.session_state.get('is_valid', True)
            max_conf = st.session_state.get('max_confidence', 0)
            threshold = st.session_state.get('threshold', 50)
            top_pred = predictions[0]
            
            if not is_valid:
                # Ảnh không đạt ngưỡng - không phải hoa hoặc hoa ngoài 7 loài
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fff9e6 0%, #ffe6e6 100%); 
                            padding: 25px; border-radius: 15px; 
                            box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3); margin: 15px 0;
                            border: 3px solid #ff9800;'>
                    <p style='color: #e65100; margin: 0; font-size: 1.8em; text-align: center;'>⚠️</p>
                    <h2 style='color: #e65100; font-size: 2em; margin: 10px 0; text-align: center; font-weight: bold;'>
                        Không Nhận Diện Được
                    </h2>
                    <p style='color: #e65100; text-align: center; margin: 0; font-size: 1.1em;'>
                        Độ tin cậy cao nhất: <strong>{max_conf:.1f}%</strong><br>
                        <span style='font-size: 0.9em;'>(Dưới ngưỡng {threshold:.0f}%)</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.error(f"""
                **🚫 Có thể do:**
                - 🐕 **Không phải hoa:** động vật, người, đồ vật, phong cảnh
                - 🌺 **Hoa ngoài danh sách:** không thuộc 7 loài đã học
                - 📷 **Ảnh không rõ:** mờ, xa, góc chụp khó nhận diện
                - 🎨 **Ảnh vẽ/đồ họa:** không phải ảnh thật
                
                **💡 Giải pháp:**
                - Thử tải ảnh hoa rõ nét hơn (thuộc 7 loài: Daisy, Dandelion, Rose, Sunflower, Tulip, Orchid, Lily)
                - Hoặc giảm ngưỡng xuống {max(30, threshold-10):.0f}% trong thanh bên trái (không khuyến nghị)
                """)
                
                # Vẫn hiển thị kết quả dự đoán để tham khảo
                with st.expander("🔍 Xem dự đoán gần nhất (chỉ tham khảo)"):
                    st.warning("⚠️ Mặc dù dưới ngưỡng, mô hình vẫn đưa ra dự đoán gần nhất:")
                    st.markdown(f"""
                    <div style='background: white; padding: 15px; border-radius: 10px; text-align: center; border: 2px dashed #ff9800;'>
                        <p style='color: #e91e63; margin: 0; font-size: 0.9em;'>Gần nhất với</p>
                        <h3 style='color: #c2185b; margin: 8px 0; font-weight: bold;'>{top_pred['class']}</h3>
                        <p style='color: #e91e63; margin: 0; font-size: 1.2em; font-weight: bold;'>{top_pred['confidence']:.1f}%</p>
                        <p style='color: #999; margin: 5px 0 0 0; font-size: 0.8em;'>Không đủ tin cậy để kết luận</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Ảnh đạt ngưỡng - là hoa trong 7 loài
                st.markdown(f"""
                <div class='prediction-card'>
                    <p style='color: #e91e63; margin: 0; font-size: 1.2em; font-weight: 600;'>✅ Đây là hoa</p>
                    <h2>{top_pred['class']}</h2>
                    <p style='color: #c2185b; font-size: 1.4em; margin: 0; font-weight: bold;'>{top_pred['confidence']:.1f}% tin cậy</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge
                st.plotly_chart(
                    plot_top_prediction_gauge(top_pred['confidence']),
                    use_container_width=True
                )
                
                # Interpretation
                if top_pred['confidence'] >= 80:
                    st.success("🎯 **Độ tin cậy rất cao!** Mô hình cực kỳ chắc chắn về kết quả này.")
                elif top_pred['confidence'] >= 70:
                    st.success("✅ **Độ tin cậy cao!** Mô hình rất chắc chắn về kết quả này.")
                elif top_pred['confidence'] >= 60:
                    st.info("👍 **Độ tin cậy tốt.** Kết quả đáng tin cậy.")
                else:
                    st.warning("⚠️ **Độ tin cậy trung bình.** Ảnh có thể có đặc điểm không rõ ràng hoặc hoa tương tự nhiều loài.")
        else:
            st.markdown("""
            <div class='info-box'>
                <p style='color: #ff8fb3; text-align: center; margin: 100px 0;'>
                    📸<br>
                    <strong>Chưa có dự đoán</strong><br>
                    Tải ảnh lên và nhấn "Nhận Diện"
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis
    if 'predictions' in st.session_state:
        is_valid = st.session_state.get('is_valid', True)
        
        st.markdown("---")
        
        if is_valid:
            tab1, tab2 = st.tabs(["📊 Biểu Đồ", "📋 Chi Tiết"])
            
            with tab1:
                fig = plot_predictions(st.session_state['predictions'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("#### Kết Quả Tất Cả Các Loài")
                
                for idx, pred in enumerate(predictions, 1):
                    with st.container():
                        col_a, col_b, col_c = st.columns([1, 3, 1])
                        
                        with col_a:
                            emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣'][idx-1]
                            st.markdown(f"<p style='font-size: 2em; margin: 0;'>{emoji}</p>", unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"**{pred['class']}**")
                            st.progress(pred['confidence'] / 100)
                        
                        with col_c:
                            st.markdown(f"**{pred['confidence']:.1f}%**")
                        
                        if idx < len(predictions):
                            st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)
        else:
            # Nếu không đạt ngưỡng, chỉ hiển thị biểu đồ tham khảo
            st.markdown("#### 📊 Phân Tích Chi Tiết (Tham Khảo)")
            st.info("⚠️ Các kết quả dưới đây chỉ mang tính tham khảo vì độ tin cậy không đạt ngưỡng.")
            
            fig = plot_predictions(st.session_state['predictions'])
            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #ff8fb3; padding: 10px;'>
        <p>🌸 Được xây dựng với TensorFlow, Transformers và Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()