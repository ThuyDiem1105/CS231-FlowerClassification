import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pickle
from PIL import Image
import os
# Import ViTConfig và TFViTModel để tái tạo kiến trúc
from transformers import ViTConfig, TFViTModel 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# =================================================================
# 1. CẤU HÌNH THAM SỐ VÀ KHAI BÁO
# =================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
PRETRAINED_MODEL = "google/vit-base-patch16-224"

# Đường dẫn đến các file đã lưu (Kiểm tra lại đường dẫn này!)
FEATURE_EXTRACTOR_WEIGHTS_PATH = 'feature_extractor.weights.h5'
SVM_MODEL_PATH = 'svm_classifier.pkl'
SCALER_PATH = 'feature_scaler.pkl'
# CẬP NHẬT CLASS NAMES CỦA BẠN (7 LỚP)
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip', 'class_5', 'class_6'] 

# =================================================================
# 2. ĐỊNH NGHĨA LỚP BỌC (Tái tạo kiến trúc ViT)
# =================================================================
class ViTFeatureExtractorLayer(tf.keras.layers.Layer):
    """Gói TFViTModel. Khởi tạo model từ Config để tránh lỗi loading PyTorch weights."""
    def __init__(self, model_name=PRETRAINED_MODEL, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.vit_model = None 

    def build(self, input_shape):
        if self.vit_model is None:
            # 1. Tải cấu hình ViT
            config = ViTConfig.from_pretrained(self.model_name)
            
            # 2. Khởi tạo TFViTModel từ Config (tạo model từ scratch)
            self.vit_model = TFViTModel(config, name='vit_transfer')
            self.vit_model.config.output_hidden_states = True
            self.vit_model.config.output_attentions = False
            
        super().build(input_shape)

    def call(self, inputs):
        # inputs phải là (N, C, H, W)
        outputs = self.vit_model(pixel_values=inputs, training=False)
        return outputs.pooler_output

# =================================================================
# 3. HÀM XÂY DỰNG KIẾN TRÚC FEATURE EXTRACTOR
# =================================================================
def build_feature_extractor_architecture():
    """Xây dựng kiến trúc Feature Extractor đúng như trong code training."""
    
    inputs = layers.Input(shape=INPUT_SHAPE, name='pixel_values')
    
    x = layers.Normalization(
        mean=[0.5, 0.5, 0.5],
        variance=[0.25, 0.25, 0.25]
    )(inputs)

    x = layers.Permute((3, 1, 2))(x)
    
    vit_feature_layer = ViTFeatureExtractorLayer(model_name=PRETRAINED_MODEL)
    features_vit = vit_feature_layer(x) 
    
    # Các lớp Dense dùng để trích xuất features cuối cùng
    features = layers.Dense(256, activation="gelu", name="feature_dense_1")(features_vit)
    features = layers.Dropout(0.5, name="feature_dropout_1")(features)
    features = layers.Dense(128, activation="gelu", name="feature_dense_2")(features)
    
    feature_extractor = tf.keras.Model(
        inputs=inputs,
        outputs=features,
        name="ViT_Transfer_FeatureExtractor"
    )
    return feature_extractor

# =================================================================
# 4. HÀM TẢI MÔ HÌNH VÀ SCALER (SỬ DỤNG CACHING)
# =================================================================
@st.cache_resource
def load_feature_extractor():
    """Tải và cache Feature Extractor Model (TF Model lớn)."""
    st.write("Đang xây dựng kiến trúc ViT Feature Extractor...")
    feature_extractor = build_feature_extractor_architecture()
    
    st.write("Đang tải ViT Feature Extractor weights đã lưu...")
    try:
        if not os.path.exists(FEATURE_EXTRACTOR_WEIGHTS_PATH):
            raise FileNotFoundError(f"Không tìm thấy file weights: {FEATURE_EXTRACTOR_WEIGHTS_PATH}")
        feature_extractor.load_weights(FEATURE_EXTRACTOR_WEIGHTS_PATH)
        st.success("Tải ViT Feature Extractor weights thành công!")
        return feature_extractor
    except Exception as e:
        st.error(f"❌ Lỗi tải ViT weights: {e}")
        return None

@st.cache_data
def load_svm_and_scaler():
    """Tải và cache SVM và Scaler (các đối tượng pickle)."""
    
    # Tải SVM Classifier
    st.write("Đang tải SVM Classifier...")
    try:
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy file SVM: {SVM_MODEL_PATH}")
        with open(SVM_MODEL_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        st.success("Tải SVM model thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi tải SVM model: {e}")
        return None, None

    # Tải StandardScaler
    st.write("Đang tải StandardScaler...")
    try:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Không tìm thấy file Scaler: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        st.success("Tải StandardScaler thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi tải StandardScaler: {e}")
        return None, None
        
    return svm_model, scaler

# =================================================================
# 5. HÀM TIỀN XỬ LÝ VÀ DỰ ĐOÁN
# =================================================================
def preprocess_image(image):
    """Tiền xử lý ảnh cho ViT."""
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = tf.expand_dims(img_array, 0)
    return img_tensor

def predict_class(image_tensor, feature_extractor, scaler, svm_model, class_names):
    """Trích xuất features, chuẩn hóa và dự đoán bằng SVM."""
    # Trích xuất Features (chỉ chạy inference)
    features = feature_extractor.predict(image_tensor, verbose=0)
    
    # Chuẩn hóa Features
    features_scaled = scaler.transform(features)
    
    # Dự đoán bằng SVM
    pred_class_index = svm_model.predict(features_scaled)[0]
    
    return class_names[pred_class_index], pred_class_index

# =================================================================
# 6. GIAO DIỆN STREAMLIT
# =================================================================
st.set_page_config(
    page_title="Demo: ViT + SVM Phân Loại Hoa",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🌺 Demo Phân Loại Hoa: ViT Transfer Learning + SVM")
st.subheader("Mô hình đã huấn luyện: ViT Feature Extractor + SVM (Kernel RBF)")
st.markdown("---")

# Tải mô hình bằng các hàm cache đã tách biệt
feature_extractor = load_feature_extractor()
svm_model, scaler = load_svm_and_scaler()

if feature_extractor is None or svm_model is None or scaler is None:
    st.error("❌ Không thể tải đủ các thành phần mô hình. Vui lòng kiểm tra lại đường dẫn file và các thông báo lỗi tải ở trên.")
else:
    st.success("✅ Tải mô hình thành công. Bắt đầu Demo!")
    
    # Upload ảnh
    uploaded_file = st.file_uploader(
        "Tải lên một hình ảnh hoa:", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Ảnh tải lên', use_column_width=True)

        with col2:
            st.markdown("### 🔍 Kết quả Dự đoán")
            
            # Tiền xử lý
            with st.spinner('Đang tiền xử lý và trích xuất features...'):
                image_tensor = preprocess_image(image)
            
            # Dự đoán
            with st.spinner('Đang dự đoán bằng SVM...'):
                pred_class, pred_index = predict_class(
                    image_tensor, 
                    feature_extractor, 
                    scaler, 
                    svm_model, 
                    CLASS_NAMES
                )
            
            # Hiển thị kết quả
            st.metric(
                label="Lớp Hoa Dự Đoán:", 
                value=f"**{pred_class.upper()}**", 
                delta=None
            )
            st.success("🎉 Dự đoán hoàn tất!")
            
            st.markdown("---")
            st.markdown(f"**Thông tin chi tiết:**")
            st.markdown(f"* **Mô hình Trích xuất:** ViT-base-patch16-224 (Tái tạo kiến trúc)")
            st.markdown(f"* **Bộ phân loại:** Support Vector Machine (Kernel: RBF)")

st.markdown("---")
st.caption("Ứng dụng demo bởi Gemini. Vui lòng đảm bảo `tensorflow`, `transformers`, `scikit-learn` đã được cài đặt.")