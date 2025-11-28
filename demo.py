import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import pandas as pd
import os 
import warnings
import joblib 
import cv2  
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.svm import SVC 

warnings.filterwarnings('ignore') 

# =================================================================
# 1. CẤU HÌNH THAM SỐ (PHẢI KHỚP VỚI LÚC HUẤN LUYỆN)
# =================================================================
# Đường dẫn cho cả hai mô hình
MODEL_PATHS = {
    "ViT (Vision Transformer)": 'vit_flowers_model.weights.h5',
    "BoVW + SIFT + HSV (SVM)": 'bovw_sift_hsv_svm.pkl' 
}

# Tham số Kích thước ảnh đầu vào (Phải khớp với ViT và BoVW)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_CLASSES = 7
CLASS_NAMES = ['daisy', 'dandelion', 'lily', 'orchid', 'rose', 'sunflower', 'tulip']

# Tham số ViT
PATCH_SIZE = 16
NUM_PATCHES = 196 
PROJECTION_DIM = 128
NUM_HEADS = 4
TRANSFORMER_LAYERS = 6
MLP_UNITS = [256, 128]
MLP_HEAD_UNITS = [128]

# Tham số BoVW
K_CLUSTERS = 183 # <--- ĐÃ SỬA: 192 (SVM features) - 9 (HSV features) = 183
# -----------------------------------------------------------------


# --- KIỂM TRA ĐƯỜNG DẪN FILE MÔ HÌNH ---
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        st.error(f"LỖI KHỞI TẠO: Không tìm thấy file mô hình `{name}` tại đường dẫn: `{path}`")
        st.stop()


# =================================================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG BOVW (Cho mô hình SVM)
# =================================================================

def extract_rootsift_descriptors(img_gray, max_kp=500):
    sift = cv2.SIFT_create()
    keypoints, desc = sift.detectAndCompute(img_gray, None)
    if desc is None:
        return None
    if desc.shape[0] > max_kp:
        desc = desc[:max_kp]
    desc = desc.astype("float32")
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc)
    return desc

def extract_hsv_hist(img_bgr, bins=(4,4,4)):
    img_resized = cv2.resize(img_bgr, (256, 256))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0,1,2], None,
        bins,                       # (H,S,V)
        [0,180, 0,256, 0,256]
    )
    hist = hist.astype("float32")
    hist = hist.flatten()
    hist /= (hist.sum() + 1e-7)
    hist = np.sqrt(hist)
    return hist

def extract_center_hsv_hist(img_bgr, bins=(4,4,4)):
    h, w = img_bgr.shape[:2]
    # cắt ô giữa ảnh (ví dụ 1/2 chiều cao, 1/2 chiều rộng)
    x1, x2 = w//4, 3*w//4
    y1, y2 = h//4, 3*h//4
    center = img_bgr[y1:y2, x1:x2]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = hist.astype("float32").flatten()
    hist /= (hist.sum() + 1e-7)
    hist = np.sqrt(hist)   # Hellinger
    return hist

def process_image_for_prediction(img_bgr, model_obj):
    """
    Trả về feature vector = [BoVW_RootSIFT, HSV_hist]
    """
    kmeans, scaler, pca, svm_model, class_names, K = model_obj
    
    img_resized = cv2.resize(img_bgr, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # --- BoVW từ RootSIFT ---
    desc = extract_rootsift_descriptors(gray)
    if desc is None:
        bovw_hist = np.zeros(K, dtype=np.float32)
    else:
        words = kmeans.predict(desc)
        bovw_hist, _ = np.histogram(words, bins=np.arange(K+1))
        bovw_hist = bovw_hist.astype("float32")
        # Hellinger
        bovw_hist /= (bovw_hist.sum() + 1e-7)
        bovw_hist = np.sqrt(bovw_hist)

    # --- HSV color feature ---
    global_hsv = extract_hsv_hist(img_resized, bins=(4,4,4))  # 64 dims
    center_hsv = extract_center_hsv_hist(img_resized, bins=(4,4,4))
    
    # Gộp
    feat = np.hstack([bovw_hist, global_hsv, center_hsv])
    
    # Reshape (1, N) để đưa vào scaler
    feat = feat.reshape(1, -1)
    
    # Scale & PCA
    feat_scaled = scaler.transform(feat)
    feat_pca = pca.transform(feat_scaled)

    return feat_pca
# =================================================================
# 3. KIẾN TRÚC VIT VÀ HÀM TẢI MÔ HÌNH
# =================================================================

# --- Create Patches ---
class PatchLayer(layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dim = PATCH_SIZE * PATCH_SIZE * 3
        return tf.reshape(patches, [-1, NUM_PATCHES, patch_dim])

# --- Patch Encoder (Linear Projection + Positional Encoding) ---
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, 
            output_dim=projection_dim
        )
        
    def call(self, patch_tokens):
        positions = tf.range(start=0, limit=self.num_patches)
        encoded = self.projection(patch_tokens) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


# --- Transformer Encoder Block ---
def transformer_encoder(inputs):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, 
        key_dim=PROJECTION_DIM, 
        dropout=0.1
    )(x, x)
    attn = layers.Dropout(0.1)(attn)
    x = layers.Add()([attn, inputs])

    # FFN
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(MLP_UNITS[0], activation='gelu')(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(PROJECTION_DIM)(y)
    return layers.Add()([x, y])

# --- Build ViT model ---
def build_vit(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # 1) Make patches
    patches = PatchLayer()(inputs) 

    # 2) Patch encoding 
    patch_embeddings = PatchEncoder()(patches)

    # 3) class token variable (trainable)
    class_token = tf.Variable(
        tf.zeros((1, 1, PROJECTION_DIM)), 
        trainable=True, 
        name="class_token"
    )

    # 4) use a Lambda layer to repeat & concat class token
    def _prepend_token(patch_emb):
        batch = tf.shape(patch_emb)[0]
        tokens = tf.repeat(class_token, repeats=batch, axis=0)
        return tf.concat([tokens, patch_emb], axis=1)

    x = layers.Lambda(_prepend_token, name="prepend_class_token")(patch_embeddings)

    # 5) Transformer encoder stacks
    for i in range(TRANSFORMER_LAYERS):
        x = transformer_encoder(x)

    # 6) Take class token output (index 0)
    x = layers.LayerNormalization(epsilon=1e-6, name="pre_head_ln")(x[:, 0])

    # 7) MLP head
    for units in MLP_HEAD_UNITS:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ViT_Flowers")

# --- TẢI MÔ HÌNH CHUNG ---
@st.cache_resource
def load_model(model_name):
    path = MODEL_PATHS[model_name]
    
    if model_name.startswith("ViT"):
        try:
            input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3) 
            num_classes = len(CLASS_NAMES)
            
            # Xây dựng lại kiến trúc mô hình ViT
            model = build_vit(input_shape, num_classes)
            model.load_weights(path)
            return model
        except Exception as e:
            st.error(f"Lỗi Tải Trọng Số ViT: {e}. Kiến trúc không khớp.")
            print(f"[LỖI TẢI VIT] Chi tiết: {e}")
            return None
            
    elif model_name.startswith("BoVW"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            kmeans = data["kmeans"]
            scaler = data["scaler"]
            pca = data["pca"]
            svm_model = data["model"]
            class_names = data["class_names"]
            K = data["K_value"]

            return (kmeans, scaler, pca, svm_model, class_names, K)
                 
        except Exception as e:
            st.error(f"Lỗi Tải Mô Hình BoVW: {e}. Vui lòng kiểm tra lại cấu trúc file .pkl.")
            print(f"[LỖI TẢI BOVW] Chi tiết: {e}")
            return None
    return None

# --- HÀM DỰ ĐOÁN CHUNG ---
def predict_image(model_name, model_obj, image, size, class_names):
    
    results = []
    
    if model_name.startswith("ViT"):
        # ********* LOGIC DỰ ĐOÁN VIT *********
        img_resized = image.resize((size, size)).convert('RGB')

        img_array = keras.preprocessing.image.img_to_array(img_resized) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0 
        
        predictions = model_obj.predict(img_array)
        
        # Lấy xác suất
        results = [{'class': name, 'probability': prob} for name, prob in zip(class_names, predictions[0])]
        
    elif model_name.startswith("BoVW"):
        # ********* LOGIC DỰ ĐOÁN BOVW *********
        # kmeans, scaler, pca, svm_model, class_names, K = model_obj
            
        # Chuyển đổi PIL sang cv2 (numpy BGR)
        img_array = np.array(image.convert('RGB')) # Đảm bảo là RGB trước khi sang numpy
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # --- TRÍCH XUẤT ĐẶC TRƯNG BOVW + HSV ---
        features = process_image_for_prediction(img_bgr, model_obj)
            
        # --- THỰC HIỆN DỰ ĐOÁN ---
        probabilities = model_obj[3].predict_proba(features)[0]

        results = [{'class': name, 'probability': prob} for name, prob in zip(class_names, probabilities)]   
    return results


# =================================================================
# 4. GIAO DIỆN STREAMLIT
# =================================================================

st.title("🌺 Demo Phân Loại Hoa Đa Mô Hình")
st.markdown("Chọn một mô hình (ViT hoặc BoVW) và tải lên ảnh để kiểm tra kết quả phân loại.")

# 4a. Thanh chọn mô hình
selected_model_name = st.selectbox(
    "Chọn Mô Hình Phân Loại:",
    list(MODEL_PATHS.keys())
)

# Tải mô hình đã chọn
model_obj = load_model(selected_model_name)

if model_obj is not None:
    st.success(f"✅ Mô hình **{selected_model_name}** đã được tải thành công.")

    uploaded_file = st.file_uploader(
        "Chọn một file ảnh...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Đọc ảnh từ file đã upload
        image = Image.open(uploaded_file)
        
        # Hiển thị ảnh
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
        st.write("")
        
        # Nút Phân loại
        if st.button('Phân loại ngay!'):
            with st.spinner(f'Đang chạy dự đoán bằng {selected_model_name}...'):
                
                # Thực hiện dự đoán
                results = predict_image(selected_model_name, model_obj, image, IMAGE_SIZE, CLASS_NAMES)
                
                # Sắp xếp kết quả theo xác suất giảm dần
                results.sort(key=lambda x: x['probability'], reverse=True)
                
                best_pred = results[0]

                # Hiển thị kết quả chính
                st.success(f"✅ DỰ ĐOÁN HOÀN TẤT!")
                st.markdown(f"**Loại Hoa Dự Đoán là:** <span style='font-size: 24px; color: #ff4b4b;'>{best_pred['class'].capitalize()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Độ tự tin:** `{best_pred['probability']:.2%}`")
                
                st.write("---")

                # Hiển thị bảng xác suất chi tiết
                st.subheader("Bảng Xác Suất Chi Tiết")
                
                # Định dạng dữ liệu cho DataFrame
                df_results = pd.DataFrame([
                    {'Loại Hoa': r['class'].capitalize(), 'Xác Suất': f"{r['probability']:.2%}"} 
                    for r in results
                ])
                st.dataframe(df_results, use_container_width=True, hide_index=True)

else:
    # Nếu tải mô hình lỗi, thông báo lỗi cụ thể đã được hiển thị bên trong load_model
    st.error("⚠️ Ứng dụng không thể khởi động do lỗi tải mô hình. Vui lòng kiểm tra các thông báo lỗi cụ thể.")