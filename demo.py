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
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.svm import SVC 

warnings.filterwarnings('ignore') 

# =================================================================
# 1. CẤU HÌNH THAM SỐ (PHẢI KHỚP VỚI LÚC HUẤN LUYỆN)
# =================================================================
# Đường dẫn cho cả hai mô hình
MODEL_PATHS = {
    "ViT (Vision Transformer)": 'vit_flowers_model.weights.h5',
    "BoVW + SIFT + HSV (SVM)": 'bovw_sift_hsv_model.pkl' 
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

def extract_bovw_features(image_cv, kmeans_model, k_clusters):
    """
    Thực hiện trích xuất SIFT và Color Histogram (HSV), sau đó tạo vector BoVW.
    image_cv: Ảnh đã resize (dùng cv2.resize) ở định dạng BGR.
    """
    
    # Chuyển đổi ảnh sang ảnh xám cho SIFT
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 1. SIFT Extraction
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    # 2. BoVW Histogram
    if descriptors is None or len(descriptors) == 0:
        # Kích thước phải là K_CLUSTERS + 9 (cho HSV)
        return np.zeros((1, k_clusters + 9), dtype=np.float32) 
    
    # Quantize SIFT descriptors
    try:
        # Sử dụng KMeans để quantize (gán cụm) descriptors
        clusters = kmeans_model.predict(descriptors)
    except AttributeError:
        # Lỗi này chỉ xảy ra nếu KMeans không phải là mô hình Scikit-learn hợp lệ
        # Nếu gặp lỗi này, hãy kiểm tra lại Kmeans Model được lưu
        st.error("Lỗi: Mô hình KMeans không có phương thức predict. Không thể trích xuất SIFT.")
        return np.zeros((1, k_clusters + 9), dtype=np.float32) 
        
    bovw_hist, _ = np.histogram(clusters, bins=range(k_clusters + 1), density=True)
    
    # 3. HSV Color Histogram (9 features: 3 bins per H, S, V)
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    # LƯU Ý: Nếu mô hình SVM của bạn chỉ dùng 3 bins cho H, S, V TỔNG CỘNG (tức 1 feature H, 1 feature S, 1 feature V)
    # thì K_CLUSTERS = 189. Hiện tại, chúng ta dùng 9 features (3 bins cho mỗi kênh)
    h_hist = cv2.calcHist([hsv_image], [0], None, [3], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], None, [3], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], None, [3], [0, 256]).flatten()
    
    color_hist = np.concatenate([h_hist, s_hist, v_hist])
    color_hist /= (color_hist.sum() + 1e-7) # Chuẩn hóa màu sắc
    
    # 4. Concatenate
    final_features = np.concatenate([bovw_hist, color_hist])
    return final_features.reshape(1, -1)


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
            bovw_obj = joblib.load(path)
            
            # Khởi tạo giá trị mặc định
            kmeans_model = None
            svm_classifier = None
            
            # --- XỬ LÝ LỖI BOVW (Phát hiện đối tượng dict) ---
            if isinstance(bovw_obj, dict):
                st.info("Phát hiện: File PKL chứa đối tượng `dict`. Đang tìm kiếm KMeans và SVM theo khóa...")
                
                # Cơ chế tìm kiếm linh hoạt trong dict
                found_kmeans = False
                found_svm = False
                
                for key, obj in bovw_obj.items():
                    # Tìm KMeans (có predict và tên chứa 'kmeans')
                    if hasattr(obj, 'predict') and obj.__class__.__name__.lower().find('kmeans') != -1:
                        kmeans_model = obj
                        found_kmeans = True
                    # Tìm SVM/Pipeline (có predict và tên chứa 'svc' hoặc 'pipeline')
                    elif hasattr(obj, 'predict') and (obj.__class__.__name__.lower().find('svc') != -1 or obj.__class__.__name__.lower().find('pipeline') != -1):
                        svm_classifier = obj
                        found_svm = True

                if found_kmeans and found_svm:
                    return (kmeans_model, svm_classifier)
                elif found_kmeans and not found_svm:
                    st.warning("CẢNH BÁO BOVW: Phát hiện KMeans, nhưng SVM Classifier bị thiếu trong dict.")
                    return (kmeans_model, None) # Trả về tuple (KMeans, None)
                else:
                    st.error(f"Lỗi BoVW: Không tìm thấy KMeans và/hoặc SVM Classifier trong dict. Các khóa trong dict: {list(bovw_obj.keys())}")
                    return None
            
            # Trường hợp 2: Tuple (KMeans, SVM)
            elif isinstance(bovw_obj, tuple) and len(bovw_obj) == 2 and hasattr(bovw_obj[1], 'predict'):
                st.info("Phát hiện: Tuple (KMeans, SVM). Sử dụng cả hai.")
                return bovw_obj
            
            # Trường hợp 3: Chỉ có KMeans (Dựa trên lỗi trước)
            elif hasattr(bovw_obj, 'predict') and bovw_obj.__class__.__name__.lower().find('kmeans') != -1:
                st.warning("CẢNH BÁO BOVW: Phát hiện chỉ có Mô hình KMeans (Từ vựng). Không có SVM Classifier.")
                return (bovw_obj, None)
            
            # Các trường hợp lỗi khác
            else:
                class_name_actual = bovw_obj.__class__.__name__
                st.error(f"Cấu trúc file .pkl BoVW không xác định. Đối tượng là loại `{class_name_actual}`. Vui lòng kiểm tra lại quá trình lưu mô hình.")
                return None
                 
        except Exception as e:
            st.error(f"Lỗi Tải Mô Hình BoVW: {e}. Vui lòng kiểm tra lại cấu trúc file .pkl.")
            print(f"[LỖI TẢI BOVW] Chi tiết: {e}")
            return None
    return None

# --- HÀM DỰ ĐOÁN CHUNG ---
def predict_image(model_name, model_obj, image, size, class_names):
    
    # 1. Tiền xử lý ảnh PIL (Resize & RGB)
    img_resized = image.resize((size, size)).convert('RGB')
    
    if model_name.startswith("ViT"):
        # ********* LOGIC DỰ ĐOÁN VIT *********
        
        img_array = keras.preprocessing.image.img_to_array(img_resized) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0 
        
        predictions = model_obj.predict(img_array)
        
        # Lấy xác suất
        results = [{'class': name, 'probability': prob} for name, prob in zip(class_names, predictions[0])]
        
    elif model_name.startswith("BoVW"):
        # ********* LOGIC DỰ ĐOÁN BOVW *********
        
        # model_obj có thể là (KMeans, SVM) hoặc (KMeans, None)
        kmeans_model, svm_or_pipeline = model_obj
        
        # --- KIỂM TRA MÔ HÌNH THIẾU ---
        if svm_or_pipeline is None:
            # Thông báo lỗi đã được in ở giao diện bởi load_model, chỉ trả về 0
            return [{'class': c, 'probability': 0.0} for c in class_names]
            
        # Chuyển đổi PIL sang cv2 (numpy BGR)
        img_cv_rgb = np.array(img_resized)
        img_cv_bgr = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
        
        # --- QUY TRÌNH TRÍCH XUẤT FEATURES (Chỉ khi có KMeans) ---
        if kmeans_model is not None:
            # Trích xuất SIFT + BoVW thủ công
            features = extract_bovw_features(img_cv_bgr, kmeans_model, K_CLUSTERS)
        else:
            # Trường hợp lỗi thiếu KMeans (đã được xử lý trong load_model)
            st.error("LỖI BOVW: Không có KMeans/Từ vựng để trích xuất SIFT/BoVW features.")
            return [{'class': c, 'probability': 0.0} for c in class_names]
            
        # --- THỰC HIỆN DỰ ĐOÁN ---
        if features is not None and features.shape[1] == (K_CLUSTERS + 9):
            prediction_index = svm_or_pipeline.predict(features)[0]
            
            if hasattr(svm_or_pipeline, 'predict_proba'):
                probabilities = svm_or_pipeline.predict_proba(features)[0]
            else:
                probabilities = np.zeros(len(class_names))
                probabilities[prediction_index] = 1.0

            results = [{'class': name, 'probability': prob} for name, prob in zip(class_names, probabilities)]
        else:
             st.error(f"Lỗi Dự đoán BoVW: Kích thước vector features không khớp ({features.shape[1]} != {K_CLUSTERS + 9}).")
             return [{'class': c, 'probability': 0.0} for c in class_names]
        
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