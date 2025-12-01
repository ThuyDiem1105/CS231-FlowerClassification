import cv2
import numpy as np
import pickle
import json
import streamlit as st
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONSTANTS ---
JSON_FILE = "summarize.json" 
# Chỉ giữ lại MODEL_HOG_HSV
MODEL_HOG_HSV = 'HOG/best_svm_pca_hog_hsv_model.pkl' 

# --- TẢI CÁC THÀNH PHẦN ---
@st.cache_data(show_spinner=True)
def load_components(path):
    """Tải model, scaler, pca, và thông số cần thiết từ file .pkl."""
    st.write(f"Đang tải model từ: **{path}**...")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file model tại {path}")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        return None

@st.cache_data
def load_descriptions(path=JSON_FILE):
    """Tải mô tả các lớp từ file summarize.json."""
    try:
        current_dir = Path(__file__).parent
        file_path = current_dir / path
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô tả tại {path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Lỗi: File {path} không hợp lệ (JSON Error)")
        return {}

# --- HÀM TRÍCH XUẤT ĐẶC TRƯNG CHUNG ---
def extract_color_hist_hsv(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_L1) 
    return hist.flatten()

def extract_hog(img, orientations):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat, _ = hog(
        gray,
        orientations=orientations, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
        channel_axis=None,
        visualize=True
    )
    return hog_feat

# Đã loại bỏ extract_bovw_only_feature do chỉ dùng HOG+HSV

# --- HÀM TỔNG HỢP ĐẶC TRƯNG CHÍNH (Đơn giản hóa) ---
def get_final_feature_vector(img_bgr, model_data):
    """Thực hiện toàn bộ quá trình: Resize, Trích xuất HOG+HSV, Scale, PCA."""
    
    # Giả định luôn là HOG_HSV
    feature_type_code = 'HOG_HSV'
    st.info(f"-> Phát hiện: **Mô hình {feature_type_code}**. Đang trích xuất...")

    scaler = model_data['scaler']
    pca = model_data['pca']
    
    # KIỂM TRA AN TOÀN CHO THAM SỐ MODEL
    resize_shape = model_data.get('img_size', (128, 128)) 
    orientations = model_data.get('orientations', 9) 
    
    img_resized = cv2.resize(img_bgr, resize_shape)
    img_resized = np.ascontiguousarray(img_resized) 

    hog_feat = extract_hog(img_resized, orientations=orientations)
    color_feat = extract_color_hist_hsv(img_resized)
    
    # Nối HOG và HSV
    features = np.hstack([hog_feat, color_feat])
    features = features.reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    return features_pca, feature_type_code, img_resized

# --- HÀM XỬ LÝ DỰ ĐOÁN VÀ HIỂN THỊ KẾT QUẢ ---
def run_prediction(image_bgr, model_data, descriptions):
    """Hàm xử lý dự đoán và hiển thị kết quả cho Streamlit (chỉ HOG+HSV)."""
    
    # Chạy trích xuất đặc trưng (không cần model_choice)
    feature_vector_pca, feature_type_used, img_to_visualize = get_final_feature_vector(image_bgr, model_data)
    
    if feature_vector_pca is None:
        return
        
    svm_model = model_data['model']
    class_names = model_data['class_names']
    
    prediction_index = svm_model.predict(feature_vector_pca)[0]
    predicted_class = class_names[prediction_index]
    
    probabilities = svm_model.predict_proba(feature_vector_pca)[0]
    confidence = probabilities[prediction_index] * 100
    
    description_text = str(descriptions.get(predicted_class, "Không tìm thấy mô tả chi tiết cho loại hoa này."))
    
    # 6. HIỂN THỊ KẾT QUẢ
    st.markdown("### 🌼 KẾT QUẢ DỰ ĐOÁN")
    st.success(f"Dự đoán: **{predicted_class.upper()}** (Độ tin cậy: **{confidence:.2f}%**)")
    st.write(f"Phương pháp Đặc trưng: `{feature_type_used}`")
    st.write(f"Kích thước vector đặc trưng sau PCA: `{feature_vector_pca.shape[1]}`")

    # Bảng chi tiết xác suất
    with st.expander("Bảng Xác suất chi tiết"):
        sorted_indices = np.argsort(probabilities)[::-1]
        data = {
            "Loại Hoa": [class_names[i].capitalize() for i in sorted_indices],
            "Xác suất": [f"{probabilities[i]*100:.2f}%" for i in sorted_indices]
        }
        st.table(data)
        
    # Mô tả tóm tắt
    st.markdown("### Mô tả Tóm tắt")
    description_html = description_text.replace('\n', '<br>')
    st.markdown(f"**Loại hoa {predicted_class.capitalize()}**: \n > {description_html}", unsafe_allow_html=True)
        
    # 7. HIỂN THỊ TRỰC QUAN (Visualization) - CHỈ HOG+HSV
    st.markdown("### Trực quan hóa (HOG + HSV)")
    
    gray_image = cv2.cvtColor(img_to_visualize, cv2.COLOR_BGR2GRAY)
    model_orientations = model_data.get('orientations', 9)

    # 7.1. Trực quan hóa HOG
    _, hog_image = hog(
        gray_image, 
        orientations=model_orientations, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=True,
        feature_vector=False
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255)).astype(np.uint8)
    
    # 7.2. Hiển thị biểu đồ HSV 
    st.markdown("#### Biểu đồ Tần suất Màu HSV (8 bins)")
    img_hsv = cv2.cvtColor(img_to_visualize, cv2.COLOR_BGR2HSV)
    colors = ('Hue (0-180)', 'Saturation (0-255)', 'Value (0-255)')
    ranges = ([0, 180], [0, 256], [0, 256])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_hsv], [i], None, [8], ranges[i])
        hist = hist / hist.sum()
        axes[i].bar(range(8), hist.flatten(), color='gray', alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel('Bins')
        axes[i].set_xlim([0, 8])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 7.3. Hiển thị ảnh gốc và HOG
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=f'Ảnh gốc (Dự đoán: {predicted_class.capitalize()})', use_container_width=True)
        
    with col2:
        st.image(hog_image_rescaled, caption=f'Đặc trưng HOG (Ảnh sau Resize)', use_container_width=True)


# --- HÀM MAIN CHO STREAMLIT ---
def main():
    st.set_page_config(page_title="Hệ thống Nhận dạng Hoa", layout="wide")
    st.title("🌺 Hệ thống Nhận dạng Hoa (Flower Classifier)")
    st.markdown("Demo sử dụng phương pháp rút trích đặc trưng **HOG + HSV** và SVM để phân loại.")
    st.sidebar.header("Tùy chọn")
    
    # 1. TẢI ẢNH TỪ MÁY TÍNH
    uploaded_file = st.sidebar.file_uploader(
        "Chọn một ảnh hoa để dự đoán...",
        type=['jpg', 'jpeg', 'png']
    )
    
    # Tải model HOG+HSV duy nhất
    model_data = load_components(MODEL_HOG_HSV)
    descriptions = load_descriptions()
    
    if uploaded_file is not None and model_data is not None:
        
        # Đọc ảnh từ file upload (dạng Streamlit UploadedFile)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Chuyển đổi thành ảnh OpenCV (BGR)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.subheader(f"Ảnh đã Tải lên")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("---")
        
        # Chạy dự đoán
        run_prediction(img_bgr, model_data, descriptions) # Bỏ model_choice
        
    elif uploaded_file is None:
        st.warning("Vui lòng tải lên một hình ảnh để bắt đầu dự đoán.")

if __name__ == '__main__':
    main()