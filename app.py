import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import io
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- CẤU HÌNH & HẰNG SỐ ---
# Các tham số trích xuất đặc trưng CẦN PHẢI GIỐNG HỆT như trong notebook
K_VALUE = 700
HSV_BINS = (4, 4, 4) # 4*4*4 = 64 dimensions cho mỗi histogram
IMG_SIZE = (256, 256)
sift = cv2.SIFT_create()

# --- 1. Hàm Tải Model (Đã sửa) ---

@st.cache_resource
def load_all_components(path='bovw_sift_hsv_svm.pkl'):
    """Tải tất cả các thành phần (kmeans, scaler, pca, model, names) từ file dictionary."""
    try:
        if not os.path.exists(path):
            st.error(f"Lỗi: Không tìm thấy file model tại {path}. Hãy chắc chắn bạn đã chạy bước lưu file trong notebook và đặt file đúng chỗ.")
            return None, None, None, None, None
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
            # Trích xuất các thành phần từ dictionary
            kmeans = model_data.get('kmeans')
            scaler = model_data.get('scaler')
            pca = model_data.get('pca')
            svm_model = model_data.get('model')
            class_names = model_data.get('class_names')
            
            # Kiểm tra tính toàn vẹn
            if None in [kmeans, scaler, pca, svm_model, class_names]:
                st.error("Lỗi: File model không chứa đủ các thành phần (kmeans, scaler, pca, model, class_names).")
                return None, None, None, None, None

            st.success("✅ Model (SVM), Visual Dictionary (KMeans), Scaler, và PCA đã được tải thành công!")
            return kmeans, scaler, pca, svm_model, class_names

    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        return None, None, None, None, None

# --- 2. Hàm Trích xuất Đặc trưng (RootSIFT, HSV) ---

def extract_rootsift_descriptors(img_gray, max_kp=500):
    """Tái tạo chính xác hàm RootSIFT từ notebook."""
    keypoints, desc = sift.detectAndCompute(img_gray, None)
    if desc is None:
        return None

    if desc.shape[0] > max_kp:
        desc = desc[:max_kp]
        
    # RootSIFT: L1 normalize + căn bậc hai
    desc = desc.astype("float32")
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc)

    return desc

def extract_hsv_hist(img_bgr, bins=HSV_BINS):
    """Trích xuất 3D HSV histogram (Hellinger) từ toàn bộ ảnh resize."""
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0,1,2], None,
        bins,                       
        [0,180, 0,256, 0,256]
    )
    hist = hist.astype("float32").flatten()

    # Hellinger
    hist /= (hist.sum() + 1e-7)
    hist = np.sqrt(hist)
    return hist

def extract_center_hsv_hist(img_bgr, bins=HSV_BINS):
    """Trích xuất 3D HSV histogram (Hellinger) từ ô giữa ảnh."""
    h, w = img_bgr.shape[:2]
    # cắt ô giữa ảnh (1/2 kích thước)
    x1, x2 = w//4, 3*w//4
    y1, y2 = h//4, 3*h//4
    center = img_bgr[y1:y2, x1:x2]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = hist.astype("float32").flatten()
    
    # Hellinger
    hist /= (hist.sum() + 1e-7)
    hist = np.sqrt(hist)   
    return hist

def image_to_feature_vector(img_bgr, kmeans: MiniBatchKMeans, scaler: StandardScaler, pca: PCA):
    """
    Tái tạo toàn bộ quy trình trích xuất và biến đổi feature.
    Trả về feature vector cuối cùng (sau PCA).
    """
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # --- 1. BoVW từ RootSIFT ---
    desc = extract_rootsift_descriptors(gray)
    if desc is None:
        bovw_hist = np.zeros(K_VALUE, dtype=np.float32)
    else:
        words = kmeans.predict(desc)
        bovw_hist, _ = np.histogram(words, bins=np.arange(K_VALUE+1))
        bovw_hist = bovw_hist.astype("float32")
        # Hellinger
        bovw_hist /= (bovw_hist.sum() + 1e-7)
        bovw_hist = np.sqrt(bovw_hist)

    # --- 2. HSV color feature (Global & Center) ---
    global_hsv = extract_hsv_hist(img_resized, bins=HSV_BINS) 
    center_hsv = extract_center_hsv_hist(img_resized, bins=HSV_BINS)
    
    # --- 3. Gộp feature ---
    feat = np.hstack([bovw_hist, global_hsv, center_hsv]) # K + 64 + 64 = 828 dims
    feat = feat.reshape(1, -1) # Đảm bảo là mảng 2D

    # --- 4. Chuẩn hóa (StandardScaler) ---
    feats_scaled = scaler.transform(feat)

    # --- 5. Giảm chiều (PCA) ---
    feats_pca = pca.transform(feats_scaled)
    
    return feats_pca


# --- 3. Ứng dụng Streamlit ---

def main():
    st.set_page_config(page_title="🌸 Hệ thống Phân loại Hoa Demo", layout="centered")
    
    st.title("🌺 Hệ thống Phân loại Hoa Dựa trên Hình ảnh")
    st.markdown("Sử dụng model **BoVW-RootSIFT + HSV + SVM** đã được huấn luyện.")
    
    # Tải Model và các thành phần
    kmeans, scaler, pca, svm_model, class_names = load_all_components()
    
    if svm_model is None:
        st.stop() # Dừng ứng dụng nếu không tải được model hoặc thiếu thành phần

    uploaded_file = st.file_uploader(
        "Tải lên hình ảnh hoa (Định dạng: .jpg, .jpeg, .png)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # 1. Hiển thị hình ảnh (Dùng PIL)
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption='Hình ảnh được tải lên.', use_container_width=True) # Đã sửa cảnh báo
            st.write("---")
            
            # Chuyển đổi PIL Image sang mảng NumPy (OpenCV format - BGR)
            # Dùng PIL để tránh lỗi cv2.imdecode như bạn gặp trước đó
            img_np_rgb = np.array(image_pil)
            img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
            
            if img_bgr is None or img_bgr.size == 0:
                st.error("Không thể đọc file hình ảnh. Vui lòng thử một file khác.")
                return

            # 2. Trích xuất Đặc trưng
            with st.spinner('Đang trích xuất đặc trưng (RootSIFT-BoVW, HSV, Scaling, PCA)...'):
                feature_vector = image_to_feature_vector(img_bgr, kmeans, scaler, pca)

            if feature_vector is not None:
                st.success(f"Đã trích xuất đặc trưng thành công. Kích thước vector cuối cùng: {feature_vector.shape[1]}")
                
                # 3. Dự đoán
                with st.spinner('Đang dự đoán loại hoa...'):
                    # Model được huấn luyện với probability=True
                    probabilities = svm_model.predict_proba(feature_vector)[0]
                    prediction = svm_model.predict(feature_vector)[0]
                    predicted_class_name = class_names[prediction]
                    
                    st.balloons()
                    st.header(f"✨ Kết quả Phân loại: **{predicted_class_name.upper()}**")
                    
                    # Hiển thị độ tin cậy
                    confidence = probabilities[prediction] * 100
                    st.subheader(f"Độ tin cậy: **{confidence:.2f}%**")
                    
                    # Bảng xếp hạng các lớp
                    st.write("### Độ tin cậy chi tiết:")
                    
                    # Sắp xếp theo xác suất giảm dần
                    sorted_indices = np.argsort(probabilities)[::-1]
                    
                    data = []
                    for i in sorted_indices:
                        data.append({
                            'Loại Hoa': class_names[i].capitalize(),
                            'Xác suất': f'{probabilities[i]*100:.2f}%'
                        })
                    
                    st.table(data)
                        
            else:
                st.error("Không thể trích xuất đặc trưng.")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == '__main__':
    main()