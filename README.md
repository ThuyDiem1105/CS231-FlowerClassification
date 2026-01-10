# Cấu trúc đồ án phân loại hoa 
```text
├── flowers/                           # Bộ dữ liệu hình ảnh hoa
│   ├── train/                         # Dữ liệu huấn luyện mô hình
│   ├── val/                           # Dữ liệu kiểm chứng (validation)
│   └── test/                          # Dữ liệu đánh giá cuối cùng (test)
├── HOG/                               # Đặc trưng hình dạng (Histogram of Oriented Gradients)
│   └── hog.ipynb                      #
├── HOG+HSV/                           # Kết hợp hình dạng và màu sắc
│   └── hog_hsv.ipynb
├── HOG+BOVW/                          # Kết hợp hình dạng và đặc trưng cục bộ
│   └── hog-bovw-flowers.ipynb
├── BOVW/                              # Đặc trưng cục bộ (Bag of Visual Words - SIFT)
│   ├── bovw-flowers-fixed-hyper-final.ipynb
│   ├── bovw-flowers-fixed-hyper.ipynb
│   └── bovw-flowers.ipynb
├── BOVW+HSV/                          # Kết hợp đặc trưng cục bộ và màu sắc (Phương pháp tối ưu)
│   ├── bovw_sift_hsv_svm.ipynb        #
│   └── bovw_sift_hsv_svm.pkl          #
├── HSV/                               # Đặc trưng màu sắc (Hue, Saturation, Value)
│   └── hsv_flowers.ipynb
├── VisionTransformer/                 # Mô hình Deep Learning (ViT) & Transfer Learning
│   ├── feature_extractor.weights.h5   # Trọng số bộ trích xuất đặc trưng
│   ├── feature_extractor.weights.py   # Script trích xuất đặc trưng
│   ├── vit_legacy.weights.h5          # Trọng số ViT phiên bản cũ
│   ├── vit_transfer_feature_extractor.keras
│   ├── vit_transfer_model.weights.h5
│   ├── vit-transfer_final.ipynb       # Notebook chuyển đổi mô hình cuối cùng
│   ├── vit-transfer.ipynb             # Notebook huấn luyện Transfer Learning
│   └── vit-transfer+SVM_final.ipynb   # Kết hợp ViT với bộ phân loại SVM
├── best_pr_svm_optimized_model.joblib # Mô hình SVM đã tối ưu hóa hiệu suất
├── demo_final.py                      # Ứng dụng giao diện người dùng (Demo)
├── explore_data.ipynb                 # Khám phá và phân tích tập dữ liệu
├── feature_scaler.pkl                 # Bộ chuẩn hóa đặc trưng (Scaler)
├── model_metadata.pkl                 # Metadata và cấu hình mô hình
├── summarize.json                     # Bản tóm tắt mô tả các loại hoa
└── svm_classifier.pkl                 # Mô hình phân loại SVM chính
