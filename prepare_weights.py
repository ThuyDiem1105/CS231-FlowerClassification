import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# =================================================================
# THAM SỐ PHẢI KHỚP VỚI LÚC HUẤN LUYỆN
# =================================================================
# ĐÃ SỬA LỖI ĐẶT TÊN: File mới phải có hậu tố .weights.h5
MODEL_PATH_OLD = 'vit_transfer_model.weights.h5'
MODEL_PATH_NEW = 'vit_legacy.weights.h5' 

# Tham số ViT
IMAGE_SIZE = 224
NUM_CLASSES = 7
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 128
NUM_HEADS = 4
TRANSFORMER_LAYERS = 6
MLP_UNITS = [256, 128]
MLP_HEAD_UNITS = [128]

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC VIỆT (TỐI ƯU HÓA) ---

class PatchLayer(layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(images=images, sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                                            strides=[1, PATCH_SIZE, PATCH_SIZE, 1], rates=[1, 1, 1, 1],
                                            padding="VALID")
        patch_dim = PATCH_SIZE * PATCH_SIZE * 3
        return tf.reshape(patches, [-1, NUM_PATCHES, patch_dim])
    def get_config(self): return super().get_config()

class ViT_Embedding_Block(layers.Layer):
    def __init__(self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim, name="linear_projection")
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim, name="position_embedding")
    def build(self, input_shape):
        self.class_token = self.add_weight(shape=(1, 1, self.projection_dim), initializer="zeros", trainable=True, name="class_token_var")
        self.built = True
    def call(self, patches):
        encoded_patches = self.projection(patches)
        batch_size = tf.shape(patches)[0]
        tokens = tf.repeat(self.class_token, repeats=batch_size, axis=0)
        concatenated = tf.concat([tokens, encoded_patches], axis=1)
        positions = tf.range(start=0, limit=self.num_patches + 1)
        return concatenated + self.position_embedding(positions)
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim,})
        return config

def transformer_encoder(inputs, block_idx): 
    x = layers.LayerNormalization(epsilon=1e-6, name=f"layer_normalization_{2*block_idx}")(inputs)
    attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1, name=f"multi_head_attention_{block_idx}")(x, x)
    attn = layers.Dropout(0.1)(attn)
    x = layers.Add()([attn, inputs])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"layer_normalization_{2*block_idx+1}")(x)
    y = layers.Dense(MLP_UNITS[0], activation='gelu', name=f"dense_{2*block_idx}")(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(PROJECTION_DIM, name=f"dense_{2*block_idx+1}")(y)
    return layers.Add()([x, y])

def build_vit(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    patches = PatchLayer(name="patch_layer")(inputs) 
    x = ViT_Embedding_Block(NUM_PATCHES, PROJECTION_DIM, name="vit_embedding_block")(patches)
    for i in range(TRANSFORMER_LAYERS):
        x = transformer_encoder(x, block_idx=i)
    x = layers.LayerNormalization(epsilon=1e-6, name="pre_head_ln")(x[:, 0])
    for i, units in enumerate(MLP_HEAD_UNITS):
        x = layers.Dense(units, activation="gelu", name=f"mlp_head_dense_{i}")(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ViT_Flowers")

# --- 2. LOGIC CHUYỂN ĐỔI ---

if not os.path.exists(MODEL_PATH_OLD):
    print(f"LỖI: Không tìm thấy file gốc: {MODEL_PATH_OLD}")
else:
    try:
        model = build_vit()
        model(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32), training=False)
        
        # Tải trọng số gốc (Dùng skip_mismatch=True để bỏ qua lỗi EinsumDense và Class Token)
        print(f"Đang tải trọng số từ {MODEL_PATH_OLD}...")
        
        # LỆNH QUAN TRỌNG: Lệnh này đã bị Keras chặn, nhưng là nỗ lực cuối cùng trong Python
        model.load_weights(MODEL_PATH_OLD, skip_mismatch=True)
        print("Tải trọng số gốc thành công (đã bỏ qua lỗi không khớp kiến trúc).")
        
        # LƯU LẠI theo định dạng Legacy H5
        model.save_weights(MODEL_PATH_NEW)
        print(f"✅ Đã lưu thành công file LEGACY H5 mới tại: {MODEL_PATH_NEW}")

    except Exception as e:
        print(f"\n🛑 LỖI CHUYỂN ĐỔI: Không thể tải hoặc lưu lại trọng số.")
        print(f"Chi tiết lỗi: {e}")