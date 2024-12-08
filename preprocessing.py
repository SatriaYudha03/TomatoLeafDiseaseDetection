import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Path input dataset dan output
input_dataset_path = "C:/Users/satri/Documents/Project-PCD/Dataset"
output_preprocessed_path = "C:/Users/satri/Documents/Project-PCD/Output preprocessed"

# Buat folder untuk menyimpan hasil preprocessing jika belum ada
if not os.path.exists(output_preprocessed_path):
    os.makedirs(output_preprocessed_path)

# Fungsi untuk preprocessing citra
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Thresholding untuk memisahkan objek dari latar belakang
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 3. Noise reduction menggunakan GaussianBlur
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)

    # 4. Edge detection (Canny)
    edges = cv2.Canny(blur, 100, 200)

    # 5. Resize image agar konsisten
    resized = cv2.resize(edges, (227, 227))

    # Gabungkan semua hasil preprocessing menjadi satu gambar
    combined = np.hstack((resized, gray, thresh, blur, edges))  # Horizontal stack

    return resized, gray, thresh, blur, edges, combined

# Fungsi untuk augmentasi citra
def augment_images(input_path, output_path):
    # Buat ImageDataGenerator untuk augmentasi citra
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Iterasi melalui setiap folder dalam dataset
    for class_name in os.listdir(input_path):
        class_folder = os.path.join(input_path, class_name)
        
        if os.path.isdir(class_folder):
            output_class_path = os.path.join(output_path, class_name)
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            # Loop untuk setiap gambar dalam folder kelas
            for image_name in tqdm(os.listdir(class_folder)):
                image_path = os.path.join(class_folder, image_name)
                
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Print status gambar yang sedang diproses
                    print(f"Memproses gambar '{image_name}' dari kelas '{class_name}'")
                    
                    # Preprocess image
                    resized_image, gray_image, thresh_image, blur_image, edges_image, combined_image = preprocess_image(image_path)

                    # Simpan gambar gabungan preprocessing
                    cv2.imwrite(os.path.join(output_class_path, f"combined_{image_name}"), combined_image)

                    # Simpan gambar preprocessing masing-masing jika dibutuhkan
                    cv2.imwrite(os.path.join(output_class_path, f"resized_{image_name}"), resized_image)
                    cv2.imwrite(os.path.join(output_class_path, f"gray_{image_name}"), gray_image)
                    cv2.imwrite(os.path.join(output_class_path, f"thresholded_{image_name}"), thresh_image)
                    cv2.imwrite(os.path.join(output_class_path, f"blurred_{image_name}"), blur_image)
                    cv2.imwrite(os.path.join(output_class_path, f"edges_{image_name}"), edges_image)

# Menjalankan preprocessing dan augmentasi
augment_images(input_dataset_path, output_preprocessed_path)

# Fungsi untuk menampilkan hasil perbandingan sebelum dan sesudah preprocessing
def show_comparison(image_path, processed_image):
    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display

    # Display original image and processed image side by side
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    # Processed image
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')

    plt.show()

# Contoh: Menampilkan hasil sebelum dan sesudah preprocessing untuk satu gambar dari tiap kelas
for class_name in os.listdir(input_dataset_path):
    class_folder = os.path.join(input_dataset_path, class_name)
    if os.path.isdir(class_folder):
        first_image_path = os.path.join(class_folder, os.listdir(class_folder)[0])
        preprocessed_image = preprocess_image(first_image_path)[5]  # Ambil hasil gabungan
        show_comparison(first_image_path, preprocessed_image)
