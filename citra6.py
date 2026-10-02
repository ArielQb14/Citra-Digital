import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def uji_sistem_ttd(image_path):
    """
    Performs signature detection pipeline on an image, including cropping, thresholding,
    morphological operations, and characteristic calculation, followed by visualization.
    """
    try:
        # 1. Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return

        original_height, original_width = img.shape[:2]

        # 2. Crop Area Tanda Tangan (Dean's signature area - top right)
        # Landscape format: y: 10% to 40%, x: 65% to 90%
        y_start = int(original_height * 0.10)
        y_end = int(original_height * 0.40)
        x_start = int(original_width * 0.65)
        x_end = int(original_width * 0.90)

        cropped_img = img[y_start:y_end, x_start:x_end]
        
        # Ensure cropped_img is not empty after cropping
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            print(f"Warning: Cropped image for {image_path} is empty. Skipping.")
            return

        # 3. Grayscale Conversion
        gray_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # 4. Dua Metode Thresholding
        # Otsu Thresholding
        _, otsu_thresh = cv2.threshold(gray_cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Adaptive Thresholding (Gaussian C)
        # Block size must be odd and > 1. C is a constant subtracted from the mean.
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 5. Morphological Operation (on Otsu result)
        kernel = np.ones((3, 3), np.uint8)
        opened_img = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)  # Remove small white noise
        morph_cleaned_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel) # Connect broken lines

        # 6. Hitung Karakteristik Area
        white_pixels = cv2.countNonZero(morph_cleaned_img)
        total_pixels = morph_cleaned_img.shape[0] * morph_cleaned_img.shape[1]
        
        # Avoid division by zero if cropped area is somehow 0 pixels
        if total_pixels == 0:
            percentage_foreground = 0.0
        else:
            percentage_foreground = (white_pixels / total_pixels) * 100

        # 7. Aturan Sederhana Keputusan
        decision = "SIGNATURE ABSENT"
        if percentage_foreground > 0.8:
            decision = "SIGNATURE PRESENT"

        # 8. Visualisasi
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Image: {os.path.basename(image_path)} - Decision: {decision} (Foreground: {percentage_foreground:.2f}%) ", fontsize=16)

        # Original Crop (BGR to RGB for matplotlib)
        axes[0].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('1. Original Crop')
        axes[0].axis('off')

        # Otsu Threshold
        axes[1].imshow(otsu_thresh, cmap='gray')
        axes[1].set_title('2. Otsu Threshold')
        axes[1].axis('off')

        # Adaptive Threshold
        axes[2].imshow(adaptive_thresh, cmap='gray')
        axes[2].set_title('3. Adaptive Threshold')
        axes[2].axis('off')

        # Morphological Cleaned
        axes[3].imshow(morph_cleaned_img, cmap='gray')
        axes[3].set_title('4. Morph Cleaned')
        axes[3].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
        plt.show()
        
        print(f"\nProcessed {os.path.basename(image_path)}: {decision} (Foreground Pixels: {percentage_foreground:.2f}%)\n")

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")


# Loop Pengujian Otomatis ke semua file gambar .jpg
print("Starting signature detection for all JPG images...")
image_files = [f for f in os.listdir('/content/') if f.lower().endswith('.jpg')]

if not image_files:
    print("No JPG image files found in /content/ directory.")
else:
    for img_file in image_files:
        uji_sistem_ttd(os.path.join('/content/', img_file))
    print("\nAll JPG images processed.")
