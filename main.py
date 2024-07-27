import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, label
from skimage.filters import median
from skimage.morphology import disk
from skimage.measure import regionprops

def load_image(image_path):
    """Load the image from the given path and convert to RGB."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read")
        print("Image loaded successfully.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image by converting to grayscale, thresholding, filling holes, and adjusting contrast."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresholded_image = gray_image > 20
    filled_image = cv2.morphologyEx(thresholded_image.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cleaned_image = remove_small_objects(filled_image.astype(bool), min_size=1000)
    preprocessed_image = (image * np.dstack([cleaned_image, cleaned_image, cleaned_image])).astype(np.uint8)
    enhanced_image = cv2.convertScaleAbs(preprocessed_image, alpha=1.5, beta=50)
    return gray_image, thresholded_image, filled_image, cleaned_image, enhanced_image

def detect_stones(enhanced_image):
    """Detect stones in the preprocessed image using median filtering and ROI masking."""
    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
    filtered_image = median(gray_image, disk(5))
    thresholded_image = filtered_image > 250

    r, c = thresholded_image.shape
    x1 = int(r / 2)
    y1 = int(c / 3)
    row = [x1, x1 + 200, x1 + 200, x1]
    col = [y1, y1, y1 + 40, y1 + 40]

    mask = np.zeros_like(thresholded_image, dtype=np.uint8)
    points = np.array([row, col]).T
    cv2.fillPoly(mask, [points], 1)

    roi_image = thresholded_image & mask.astype(bool)
    cleaned_roi_image = remove_small_objects(roi_image.astype(bool), min_size=4)
    labeled_array, number_of_stones = label(cleaned_roi_image, return_num=True)

    return labeled_array, number_of_stones, gray_image, filtered_image, thresholded_image, roi_image

def draw_detections(image, labeled_array):
    """Draw rectangles around detected stones in the original image."""
    for region in regionprops(labeled_array):
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(image, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
    return image

def display_images(images, titles):
    """Display a list of images with their corresponding titles."""
    plt.figure(figsize=(20, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 4, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_result(image, number_of_stones):
    """Display the final result with detection text."""
    if number_of_stones >= 1:
        detection_text = f"Stone is Detected. Number of stones: {number_of_stones}"
    else:
        detection_text = "No Stone is Detected"

    plt.imshow(image)
    plt.title('Processed Image')
    plt.text(10, 30, detection_text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.show()

def main(image_path):
    """Main function to load, preprocess, detect stones, and display the result."""
    image = load_image(image_path)
    if image is not None:
        gray_image, thresholded_image, filled_image, cleaned_image, enhanced_image = preprocess_image(image)
        labeled_array, number_of_stones, gray_image2, filtered_image, thresholded_image2, roi_image = detect_stones(enhanced_image)
        result_image = draw_detections(image, labeled_array)

        # Display all intermediate images
        images = [
            gray_image, thresholded_image, filled_image, cleaned_image,
            enhanced_image, gray_image2, filtered_image, roi_image
        ]
        titles = [
            'Grayscale Image', 'Thresholded Image', 'Filled Image', 'Cleaned Image',
            'Enhanced Image', 'Grayscale (Detection)', 'Filtered Image', 'ROI Masked Image'
        ]
        display_images(images, titles)
        display_result(result_image, number_of_stones)

if __name__ == "__main__":
    image_path = r'C:\Users\DELL\PycharmProjects\cgkedneystonedetection\images\img_3.png'
    main(image_path)
