import cv2
import numpy as np

def process_microplastic(image_nparray, ppm=1.0):
    # 1. Convert the Streamlit image to OpenCV format
    img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding: Turn to Black & White (Binary)
    # We use Gaussian Blur to remove tiny "noise" spots first
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find Contours (The Outline)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, 0
    
    # Get the largest object in the image
    main_contour = max(contours, key=cv2.contourArea)
    
    # 4. Calculate Feret Diameter (Longest dimension)
    # We use a rotated bounding box to find the true length
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # The dimensions are width and height of the min-area rectangle
    width, height = rect[1]
    max_dim_pixels = max(width, height)
    
    # Convert to Micrometers (um)
    size_um = max_dim_pixels * ppm
    
    # Draw the box on the image for the UI
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    
    return img, thresh, size_um
