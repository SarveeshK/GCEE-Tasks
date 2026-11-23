import cv2

# Read image
image = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------- FILTERING ----------------

gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
median = cv2.medianBlur(gray, 5)

# ---------------- EDGE DETECTION ----------------

sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
canny = cv2.Canny(gray, 100, 200)

# ---------------- THRESHOLDING ----------------

# Global threshold
_, global_threshold = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Otsu threshold
otsu_value, otsu = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(otsu_value)
# Adaptive threshold
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# ---------------- MORPHOLOGICAL OPERATIONS ----------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (5, 5)
)

erosion = cv2.erode(global_threshold, kernel)
dilation = cv2.dilate(global_threshold, kernel)

opening = cv2.morphologyEx(
    global_threshold,
    cv2.MORPH_OPEN,
    kernel
)

closing = cv2.morphologyEx(
    global_threshold,
    cv2.MORPH_CLOSE,
    kernel
)

# ---------------- DISPLAY ----------------

# Function to resize image for display
def resize_for_display(image, width=600):
    height = int(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height))


# Original and Grayscale
cv2.imshow("Original", resize_for_display(image))
cv2.imshow("Grayscale", resize_for_display(gray))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Filtering
cv2.imshow("Gaussian Blur", resize_for_display(gaussian))
cv2.imshow("Median Blur", resize_for_display(median))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Edge Detection
cv2.imshow("Sobel", resize_for_display(sobel))
cv2.imshow("Laplacian", resize_for_display(laplacian))
cv2.imshow("Canny", resize_for_display(canny))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Thresholding
cv2.imshow("Global Threshold", resize_for_display(global_threshold))
cv2.imshow("Otsu", resize_for_display(otsu))
cv2.imshow("Adaptive Threshold", resize_for_display(adaptive))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Morphological Operations
cv2.imshow("Erosion", resize_for_display(erosion))
cv2.imshow("Dilation", resize_for_display(dilation))
cv2.imshow("Opening", resize_for_display(opening))
cv2.imshow("Closing", resize_for_display(closing))
cv2.waitKey(0)
cv2.destroyAllWindows()