import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Focus on 4 specific weather categories
TARGET_CLASSES = ['lightning', 'rain', 'sandstorm', 'snow']

def extract_color_features(img_bgr):
    """
    Extracts color moments (mean, std dev) and normalized 16-bin histograms in HSV space.
    • Sandstorms -> Strong warm (yellow/brown) Hue values.
    • Snow/Lightning -> High Value (brightness) and low Saturation.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mean = np.mean(img_hsv, axis=(0, 1))
    std = np.std(img_hsv, axis=(0, 1))
    
    h_hist = cv2.calcHist([img_hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
    
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, s_hist)
    
    return np.hstack([mean, std, h_hist.flatten(), s_hist.flatten()])


def extract_glcm_features(img_gray):
    """
    Calculates Haralick spatial texture metrics using Gray-Level Co-occurrence Matrix (GLCM).
    • Smooth atmospheres (sandstorm) -> High Homogeneity.
    • Icy/crystalline patterns (snow) -> High Contrast and Energy.
    """
    img_q = (img_gray // 8).astype(np.uint8) # Quantize 256 gray levels down to 32
    
    glcm = graycomatrix(
        img_q, 
        distances=[1, 3], 
        angles=[0, np.pi/4, np.pi/2], 
        levels=32, 
        symmetric=True, 
        normed=True
    )
    
    return np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ])


def extract_lbp_features(img_gray, radius=1, n_points=8):
    """
    Extracts Local Binary Pattern (LBP) histogram to capture surface micro-textures.
    """
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_gabor_features(img_gray):
    """
    Filters image across 4 orientations (0°, 45°, 90°, 135°) to isolate directional streaks.
    • Falling rain and sharp lightning bolts produce high directional energy responses.
    """
    gabor_feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
        gabor_feats.append(filtered.mean())
        gabor_feats.append(filtered.std())
    return np.array(gabor_feats)


def extract_full_feature_vector(image_path, target_size=(128, 128)):
    """
    Master function: Loads an image, applies CLAHE contrast enhancement, extracts all 4 feature domains,
    and concatenates them into a unified 1D numerical array.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    color_feats = extract_color_features(img)
    glcm_feats = extract_glcm_features(gray_enhanced)
    lbp_feats = extract_lbp_features(gray_enhanced)
    gabor_feats = extract_gabor_features(gray_enhanced)
    
    return np.hstack([color_feats, glcm_feats, lbp_feats, gabor_feats])