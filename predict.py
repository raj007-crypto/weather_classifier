import joblib
import cv2
import matplotlib.pyplot as plt
from feature_extractor import extract_full_feature_vector, TARGET_CLASSES

def predict_single_image(image_path, model_path='weather_svm_model.pkl', scaler_path='weather_scaler.pkl'):
    """
    Loads saved model weights, extracts features for a test image, and displays the predicted weather class.
    """
    # 1. Load saved model and standard scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print("Model or Scaler file not found. Run train.py first!")
        return
    
    # 2. Extract feature vector
    feats = extract_full_feature_vector(image_path)
    if feats is None:
        print(f"Error loading image at path: {image_path}")
        return
    
    # 3. Scale feature vector (using saved training parameters)
    feats_scaled = scaler.transform([feats])
    
    # 4. Predict class index
    pred_idx = model.predict(feats_scaled)[0]
    predicted_class = TARGET_CLASSES[pred_idx]
    
    # 5. Display visual result
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"Predicted Weather: {predicted_class.upper()}", fontsize=14, color='darkblue')
    plt.axis('off')
    plt.show()
    
    return predicted_class

if __name__ == "__main__":
    # Test on any sample image path
    TEST_IMAGE_PATH = r"C:\Users\rajar\OneDrive\study\dataset\lightning\2206.jpg"
    
    prediction = predict_single_image(TEST_IMAGE_PATH)
    print(f"Result for '{TEST_IMAGE_PATH}': {prediction}")