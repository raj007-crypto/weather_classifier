import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from feature_extractor import extract_full_feature_vector, TARGET_CLASSES

def load_dataset(dataset_dir):
    X, y = [], []
    label_map = {name: i for i, name in enumerate(TARGET_CLASSES)}
    
    for class_name in TARGET_CLASSES:
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory '{class_path}' not found. Skipping.")
            continue
            
        image_paths = glob.glob(os.path.join(class_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(class_path, "*.[pP][nN][gG]")) + \
                      glob.glob(os.path.join(class_path, "*.[jJ][pP][eE][gG]"))
        
        print(f"Processing '{class_name}': {len(image_paths)} images...")
        for img_p in image_paths:
            feats = extract_full_feature_vector(img_p)
            if feats is not None:
                X.append(feats)
                y.append(label_map[class_name])
                
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Path to dataset folder containing the 4 subfolders
    DATASET_PATH = r"C:\Users\rajar\OneDrive\study\dataset"
    
    print("=== 1. Extracting Handcrafted DIP Features ===")
    X, y = load_dataset(DATASET_PATH)
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Label Matrix Shape: {y.shape}")
    
    print("\n=== 2. Splitting and Scaling Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=== 3. Training Support Vector Classifier (SVM) ===")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Model Accuracy: {acc * 100:.2f}%\n")
    print("--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))
    
    # Save model artifacts
    joblib.dump(svm_model, 'weather_svm_model.pkl')
    joblib.dump(scaler, 'weather_scaler.pkl')
    print("Saved 'weather_svm_model.pkl' and 'weather_scaler.pkl' successfully!")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title("4-Class Weather Classification Confusion Matrix (DIP + SVM)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()