# src/08_evaluate_il_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = '../data'
CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')
MODEL_PATH = '../saved_models/il_model.keras'
IMAGE_HEIGHT = 130
IMAGE_WIDTH = 600

def main():
    print(f"--- Loading Imitation Learning model from {MODEL_PATH} ---")
    
    try:
        model = load_model(MODEL_PATH, safe_mode=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("--- Loading dataset for evaluation ---")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset '{CSV_PATH}' not found.")
        return

    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Evaluating model on {len(test_df)} test samples...")

    y_true = []
    y_pred = []

    for index, row in test_df.iterrows():
        img_path = os.path.join(DATA_DIR, row['image_path'].strip())
        image = load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        true_action = int(row['action'])
        y_true.append(true_action)

        predictions = model.predict(image, verbose=0)
        predicted_action = np.argmax(predictions)
        y_pred.append(predicted_action)

    action_labels = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    # Define the full set of expected labels
    action_ids = [0, 1, 2, 3, 4]
    
    print("\n" + "="*50)
    print("           CLASSIFICATION METRICS REPORT")
    print("="*50)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")
    
    # UPDATED LINE: Added the 'labels' parameter to handle missing classes
    report = classification_report(y_true, y_pred, target_names=action_labels, labels=action_ids, zero_division=0)
    print(report)
    
    print("="*50)
    print("               CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred, labels=action_ids)
    print("Rows: True Action | Columns: Predicted Action\n")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=action_labels, yticklabels=action_labels)
    plt.title('Confusion Matrix for IL Model')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.show()

if __name__ == '__main__':
    main()