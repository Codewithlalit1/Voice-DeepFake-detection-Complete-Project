import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
REAL_PATH = 'data/real/'   # Folder with Human voices
FAKE_PATH = 'data/fake/'   # Folder with AI voices
MODEL_FILE = 'voice_detector_model.pkl' # Where to save the brain

# --- FEATURE EXTRACTION FUNCTION ---
def extract_features(file_path):
    """
    Loads an audio file and calculates its 'fingerprint' (MFCCs).
    """
    try:
        # Load audio (sr=None uses the file's original quality)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # Extract MFCCs (The complex physics part)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Average the features over time to get a single fingerprint
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- MAIN TRAINING LOOP ---
def start_training():
    features_list = []
    labels_list = []  # 0 = Real, 1 = Fake

    # 1. Load Real Data
    print(f"Scanning {REAL_PATH}...")
    if not os.path.exists(REAL_PATH):
        print(f"ERROR: Folder '{REAL_PATH}' not found.")
        return

    real_files = [f for f in os.listdir(REAL_PATH) if f.endswith('.wav')]
    print(f"Found {len(real_files)} Real files.")

    for filename in real_files:
        path = os.path.join(REAL_PATH, filename)
        data = extract_features(path)
        if data is not None:
            features_list.append(data)
            labels_list.append(0)  # LABEL: 0 is HUMAN

    # 2. Load Fake Data
    print(f"Scanning {FAKE_PATH}...")
    if not os.path.exists(FAKE_PATH):
        print(f"ERROR: Folder '{FAKE_PATH}' not found.")
        return

    fake_files = [f for f in os.listdir(FAKE_PATH) if f.endswith('.wav')]
    print(f"Found {len(fake_files)} Fake files.")

    for filename in fake_files:
        path = os.path.join(FAKE_PATH, filename)
        data = extract_features(path)
        if data is not None:
            features_list.append(data)
            labels_list.append(1)  # LABEL: 1 is AI

    # 3. Check if we have data
    if len(features_list) == 0:
        print("ERROR: No .wav files found! Did you convert them correctly?")
        return

    print(f"\nTraining on {len(features_list)} total files...")

    # 4. Prepare Data for AI
    X = np.array(features_list)
    y = np.array(labels_list)

    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train the Model (Random Forest)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("-" * 40)
    print(f"TRAINING COMPLETE!")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)
    
    # Detailed Report
    print(classification_report(y_test, predictions, target_names=['Real (Human)', 'Fake (AI)']))

    # 7. Save the Model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved as '{MODEL_FILE}'")

if __name__ == "__main__":
    start_training()